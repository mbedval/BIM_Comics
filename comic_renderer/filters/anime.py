"""AnimeFilter – style transfer filter using AnimeGANv2 ONNX models.

Applies pre-trained AnimeGANv2 generator models to stylize photographic
images into hand-drawn animation styles.

JSON preset usage
-----------------
.. code-block:: json

    {
        "filter": "anime",
        "params": {
            "model_path": "comic_renderer/weights/AnimeGANv2_Hayao.onnx",
            "model_url": "https://huggingface.co/vumichien/AnimeGANv2_Hayao/resolve/main/AnimeGANv2_Hayao.onnx",
            "max_dim": 1024
        }
    }
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from comic_renderer.filters.base import BaseFilter

logger = logging.getLogger(__name__)


class AnimeFilter(BaseFilter):
    """Transform image into anime artwork using AnimeGANv2 ONNX.

    Parameters
    ----------
    params:
        model_path : str, default "comic_renderer/weights/AnimeGANv2_Hayao.onnx"
            Local filepath to the pre-converted ONNX model.
        model_url : str, default "https://huggingface.co/vumichien/AnimeGANv2_Hayao/resolve/main/AnimeGANv2_Hayao.onnx"
            URL to download the model from if not found locally.
        max_dim : int, default 1024
            Maximum dimension (width or height) to resize to for inference,
            preventing out-of-memory or slow performance. Must be >= 32.
    """

    FILTER_NAME: str = "anime"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        self._model_path: str = str(
            self._params.get("model_path", "comic_renderer/weights/AnimeGANv2_Hayao.onnx")
        )
        self._model_url: str = str(
            self._params.get(
                "model_url",
                "https://huggingface.co/vumichien/AnimeGANv2_Hayao/resolve/main/AnimeGANv2_Hayao.onnx",
            )
        )
        self._max_dim: int = int(self._params.get("max_dim", 1024))
        if self._max_dim < 32:
            raise ValueError(f"AnimeFilter: max_dim must be >= 32, got {self._max_dim!r}")

        # Lazily initialized in apply to avoid multiprocessing serialization errors
        self._session: ort.InferenceSession | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Exclude ONNX InferenceSession from serialization state."""
        state = self.__dict__.copy()
        state["_session"] = None
        return state

    def _ensure_model_exists(self) -> None:
        """Check if model exists locally, otherwise download it."""
        path = Path(self._model_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Anime model not found at %s. Downloading from %s...", path, self._model_url)
            try:
                urllib.request.urlretrieve(self._model_url, str(path))
                logger.info("Download completed successfully.")
            except Exception as e:
                logger.error("Failed to download anime model: %s", e)
                raise RuntimeError(f"Anime model missing and download failed: {e}") from e

    def _get_session(self) -> ort.InferenceSession:
        """Initialize and return ONNX session with configuration safe for sandboxes."""
        if self._session is None:
            self._ensure_model_exists()
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(self._model_path, sess_options=opts)
        return self._session

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply AnimeGANv2 ONNX style transfer.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Anime-styled RGB ``uint8`` array ``(H, W, 3)``.
        """
        h, w, c = image.shape

        # Compute optimal processing dimensions as multiples of 32
        target_h = max(32, (h // 32) * 32)
        target_w = max(32, (w // 32) * 32)

        # Scale down if exceeding maximum dimension limit
        if target_h > self._max_dim or target_w > self._max_dim:
            if target_h > target_w:
                target_w = int(target_w * self._max_dim / target_h)
                target_h = self._max_dim
            else:
                target_h = int(target_h * self._max_dim / target_w)
                target_w = self._max_dim
            target_h = max(32, (target_h // 32) * 32)
            target_w = max(32, (target_w // 32) * 32)

        # Preprocess input image to BHWC float32 in range [-1.0, 1.0]
        input_img = cv2.resize(image, (target_w, target_h))
        input_img = input_img.astype(np.float32)
        input_img = input_img / 127.5 - 1.0
        input_img = np.expand_dims(input_img, axis=0)

        # Run ONNX inference
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        out = session.run([output_name], {input_name: input_img})[0]

        # Postprocess output
        out = np.squeeze(out, axis=0) # [1, H, W, 3] -> [H, W, 3]
        out = (out + 1.0) * 127.5
        out = np.clip(out, 0, 255).astype(np.uint8)

        # Restore original image size
        if out.shape[0] != h or out.shape[1] != w:
            out = cv2.resize(out, (w, h))

        return out
