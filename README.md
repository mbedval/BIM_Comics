# BIS Comic Renderer

A production-quality, classical computer vision framework to transform digital photographs into hand-drawn, print, and graphic novel comic styles.

> [!IMPORTANT]
> **No AI / Generative Models.** This library is built entirely on deterministic computer vision and classical image processing techniques (OpenCV, NumPy, scikit-image). Renders are reproducible, fast, and run completely offline.

---

## Installation

Ensure you have Python 3.12+ installed.

### Install from Source
To install the package and its CLI tool:
```bash
pip install .
```

### Install Dependencies Manually
Alternatively, install requirements:
```bash
pip install -r requirements.txt
```

---

## Command Line Interface (CLI)

The package provides a command-line script to batch-process folders of images (stories).

```bash
# Basic usage
python bis_comic_main.py --storypath images/scene/story1 --preset graphic_novel

# Overwrite existing renders, process with 4 workers in parallel, print debug details
python bis_comic_main.py --storypath images/scene/story1 --preset noir --overwrite --jobs 4 --verbose
```

### CLI Arguments Reference

| Argument | Short | Default | Description |
|---|---|---|---|
| `--storypath DIR` | | (Required) | Directory containing input story images (`.jpg`, `.png`, etc.). |
| `--preset NAME` | | `graphic_novel` | Processing preset configuration to apply. |
| `--output DIR` | | `images/output` | Destination output directory. |
| `--overwrite` | | `False` | Overwrite existing output files (default skips processed images). |
| `--verbose` | | `False` | Enable DEBUG level console logging (disables progress bar). |
| `--list-presets` | | `False` | List all discovered preset names and exit. |
| `--jobs INT` | `-j` | `1` | Number of parallel worker processes. `0` uses all available CPUs. |

---

## Configurable Presets

A preset is defined as a JSON configuration file inside the `comic_renderer/presets/` folder. It outlines an ordered sequence of filters with their corresponding parameters.

Supported presets include:
- `graphic_novel`: Realistic grayscale comic with local contrast, bilateral smoothing, Canny outlines, and overlay blending.
- `noir`: Deep shadows, high contrast, heavy outlines, and multiply-blending.
- `sin_city`: Extreme high-contrast black-and-white.
- `manga_gray`: Clean manga lines and soft gray tones.
- `newspaper`: Grayscale print look using rotated halftone screens.
- `vintage_print`: Aged paper texture and warm sepia borders.
- `ink_wash`: Smooth watercolor-like gradients with soft Sobel bounds.
- `woodcut`: Coarse woodblock relief texture look.
- `posterized`: Multi-level posterized pop art.
- `pencil_comic`: Soft sketch-like look.

---

## Filters Reference

The framework includes 12 modular filters that inherit from `BaseFilter`.

| Filter Key | Class Name | Main Purpose |
|---|---|---|
| `grayscale` | `GrayscaleFilter` | Desaturates images (luminance, lightness, average). |
| `clahe` | `CLAHEFilter` | Color-preserving adaptive histogram equalization. |
| `bilateral` | `BilateralFilter` | Edge-preserving smoothing to cartoonist flat shapes. |
| `posterize` | `PosterizeFilter` | Floor quantization to discrete color/tone steps. |
| `edge` | `EdgeFilter` | Canny, Sobel, or Laplacian edge extraction and burning. |
| `contrast` | `ContrastFilter` | Affine scaling of pixel values (`alpha * I + beta`). |
| `sharpen` | `SharpenFilter` | Gaussian residual unsharp masking. |
| `texture` | `TextureFilter` | Noise grain or multi-scale paper fiber overlay. |
| `blend` | `BlendFilter` | Photoshop self-blending (multiply, screen, overlay, soft-light). |
| `morphology` | `MorphologyFilter` | Dilate, erode, close, open shape operations. |
| `tonecurve` | `ToneCurveFilter` | Custom control-point piecewise lookup table mapping. |
| `vignette` | `VignetteFilter` | Centered radial color border fall-off. |
| `halftone` | `HalftoneFilter` | Rotated digital halftone screen dot rendering. |

---

## Programmatic Python Usage

You can build custom processing pipelines programmatically in Python without the CLI:

```python
from pathlib import Path
import numpy as np
from PIL import Image
from comic_renderer.filters.grayscale import GrayscaleFilter
from comic_renderer.filters.clahe import CLAHEFilter
from comic_renderer.filters.edge import EdgeFilter

# Load image
img = np.array(Image.open("001.jpg"))

# Instantiate filters
gray = GrayscaleFilter(params={"method": "luminance"})
clahe = CLAHEFilter(params={"clip_limit": 2.0})
edge = EdgeFilter(params={"method": "canny", "blend_strength": 0.8})

# Apply filters sequentially
img = gray.apply(img)
img = clahe.apply(img)
img = edge.apply(img)

# Save result
Image.fromarray(img).save("out.png")
```

---

## Testing

Run the full automated test suite containing 384 tests:
```bash
python3 -m pytest
```
