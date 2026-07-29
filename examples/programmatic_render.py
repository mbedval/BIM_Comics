"""Example script showing programmatic usage of the BIS Comic Renderer framework.

This script runs the processing pipeline on an image programmatically by instantiating
filters directly, bypassing the command line interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to python path to allow running directly from any directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image

from comic_renderer.filters.bilateral import BilateralFilter
from comic_renderer.filters.clahe import CLAHEFilter
from comic_renderer.filters.edge import EdgeFilter
from comic_renderer.filters.grayscale import GrayscaleFilter
from comic_renderer.filters.texture import TextureFilter
from comic_renderer.filters.vignette import VignetteFilter


def generate_synthetic_image(h: int = 400, w: int = 600) -> np.ndarray:
    """Generate a colorful synthetic test image with geometric shapes."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Background gradient
    for y in range(h):
        img[y, :, 0] = int(y / h * 200)
        img[y, :, 1] = int((h - y) / h * 150)
        img[y, :, 2] = 100

    # Draw shapes
    # White circle in the center
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    y_idx, x_idx = np.indices((h, w))
    dist = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    img[dist < r] = [240, 240, 240]

    # Red rectangle
    img[h//4:3*h//4, w//8:w//4] = [220, 30, 30]

    # Blue rectangle
    img[h//3:2*h//3, 3*w//4:7*w//8] = [30, 30, 220]

    return img


def main() -> None:
    # 1. Define paths
    input_path = Path("images/input/story1/mb.jpg")
    output_dir = Path("images/output/programmatic")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "programmatic_render.png"

    print("Step 1: Loading or generating input image...")
    if input_path.exists():
        print(f"  Loading existing image: {input_path}")
        image = np.array(Image.open(input_path).convert("RGB"))
    else:
        print("  Input image not found. Generating a synthetic test image...")
        image = generate_synthetic_image()

    print(f"  Input image shape: {image.shape}, dtype: {image.dtype}")

    # 2. Instantiate individual filters programmatically
    print("\nStep 2: Instantiating filters...")
    gray_filter = GrayscaleFilter(params={"method": "luminance"})
    clahe_filter = CLAHEFilter(params={"clip_limit": 2.5})
    bilateral_filter = BilateralFilter(params={"diameter": 9, "sigma_color": 75.0, "sigma_space": 75.0, "iterations": 2})
    edge_filter = EdgeFilter(params={"method": "canny", "low_threshold": 30, "high_threshold": 95, "blend_strength": 0.85})
    texture_filter = TextureFilter(params={"mode": "paper", "strength": 0.12, "seed": 99})
    vignette_filter = VignetteFilter(params={"strength": 0.5, "radius": 1.1, "color": [0, 0, 0]})

    # 3. Apply the filters sequentially
    print("\nStep 3: Running pipeline filters programmatically...")
    
    print("  Applying GrayscaleFilter...")
    gray_img = gray_filter.apply(image)

    print("  Applying CLAHEFilter...")
    clahe_img = clahe_filter.apply(gray_img)

    print("  Applying BilateralFilter...")
    bilateral_img = bilateral_filter.apply(clahe_img)

    print("  Applying EdgeFilter...")
    edged_img = edge_filter.apply(bilateral_img)

    print("  Applying TextureFilter...")
    textured_img = texture_filter.apply(edged_img)

    print("  Applying VignetteFilter...")
    final_img = vignette_filter.apply(textured_img)

    # 4. Save the result
    print(f"\nStep 4: Saving rendered image to: {output_path}")
    Image.fromarray(final_img).save(output_path)
    print("  ✓ Render complete and saved successfully.")


if __name__ == "__main__":
    main()
