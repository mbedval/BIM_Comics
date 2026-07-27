#!/usr/bin/env python3
"""make_comic_page.py – CLI for generating a 4-panel comic page.

Usage
-----
    python3 make_comic_page.py \\
        --storypath images/scene/story_test \\
        --preset cartoon \\
        --output images/output/comic_page.png

    # Skip background removal (faster, for quick previews):
    python3 make_comic_page.py \\
        --storypath images/scene/story_test \\
        --preset cartoon \\
        --no-bg-removal \\
        --output images/output/comic_page_preview.png

    # Use specific images instead of a storypath:
    python3 make_comic_page.py \\
        --images img1.jpg img2.jpg img3.jpg img4.jpg \\
        --preset cartoon \\
        --output images/output/comic_page.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("make_comic_page")

# Supported image extensions
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_images(storypath: Path, limit: int = 4) -> list[Path]:
    """Collect the first *limit* images from *storypath* (sorted by name)."""
    images = sorted(
        p for p in storypath.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    )
    if not images:
        logger.error("No images found in: %s", storypath)
        sys.exit(1)
    if len(images) < limit:
        logger.warning(
            "Found %d image(s) in '%s', need %d. "
            "Images will be reused to fill remaining panels.",
            len(images), storypath, limit,
        )
        # Cycle images to fill 4 panels
        while len(images) < limit:
            images += images
        images = images[:limit]
    return images[:limit]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a 4-panel comic page from source images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--storypath",
        type=Path,
        metavar="DIR",
        help="Directory containing source images (first 4 used).",
    )
    source.add_argument(
        "--images",
        nargs=4,
        type=Path,
        metavar="IMG",
        help="Exactly 4 image paths to use as panels.",
    )

    p.add_argument(
        "--preset",
        default="cartoon",
        help="Preset pipeline to apply to each image before compositing.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("images/output/comic_page.png"),
        help="Output path for the generated comic page.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1400,
        help="Comic page canvas width in pixels.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=1000,
        help="Comic page canvas height in pixels.",
    )
    p.add_argument(
        "--border",
        type=int,
        default=8,
        help="Border / divider thickness in pixels.",
    )
    p.add_argument(
        "--no-bg-removal",
        action="store_true",
        dest="no_bg_removal",
        help="Skip background removal (faster, for previews).",
    )
    p.add_argument(
        "--no-preset",
        action="store_true",
        dest="no_preset",
        help="Skip the preset pipeline (use raw source images).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Resolve image list
    if args.images:
        image_paths: list[Path] = list(args.images)
        for p in image_paths:
            if not p.exists():
                logger.error("Image not found: %s", p)
                sys.exit(1)
    else:
        storypath = args.storypath
        if not storypath.is_dir():
            logger.error("storypath is not a directory: %s", storypath)
            sys.exit(1)
        image_paths = _collect_images(storypath, limit=4)

    preset_name = None if args.no_preset else args.preset

    logger.info("Comic page generator")
    logger.info("  Images   : %s", [p.name for p in image_paths])
    logger.info("  Preset   : %s", preset_name or "(none)")
    logger.info("  BG remove: %s", not args.no_bg_removal)
    logger.info("  Output   : %s", args.output)
    logger.info("  Canvas   : %dx%d  border=%dpx", args.width, args.height, args.border)

    # Import builder here so startup is fast for --help
    from comic_renderer.compositor.page_builder import ComicPageBuilder

    builder = ComicPageBuilder(
        preset_name=preset_name,
        page_width=args.width,
        page_height=args.height,
        border=args.border,
        skip_bg_removal=args.no_bg_removal,
    )

    builder.build(image_paths, args.output)
    logger.info("✓ Done! Comic page saved to: %s", args.output)


if __name__ == "__main__":
    main()
