"""BIS Comic Renderer – main CLI entry point.

Usage
-----
    python bis_comic_main.py --storypath images/scene/story1 --preset graphic_novel

Run ``python bis_comic_main.py --help`` for the full option reference.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing
import sys
import time
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from comic_renderer.config.settings import IOConfig, LoggingConfig, RunConfig
from comic_renderer.filters.bilateral import BilateralFilter
from comic_renderer.filters.blend import BlendFilter
from comic_renderer.filters.clahe import CLAHEFilter
from comic_renderer.filters.contrast import ContrastFilter
from comic_renderer.filters.edge import EdgeFilter
from comic_renderer.filters.grayscale import GrayscaleFilter
from comic_renderer.filters.halftone import HalftoneFilter
from comic_renderer.filters.morphology import MorphologyFilter
from comic_renderer.filters.passthrough import PassThroughFilter
from comic_renderer.filters.posterize import PosterizeFilter
from comic_renderer.filters.registry import FilterRegistry
from comic_renderer.filters.sharpen import SharpenFilter
from comic_renderer.filters.texture import TextureFilter
from comic_renderer.filters.tonecurve import ToneCurveFilter
from comic_renderer.filters.vignette import VignetteFilter
from comic_renderer.io.discovery import StoryDiscovery
from comic_renderer.io.loader import load_image
from comic_renderer.io.writer import ImageWriter
from comic_renderer.pipeline.executor import PipelineExecutor
from comic_renderer.pipeline.models import PresetConfig
from comic_renderer.pipeline.preset_loader import PresetLoader
from comic_renderer.utils.logging_config import setup_logging

# ---------------------------------------------------------------------------
# Default values (no magic literals elsewhere)
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_ROOT = Path("images/output")
_DEFAULT_PRESET = "graphic_novel"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# The presets directory lives alongside the comic_renderer package.
_PRESETS_DIR: Path = Path(__file__).parent / "presets"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser.

    All future-compatibility flags (``--compare``, ``--strength``,
    ``--preview``) are registered here but marked as *not yet implemented*
    so that the interface is stable from day one.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="bis_comic_main.py",
        description=(
            "BIS Comic Renderer – converts photographs into comic styles "
            "using classical computer vision algorithms."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python bis_comic_main.py --storypath images/scene/story1 --preset graphic_novel\n"
            "  python bis_comic_main.py --list-presets\n"
        ),
    )

    # --- Required / primary ---
    parser.add_argument(
        "--storypath",
        metavar="DIR",
        type=Path,
        help="Directory containing source images (e.g. images/scene/story1).",
    )
    parser.add_argument(
        "--preset",
        metavar="NAME",
        default=_DEFAULT_PRESET,
        help=(
            f"Name of the rendering preset to apply "
            f"(default: {_DEFAULT_PRESET!r}). "
            f"Use --list-presets to see all available presets."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help=(
            f"Root output directory (default: {_DEFAULT_OUTPUT_ROOT}). "
            "A sub-directory named after the story is created automatically."
        ),
    )

    # --- Behaviour modifiers ---
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files (default: skip them).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel worker processes. 0 means use all available CPUs.",
    )

    # --- Informational ---
    parser.add_argument(
        "--list-presets",
        action="store_true",
        default=False,
        dest="list_presets",
        help="List all available presets and exit.",
    )

    # --- Future-compatibility flags (not yet implemented) ---
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="[FUTURE] Save a side-by-side comparison image.",
    )
    parser.add_argument(
        "--strength",
        metavar="FLOAT",
        type=float,
        default=1.0,
        help="[FUTURE] Global pipeline effect strength multiplier (0.0–1.0).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=False,
        help="[FUTURE] Display each processed image in a window.",
    )

    return parser


# ---------------------------------------------------------------------------
# Preset listing
# ---------------------------------------------------------------------------


def list_presets() -> None:
    """Print all available preset names (discovered dynamically from JSON files)."""
    try:
        loader = PresetLoader(presets_dir=_PRESETS_DIR)
        names = loader.list_available()
    except FileNotFoundError:
        names = []
    print("Available presets:")
    if names:
        for name in names:
            print(f"  • {name}")
    else:
        print("  (no presets found)")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate parsed arguments and exit with a helpful message on failure."""
    if args.list_presets:
        return  # No further validation needed.

    if args.storypath is None:
        parser.error("--storypath is required unless --list-presets is used.")

    # Validate the preset name against JSON files on disk.
    try:
        loader = PresetLoader(presets_dir=_PRESETS_DIR)
        available = loader.list_available()
    except FileNotFoundError:
        available = []

    if args.preset not in available:
        parser.error(
            f"Unknown preset {args.preset!r}. "
            f"Available presets: {available}. "
            f"Run --list-presets for details."
        )

    if args.compare:
        logger.warning("--compare is not yet implemented; flag ignored.")
    if args.preview:
        logger.warning("--preview is not yet implemented; flag ignored.")
    if args.strength != 1.0:
        logger.warning("--strength is not yet implemented; flag ignored.")

    if hasattr(args, "jobs") and args.jobs < 0:
        parser.error("--jobs / -j must be a non-negative integer (0 or greater).")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    """Construct a :class:`RunConfig` from parsed CLI arguments."""
    io_config = IOConfig(
        story_path=args.storypath.resolve(),
        output_root=args.output.resolve(),
        overwrite=args.overwrite,
    )
    logging_config = LoggingConfig(verbose=args.verbose)
    return RunConfig(
        io=io_config,
        logging=logging_config,
        preset_name=args.preset,
        jobs=args.jobs,
    )


def _build_registry() -> FilterRegistry:
    """Construct and populate the filter registry.

    Every concrete filter must be registered here.  Future milestones add
    new filter imports and ``registry.register(...)`` calls below.

    Returns
    -------
    FilterRegistry
        Fully populated registry ready for use by the executor.
    """
    registry = FilterRegistry()
    # Milestone 2
    registry.register(PassThroughFilter)
    # Milestone 3
    registry.register(GrayscaleFilter)
    registry.register(CLAHEFilter)
    registry.register(PosterizeFilter)
    registry.register(SharpenFilter)
    registry.register(ContrastFilter)
    # Milestone 4
    registry.register(EdgeFilter)
    registry.register(TextureFilter)
    registry.register(BlendFilter)
    registry.register(MorphologyFilter)
    # Milestone 5
    registry.register(BilateralFilter)
    # Milestone 6
    registry.register(ToneCurveFilter)
    registry.register(VignetteFilter)
    registry.register(HalftoneFilter)
    return registry


def _load_preset(preset_name: str) -> PresetConfig:
    """Load the named preset from the JSON presets directory.

    Parameters
    ----------
    preset_name:
        Bare name such as ``"graphic_novel"``.

    Returns
    -------
    PresetConfig
        Fully parsed and validated preset.

    Raises
    ------
    FileNotFoundError
        When the preset JSON file cannot be found.
    ValueError
        When the preset JSON is structurally invalid.
    """
    loader = PresetLoader(presets_dir=_PRESETS_DIR)
    return loader.load(preset_name)


def _process_image_worker(
    img_path: Path,
    run_config: RunConfig,
    preset_name: str,
) -> tuple[str, bool, str | None, float]:
    """Worker process target to run the pipeline on a single image.

    This function does NOT print or log to avoid race conditions. Instead,
    it returns the status so the main process can log properly.
    """
    import time
    from comic_renderer.bis_comic_main import _build_registry, _load_preset
    from comic_renderer.io.loader import load_image
    from comic_renderer.io.writer import ImageWriter
    from comic_renderer.pipeline.executor import PipelineExecutor

    start_time = time.perf_counter()
    try:
        preset = _load_preset(preset_name)
        registry = _build_registry()
        executor = PipelineExecutor(registry=registry)
        writer = ImageWriter(io_config=run_config.io)

        image = load_image(img_path)
        processed = executor.run(image, preset)
        output_path = writer.write(processed, img_path)
        elapsed = time.perf_counter() - start_time
        return img_path.name, True, str(output_path.name), elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        return img_path.name, False, str(exc), elapsed


def process_story(run_config: RunConfig) -> int:
    """Discover, pipeline-process, and save every image in the story directory.

    The pipeline is driven by the JSON preset selected via ``--preset``.
    Each image is loaded, passed through the full filter chain defined in
    the preset, and written to the output directory as PNG.

    Parameters
    ----------
    run_config:
        Fully constructed runtime configuration.

    Returns
    -------
    int
        Exit code: ``0`` on success, ``1`` if any image failed.
    """
    io_cfg = run_config.io
    logger.info("Story path  : %s", io_cfg.story_path)
    logger.info("Output root : %s", io_cfg.output_story_dir())
    logger.info("Preset      : %s", run_config.preset_name)
    logger.info("Overwrite   : %s", io_cfg.overwrite)

    # --- Load preset (fail fast before discovering images) ---
    try:
        preset = _load_preset(run_config.preset_name)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load preset '%s': %s", run_config.preset_name, exc)
        return 1

    logger.info(
        "Preset loaded: '%s' (%d step(s)).", preset.name, len(preset.steps)
    )

    # --- Build the filter registry and executor ---
    registry = _build_registry()
    executor = PipelineExecutor(registry=registry)

    # --- Discover images ---
    discovery = StoryDiscovery(io_config=io_cfg)
    try:
        image_paths = discovery.discover()
    except FileNotFoundError as exc:
        logger.error("Discovery failed: %s", exc)
        return 1

    if not image_paths:
        logger.warning(
            "No supported images found in '%s'. Nothing to do.", io_cfg.story_path
        )
        return 0

    logger.info("Found %d image(s) to process.", len(image_paths))

    # --- Process each image ---
    errors: list[str] = []
    total_start = time.perf_counter()
    num_images = len(image_paths)

    # Determine parallel vs sequential
    num_workers = run_config.jobs
    if num_workers <= 0:
        num_workers = multiprocessing.cpu_count()

    use_parallel = num_workers > 1 and num_images > 1

    if use_parallel:
        logger.info("Spawning %d worker process(es) for parallel rendering.", num_workers)
        
        with tqdm(
            total=num_images,
            desc="Processing story",
            disable=run_config.logging.verbose,
            unit="img",
        ) as pbar:
            with logging_redirect_tqdm():
                with ProcessPoolExecutor(max_workers=num_workers) as pool:
                    futures = {
                        pool.submit(
                            _process_image_worker,
                            img_path,
                            run_config,
                            run_config.preset_name
                        ): img_path
                        for img_path in image_paths
                    }
                    
                    for idx, future in enumerate(as_completed(futures), start=1):
                        img_name, success, detail, elapsed = future.result()
                        pbar.update(1)
                        
                        if success:
                            logger.info(
                                "[%d/%d] ✓ Processed '%s' → '%s' (%.2fs)",
                                idx,
                                num_images,
                                img_name,
                                detail,
                                elapsed,
                            )
                        else:
                            logger.error(
                                "[%d/%d] ✗ Pipeline error for '%s': %s",
                                idx,
                                num_images,
                                img_name,
                                detail,
                            )
                            errors.append(detail)
    else:
        # Sequential processing
        writer = ImageWriter(io_config=io_cfg)
        with tqdm(
            total=num_images,
            desc="Processing story",
            disable=run_config.logging.verbose,
            unit="img",
        ) as pbar:
            with logging_redirect_tqdm():
                for idx, img_path in enumerate(image_paths, start=1):
                    if run_config.logging.verbose:
                        logger.info(
                            "[%d/%d] Processing '%s' with preset '%s' …",
                            idx,
                            num_images,
                            img_path.name,
                            run_config.preset_name,
                        )
                    img_start = time.perf_counter()

                    try:
                        # LOAD
                        logger.debug("  → Loading …")
                        image = load_image(img_path)

                        # PIPELINE
                        logger.debug("  → Running pipeline (%d step(s)) …", len(preset.steps))
                        processed = executor.run(image, preset)

                        # SAVE
                        logger.debug("  → Saving …")
                        output_path = writer.write(processed, img_path)
                        elapsed = time.perf_counter() - img_start
                        logger.info(
                            "  ✓ Saved '%s' (%.2fs)", output_path.name, elapsed
                        )

                    except FileNotFoundError as exc:
                        logger.error("  ✗ Load error for '%s': %s", img_path.name, exc)
                        errors.append(str(exc))
                    except ValueError as exc:
                        logger.error("  ✗ Decode/pipeline error for '%s': %s", img_path.name, exc)
                        errors.append(str(exc))
                    except KeyError as exc:
                        logger.error("  ✗ Pipeline config error for '%s': %s", img_path.name, exc)
                        errors.append(str(exc))
                    except RuntimeError as exc:
                        logger.error("  ✗ Write error for '%s': %s", img_path.name, exc)
                        errors.append(str(exc))
                    
                    pbar.update(1)

    total_elapsed = time.perf_counter() - total_start
    logger.info(
        "Done. %d image(s) processed in %.2fs. %d error(s).",
        num_images - len(errors),
        total_elapsed,
        len(errors),
    )

    return 1 if errors else 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, configure logging, and run the story processor.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Shell exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Bootstrap logging before any other work so that validation warnings
    # are emitted with the correct format.
    setup_logging(verbose=args.verbose)

    # --list-presets
    if args.list_presets:
        list_presets()
        return 0

    _validate_args(args, parser)

    run_config = _build_run_config(args)
    return process_story(run_config)


if __name__ == "__main__":
    sys.exit(main())
