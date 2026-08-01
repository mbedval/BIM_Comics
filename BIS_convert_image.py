"""Project-root entry point for BIS Comic Renderer.

This thin wrapper exists so that the user can run:

    python BIS_convert_image.py --storypath images/input/story1 --preset noir

from the project root without having to set PYTHONPATH or use
``python -m comic_renderer.bis_comic_main``.
"""

import sys
from comic_renderer.bis_comic_main import main

if __name__ == "__main__":
    sys.exit(main())
