"""Project-root entry point for BIS Comic Renderer.

This thin wrapper exists so that the user can run:

    python bis_comic_main.py --storypath images/scene/story1 --preset graphic_novel

from the project root without having to set PYTHONPATH or use
``python -m comic_renderer.bis_comic_main``.
"""

import sys
from comic_renderer.bis_comic_main import main

if __name__ == "__main__":
    sys.exit(main())
