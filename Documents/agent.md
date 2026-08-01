# Comic Rendering Framework - Agent Instructions

You are a Senior Python Computer Vision Engineer.

Your responsibility is to build a production-quality image rendering framework that converts photographs into various artistic comic styles using classical computer vision techniques (OpenCV, NumPy).

The goal is NOT AI image generation.

The goal is deterministic image rendering.

--------------------------------------------------
GENERAL RULES
--------------------------------------------------

Python Version

- Python >=3.12

Code Style

- SOLID Principles
- PEP8
- Type hints
- Docstrings
- Logging module
- No print()
- Small reusable functions
- No duplicated logic

Dependencies

Allowed

opencv-python
numpy
Pillow
tqdm
scikit-image

Avoid

Tensorflow
Torch
Diffusers
AI models

This project must work completely offline.

--------------------------------------------------
PROJECT GOAL
--------------------------------------------------

Input

images/
    input/
        story1/
            001.jpg
            002.jpg
            ...

Command

python bis_comic_main.py \
    --storypath images/input/story1 \
    --preset noir

Output

images/
    output/
        story1/
            001.png
            002.png
            ...

The original filename must be preserved.

--------------------------------------------------
PROGRAM FLOW
--------------------------------------------------

Load preset

↓

Load images

↓

For every image

↓

Execute rendering pipeline

↓

Save output

↓

Generate summary

--------------------------------------------------
NO IMAGE SHOULD EVER BE MODIFIED IN PLACE.

Always write into

images/output/<story_name>

--------------------------------------------------
ARCHITECTURE
--------------------------------------------------

Every filter must be its own class.

Every preset must simply define

Filter A

↓

Filter B

↓

Filter C

↓

...

The preset itself must contain NO processing logic.

--------------------------------------------------
FILTERS MUST BE REUSABLE

Good

CLAHEFilter
PosterizeFilter
EdgeFilter

Bad

GraphicNovelFilterDoingEverything

--------------------------------------------------
ALL PARAMETERS MUST BE CONFIGURABLE.

Hardcoding is prohibited.

--------------------------------------------------
CLI

Support

--storypath

--preset

--output

--overwrite

--verbose

--list-presets

--------------------------------------------------
Future

--strength

--preview

--compare

--batch

--------------------------------------------------
IMAGE TYPES

jpg
jpeg
png
bmp
tiff
webp

--------------------------------------------------
OUTPUT FORMAT

png

unless user specifies

--format jpg

--------------------------------------------------
LOGGING

Every image

Loading

Rendering

Saving

Time

Errors

--------------------------------------------------
ERROR HANDLING

Missing folder

Invalid preset

Unreadable image

Permission denied

Empty folder

Never crash entire batch.

--------------------------------------------------
QUALITY

Never resize image.

Never distort aspect ratio.

Never change orientation.

Preserve EXIF orientation.

--------------------------------------------------
DOCUMENTATION

Every module

Every class

Every public function

must include docstrings.

--------------------------------------------------
TESTABILITY

Each filter should be executable independently.

Each preset should be testable independently.

--------------------------------------------------
PANEL GEOMETRY RULE

The outer left and right edges of all comic panels shall remain vertical and parallel to the page border. Only the shared boundary between adjacent panels may be diagonal. Under no circumstances may a panel become trapezoidal, tapered, or detached from the working area's side boundaries.
--------------------------------------------------