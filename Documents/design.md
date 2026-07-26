# Comic Rendering Framework Design

## Goal

Convert photographs into artistic grayscale comic illustrations using classical computer vision.

No AI.

Deterministic output.

Modular architecture.

--------------------------------------------------

Project

comic_framework/

    bis_comic_main.py

    presets/

        graphic_novel.py

        noir.py

        manga.py

        pencil.py

        watercolor.py

        sketch.py

    filters/

        grayscale.py

        bilateral.py

        clahe.py

        posterize.py

        edge.py

        morphology.py

        sharpen.py

        contrast.py

        texture.py

        tonecurve.py

        halftone.py

        vignette.py

        blend.py

    pipeline/

        renderer.py

        preset_loader.py

        pipeline_executor.py

    io/

        image_loader.py

        image_writer.py

    utils/

        logger.py

        file_utils.py

        timer.py

        config.py

    config/

        presets/

            graphic_novel.json

            noir.json

            manga.json

            ...

--------------------------------------------------

Preset

A preset is only

Ordered list of filters

plus parameters.

Example

GraphicNovel

Grayscale

↓

CLAHE

↓

Bilateral

↓

Posterize

↓

Edge Detection

↓

Contrast

↓

Sharpen

↓

Texture

↓

Blend

--------------------------------------------------

Every filter

input ndarray

↓

output ndarray

--------------------------------------------------

Pipeline

Load Image

↓

Apply Filter 1

↓

Apply Filter 2

↓

Apply Filter 3

↓

...

↓

Save

--------------------------------------------------

No filter may directly call another filter.

Pipeline controls execution.

--------------------------------------------------

Configuration

Every preset stored as JSON.

Example

graphic_novel.json

{
  "posterize_levels":5,
  "clahe_clip":3.5,
  "edge_strength":0.8,
  "contrast":1.3
}

--------------------------------------------------

Future

Color Comic

Anime

Cyberpunk

Vintage Newspaper

Woodcut

Sin City

Graphic Novel

Japanese Manga

Oil Painting

Etching

Blueprint

Pencil Sketch

--------------------------------------------------

Presets should require ZERO code changes.

Adding

config/presets/newstyle.json

should be sufficient.

--------------------------------------------------