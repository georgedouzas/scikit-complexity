# Proposal

## Why

The library registers 301 ready-made models across nine fields, and each is meant to be both a working model and the
pattern a user copies for a model of their own. Today a user sees a model only as source code and, on its field page,
as its laws and equations. Nothing shows what a model does when it runs. Rerun is the framework's visualisation layer,
yet no page shows a single Rerun view of any model.

A gallery with one runnable example and one Rerun screenshot per model makes the library browsable. It shows the
behaviour each model is known for, such as the Lotka-Volterra cycles, the Gray-Scott spots, or the flocks of boids. It
also shows the user the few lines that run a model and stream it to Rerun.

The screenshots are the gallery's thumbnails, so each must show its model's signature view legibly at the size the grid
displays it: 160x112 pixels by the default CSS, from a 400x280 thumbnail. Producing 301 faithful screenshots is hard,
and a probe with the installed Rerun 0.39 shows why:
- `rerun <file>.rrd --screenshot-to <png> --window-size <size>` takes a screenshot of the native viewer and quits, in
  about 2 to 3 seconds per recording.
- **No layout:** without a blueprint the viewer shows every parameter in its own time series panel, around the side,
  selection, blueprint, and streams panels. The model's own dynamics are a small part of the image.
- **Too early:** the screenshot is taken before the recording has loaded and before images are uploaded to the GPU. On
  Gray-Scott, 5.1 MiB of 62.2 MiB had loaded, and the image views were empty, with or without a blueprint. The time
  cursor sits at the start of the recording unless the blueprint sets the time panel to follow.
- **Overlays:** the viewer draws a notification that it listens for gRPC connections, and on macOS a wgpu
  `Surface::configure` validation error ("Requested was (40000, 1410)"), over the top right of the views.
- **Size:** a 1280x720 window on a high-density display gives a 2560x1410 PNG of 560 to 700 KB, so 301 of them would
  add about 200 MB.

The framework therefore needs layouts derived from its own visuals, gallery-sized recordings, a capture that waits for
the views to render and checks the result, and a cache, so that the gallery is reproducible, reviewed, and cheap to
rebuild.

## What Changes

- **Layouts from visuals:** every visual declares the Rerun views it belongs in. A new `default_blueprint(system)`
  builds a Rerun blueprint from the visuals of a system's element types: time series for scalars grouped by element,
  2D or 3D spatial views for points, agents, lattices, and fields, and graph views for networks. Parameters are left
  out unless asked for, the side panels are collapsed, and the time panel follows the latest frame.
  `RerunObserver(blueprint=...)` sends a default or a given blueprint, and `save_blueprint` writes a `.rbl` file.
- **Gallery recordings:** a gallery run records a bounded number of frames and stays within a recording size budget,
  so that the capture loads it completely.
- **Capture:** a capture tool turns a recording and its blueprint into a screenshot.
  - **Backend:** it runs either the native viewer or the Rerun web viewer in headless Chromium through Playwright. A
    measured spike chooses the default and keeps the other as the fallback.
  - **Waiting:** it waits until the recording has loaded and the views have rendered.
  - **Overlays:** it crops to the viewport, so that overlays and panels are excluded.
  - **Checks:** it rejects blank views, notifications, and wrong sizes.
  - **Output:** it writes an optimised PNG within a size budget, because `mkdocs-gallery` builds thumbnails only from
    PNG, JPG, SVG, and GIF.
- **Cache:** screenshots are cached by a key of the example source, the model's source, the Rerun version, and the
  capture settings. The documentation build regenerates only stale ones, and they are not committed to the repository.
- **Examples:** every registered model has one example script under `docs/examples/<field>/<subfield>/`, in the format
  of `mkdocs-gallery`. Each script states its run, its blueprint, and a caption of what the image shows, and uses only
  the public interface. A Rerun image scraper for `mkdocs-gallery` returns one screenshot of the model's primary view,
  which becomes the example's thumbnail in the gallery grid.
- **Analysis plots:** the body of each example shows Matplotlib figures computed from the simulation's own results,
  the science the model is known for, such as the simulated speeds of the ideal gas against the Maxwell-Boltzmann
  distribution, or the phase portrait and conserved quantity of Lotka-Volterra. The built-in Matplotlib scraper of
  `mkdocs-gallery` renders them after the Rerun screenshot.
- **Coverage:**
  - **Tests:** a test fails when a registered model has no example, when an example names no registered model, or
    when a built gallery lacks a screenshot.
  - **Documentation:** the field pages show each model's screenshot above its pattern, and the coverage page links to
    the examples.
  - **Interactive view:** an optional embedded Rerun web viewer opens the example's recording on request.
- **Review:** every screenshot is reviewed once against the behaviour its model is known for. The review is recorded
  in the example's caption, and the 301 examples are delivered in per-field batches.
- **Tooling:** a `gallery` dependency group (Pillow, Playwright), a `nox -s gallery` session, and a CI job with
  software rendering.

## Capabilities

### New Capabilities
- `model-gallery`: runnable example scripts per registered model, the screenshot capture with its checks and cache,
  the `mkdocs-gallery` scraper, the coverage check, and the documentation pages that show the screenshots.

### Modified Capabilities
- `rerun-visualization`: visuals declare their Rerun views, systems get a default blueprint, and the observer sends
  and saves blueprints.
- `model-library`: every registered model has a gallery example and a screenshot.

## Impact

- **Code:**
  - `skcomplex.core.viz` gains views on visuals, `default_blueprint`, `save_blueprint`, and the `blueprint` argument
    of `RerunObserver`.
  - A new private module holds the capture and its checks. It is used only by the documentation tooling and is not
    part of the runtime.
  - `docs/generate_models.py` embeds the screenshots.
- **Examples and docs:** about 301 example scripts under `docs/examples/`, the `mkdocs-gallery` configuration with the
  Rerun scraper, and pages per field in the gallery navigation.
- **Dependencies:**
  - a new development group `gallery` with Pillow and Playwright, plus a headless Chromium installed by Playwright;
  - no new runtime dependency;
  - Rerun stays at the version the project pins, and the cache key includes it.
- **CI:** a job that builds the gallery with software rendering (Mesa lavapipe or SwiftShader) and caches screenshots
  between runs.
- **Build time:** a cold gallery build runs 301 models and captures 301 screenshots. The cache keeps warm builds to
  the stale examples only.
