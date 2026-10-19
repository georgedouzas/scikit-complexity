# Design

## Context

The catalog registers 301 models:

| Formalism | Models |
|---|---|
| ordinary differential equations | 157 |
| agent-based models | 57 |
| discrete maps | 24 |
| lattice dynamics | 24 |
| stochastic differential equations | 21 |
| jump processes | 10 |
| co-simulation | 5 |
| Monte Carlo sampling | 3 |

Their visuals are the five registered in `skcomplex.core.viz`:
- `ScalarsVisual`;
- `PointsVisual`, in two or three dimensions;
- `AgentsVisual`, on grids, continuous spaces, and networks;
- `ImageVisual`, for lattices and fields;
- `MeshVisual`, for finite-element fields.

`RerunObserver` streams frames on the `sim_time` and `step` timelines, logs parameters under `parameters/`, and can
save a `.rrd` file. It sends no blueprint, so the viewer lays out views by its own heuristics.

The documentation already runs `mkdocs-gallery` 0.10.4 on `docs/examples`, which holds only a README. That version
accepts custom `image_scrapers`.

### Probe of the native capture

The probe used Rerun 0.39.0 on macOS with Apple silicon, through `rerun <rrd> [<rbl>] --screenshot-to <png>
--window-size <w>x<h> --hide-welcome-screen`:

| Recording | Blueprint | Time | Result |
|---|---|---|---|
| `lotka_volterra`, 501 frames, 146 KB | none | 3 s | Seven panels, one per parameter and population, with the side, selection, and streams panels. The time cursor at frame 10. Two notifications over the top right. |
| `gray_scott`, 101 frames, 911 KB on disk, 62.2 MiB in the viewer | none | 2 s | 5.1 MiB loaded when captured. The image view empty and the time cursor at frame 0. |
| `gray_scott`, every tenth frame, 211 KB | two 2D views, collapsed panels, following time panel | 2 s | The layout applied, the panels collapsed, and the cursor at the last frame. Both image views still empty, and both notifications still drawn. |

The notifications are "Listening for gRPC connections" and a wgpu `Surface::configure` validation error: "Requested
was (40000, 1410), maximum extent for either dimension is 16384". At a logical window size of 1280x720 the image is
2560x1410, at 560 to 700 KB as PNG.

The probe settles three things:
- a blueprint is necessary;
- the capture fires before images reach the GPU;
- overlays must be cropped or suppressed.

## Goals / Non-Goals

**Goals:**
- One reviewed example and screenshot for every registered model, showing its known behaviour.
- A layout derived from the framework's own visuals, useful beyond the gallery.
- A capture that is headless, checked, reproducible, and cached, and that runs in CI.
- A coverage check that keeps the gallery complete as models are added.

**Non-Goals:**
- Animated captures, video, or GIFs. A later change may add them from the same recordings.
- Committing screenshots or recordings to the repository.
- Visual regression testing of the viewer itself. The checks reject broken captures but do not compare against golden
  images.
- Changing the models' physics to look better. Settings and blueprints change, laws do not.

## Decisions

### Layouts belong to visuals

Each `Visual` gains `views(entity_path, spec) -> list[rrb.View]`:
- `ScalarsVisual` gives a `TimeSeriesView`.
- `PointsVisual` gives a `Spatial2DView` or a `Spatial3DView` by the dimension of its space.
- `ImageVisual` gives a `Spatial2DView` per field state.
- `AgentsVisual` gives a `Spatial2DView` on grids and continuous spaces and a `GraphView` on networks.
- `MeshVisual` gives a `Spatial2DView`.

`default_blueprint(system, *, parameters=False)` arranges these views:
- time series are grouped by element into one view per element type, with at most six series in one view;
- spatial views come first and time series below;
- `rrb.Blueprint(..., collapse_panels=True)`;
- `rrb.TimePanel(play_state='following')`, with the cursor at the last frame;
- `rrb.TopPanel(state='hidden')`.

Rerun's blueprint API is the mature mechanism, so the framework only maps its visuals onto it. The alternative was a
blueprint per model by hand. It was rejected because 301 hand-made layouts would drift from the visuals, and users
would get no layout for their own models.

### Gallery recordings are bounded

Examples run with `interval` and `until` chosen so that the timeline holds at most 300 frames, and the recording is at
most 20 MB. Long runs are thinned rather than cut, so the last frame stays the final state. The budget comes from the
probe: the viewer's in-memory size is many times the file size, about 70 times for Gray-Scott, and the capture must see
the whole recording.

### Two capture backends, chosen by a spike

**Native backend:** `rerun --screenshot-to`.
- For: no extra dependency, and the same viewer users run.
- Against: it captures on the first rendered frame with no hook to wait for loading or GPU upload. It draws
  notifications, and it needs a GPU surface: Metal on macOS, or Vulkan through Mesa lavapipe or GL under Xvfb on
  Linux.
- Mitigations: run with `--expect-data-soon` and a blueprint, crop to the viewport, and retry with a delay.

**Web backend:** serve the recording and blueprint to the Rerun web viewer of the same version, load it in headless
Chromium through Playwright, and screenshot the viewer canvas.
- For: Playwright can wait for network idle and for a rendered canvas, the element screenshot excludes everything but
  the viewport, the page can hide notifications by CSS, and Chromium renders WebGL through SwiftShader without a GPU.
- Against: Playwright plus a Chromium download in the `gallery` group, and the web viewer's own limits on recording
  size.

The first task is a spike that captures a reference set with both backends. The set covers every visual and formalism:
`lotka_volterra`, `lorenz`, `gray_scott`, `game_of_life`, `wolf_sheep`, `ideal_gas`, `voter_model` on a network,
`heat_l_shape`, `boids`, and `sir` as a jump process. The default is the backend with:
1. the highest share of captures passing the checks, at least 95%;
2. headless runs on Linux CI;
3. the lower time per capture.

The design is then updated with the measured table and the choice. Playwright is chosen over Selenium and pyppeteer
because it is maintained by Microsoft, ships wheels with managed browsers, and has element screenshots and load-state
waits built in.

### Checks are cheap image statistics

The checks use Pillow and NumPy on the captured image:
- **Size:** the image is 1600x900 pixels, at a logical size of 800x450 with density 2.
- **Blank views:** every view rectangle has a pixel standard deviation above a threshold. The rectangles come from the
  blueprint's grid, because the viewport is the cropped image and the layout is known.
- **Notifications:** no region matches the viewer's notification background colours over more than a small area.
- **Encoding:** WebP at quality 80, with the size at most 300 KB.

A failed capture is retried once with a longer wait before it fails the build. Golden-image comparison is a non-goal,
because rendering differs across GPUs and backends.

### Examples are curated scripts, scaffolded from the catalog

A generator writes a starting script per model from the catalog. The script holds:
- the module docstring with a title and a caption placeholder;
- `MODEL = '<name>'`;
- `create(MODEL, ...)` with the factory defaults;
- `Simulation(..., observers=[RerunObserver(save_path=..., blueprint='default')])` with gallery settings.

Contributors then tune the parameters, settings, and blueprint per model and write the caption. Scripts are files
users read and copy, so they stay plain code. Metadata on `register_model` was rejected because it would put
documentation concerns in the runtime.

The `mkdocs-gallery` Rerun scraper finds the `.rrd` and `.rbl` an example saved, captures them or takes them from the
cache, and returns the image to the page. A conventional path under the gallery's build directory makes the outputs
discoverable.

### The cache key is content-based

`sha256` of:
- the example source;
- the sources of the model's subfield package;
- `rerun.__version__`;
- the backend name;
- the capture settings, which are the window size, density, waits, and encoder settings.

Screenshots are stored under `docs/generated/gallery/_cache/` locally and in the CI cache. They are published with the
site only. The subfield package is the unit of the key because models share components within it, and are
self-contained across subfields by the layout rule.

### Batches per field, with review

The 301 examples are delivered in nine batches, one per field. Each batch includes the review of every screenshot
against its caption. A screenshot that does not show the caption is fixed in the example, by its parameters,
`until`, `interval`, or blueprint, never by editing images. Models whose behaviour needs a run beyond the budget, such
as glacial cycles, are thinned and say so in their caption.

## Risks / Trade-offs

- **Rerun upgrades change rendering or the CLI.** The cache key includes the Rerun version, so an upgrade recaptures
  everything. The spike's reference set runs in CI as a smoke test of both backends.
- **Headless GPU support differs across CI runners.** The web backend with SwiftShader needs no GPU. The native
  backend needs lavapipe or Xvfb, and stays the fallback.
- **A cold build is slow.** At about 5 s per model for run and capture, a cold build takes about 25 minutes serially.
  The gallery session captures in parallel processes, and the cache keeps warm builds to the changed examples.
- **Review effort is large.** It is spread over per-field batches, each with a review list, and the automatic checks
  catch blank and broken captures before review.
- **The wgpu `Surface::configure` error on macOS** may be a viewer bug. The spike records whether it affects rendered
  content. If it does, it is reported upstream, and the web backend becomes the default on macOS too.
- **Models changed by pending changes** need their examples updated in those changes. This is enforced by the
  model-library requirement and the coverage check. `add-group-reductions` and `extend-interaction-roles` rebuild
  models now, and should land before their fields' batches.

## Migration Plan

1. Land the layouts and the observer argument. `blueprint=None` keeps today's recordings unchanged.
2. Land the capture, the checks, the cache, the scraper, and the coverage check. The coverage check runs in warning
   mode until the last batch.
3. Deliver the nine field batches.
4. Switch the coverage check and the strict docs build to failing mode.

## Open Questions

- The default backend, decided by the spike.
- Whether the interactive embed publishes recordings within the site size limit of the hosting. If not, it links to a
  release asset per field instead.
