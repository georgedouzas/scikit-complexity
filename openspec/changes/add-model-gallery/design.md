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

Each `Visual` gains `views(element, problem) -> list[rrb.View]`, empty by default, so a user visual declares its views
the same way:
- `ScalarsVisual` gives a `TimeSeriesView`.
- `PointsVisual` gives a `Spatial2DView` or a `Spatial3DView` by the dimension of its position, and a `TimeSeriesView`
  of its other logged states.
- `ImageVisual` gives a `Spatial2DView` per field state, or a `TensorView` for lattices of three or more dimensions.
- `AgentsVisual` gives a `Spatial2DView` in grids and continuous spaces, a `GraphView` framed on the network on
  networks, and no view for agents without a space, which log no location.
- `MeshVisual` gives a `Spatial3DView`, not the `Spatial2DView` first planned, because it logs a `Mesh3D` with the
  value as height, which a 2D view cannot draw.

`default_blueprint(system, *, parameters=False, primary=None)` arranges these views:
- time series are grouped by element type into one view, with at most six series in one view;
- points of several elements share one spatial view, and so do the agents of several groups, since they share the
  system's space;
- spatial views come first and side by side, time series below, and an observables view and a reporters view for agent
  models come after the element views;
- time series views show the whole run (`TimeAxis` with an infinite view range), since in a following time panel the
  plot otherwise scrolls with the cursor and shows only the last minute of a long run;
- `primary` names a view by name or origin to put first, which is how an example chooses its thumbnail;
- `rrb.Blueprint(..., collapse_panels=True)`, `rrb.TimePanel(state='collapsed', timeline='sim_time',
  play_state='following')`, and `rrb.TopPanel(state='hidden')`.

`primary_view(system)` returns the first view, and `save_blueprint(blueprint, path, application_id=...)` writes a
`.rbl`. `RerunObserver(blueprint=None | 'default' | rrb.Blueprint)` sends the blueprint with the recording, embeds it
in a saved `.rrd` as its default blueprint, writes it next to the recording as a `.rbl`, and exposes it as
`blueprint_`.

Two visual fixes came out of the reference set, because without them the captures were not reproducible or not legible:
- agents on a network are logged with fixed positions: the `pos` attributes of the nodes when every node has one, and
  a spring layout with seed 0 otherwise, scaled in the user-interface points the graph view draws in. Without fixed
  positions the viewer's force layout makes every capture different, and without the scale the nodes are invisible;
- the default series lines are 2.5 points wide, so they read at the grid size.

Rerun's blueprint API is the mature mechanism, so the framework only maps its visuals onto it. The alternative was a
blueprint per model by hand. It was rejected because 301 hand-made layouts would drift from the visuals, and users
would get no layout for their own models.

### Gallery recordings are bounded

Examples run with `interval` and `until` chosen so that the timeline holds at most 300 frames, and the recording is at
most 20 MB. Long runs are thinned rather than cut, so the last frame stays the final state. The budget comes from the
probe: the viewer's in-memory size is many times the file size, about 70 times for Gray-Scott, and the capture must see
the whole recording.

### Two capture backends, chosen by a spike

The spike found that Rerun 0.39 itself solves the waiting problem of the native viewer: `rerun --headless` runs the
viewer without a window, and `rerun.experimental.ViewerClient` reports what it is loading, which views it shows, and
the timelines of each recording, moves the time cursor, and saves a screenshot of the window. The native backend
therefore uses that client instead of `--screenshot-to`, which stays only as the baseline of the comparison.

**Native backend (`NativeCapturer`):** the headless viewer, driven by `ViewerClient`.
- It starts with a fresh `HOME`, so no persisted state of the contributor's own viewer applies: no light theme, no
  open panels, and default settings.
- It starts with `RUST_LOG=warn`, so the viewer posts no info notifications, which are what the probe saw over the
  views ("Listening for gRPC connections"). The headless viewer draws no window surface, so the wgpu
  `Surface::configure` error of the windowed probe does not occur.
- It opens the `.rrd` and waits until the viewer reports nothing loading and its timelines and views have stayed the
  same for four polls of 50 ms. This replaces the fixed delay of the probe and is what fixes the empty image views.
- The viewer loads files through its catalog by default, which renames the application id of the recording to a
  dataset id, so a `.rbl` opened from a file does not apply. The capture sends the blueprint over gRPC with a
  `RecordingStream` under the dataset and segment ids the viewer reports, waits until the viewer shows the expected
  number of views, and moves the cursor to the end of the `sim_time` timeline.
- It saves the whole window, which shows only views because the capture blueprints hide every panel. The thumbnail
  blueprint holds the primary view alone with `titlebar=False`, so the thumbnail is the view and nothing else.
- The headless viewer renders at one pixel per point, so it draws the 600x420 logical layout and Pillow scales it to
  1200x840.

**Web backend (`WebCapturer`):** the Rerun web viewer of the installed version, served by `rerun --serve-web`, in
headless Chromium through Playwright.
- The recording and a `.rbl` saved with the recording's application id are served over HTTP and passed as `url`
  parameters, so the blueprint applies without a catalog.
- It waits for the network to be idle and then for three identical screenshots of the canvas.
- The page uses the dark colour scheme and a device pixel ratio of 1, since the web viewer doubles its interface on
  denser pages, so both backends draw the same layout.

**Measured comparison** on macOS 27 with Apple silicon, Rerun 0.39.0, the ten reference examples, two captures per
example (the thumbnail and a full layout, which the spike still took), each capture starting its own viewer. The
gallery now takes the thumbnail only, so a capture takes about half these times:

| Backend | Passing the checks | Time per example (both captures) | Headless |
|---|---|---|---|
| `--screenshot-to` (baseline) | 0 of 10: the contributor's persisted light theme and panels, no blueprint applied through the catalog, and two notifications over the views | 0.4 s | no, opens a window |
| native, `ViewerClient` | 10 of 10, and two captures differ by less than 2% | 2.9 to 4.6 s, median 3.1 s | yes on macOS (Metal); Linux needs Vulkan through Mesa lavapipe |
| web, Playwright | 10 of 10, and two captures differ by less than 2% | 7.8 to 13.0 s, median 10.9 s | yes, Chromium with SwiftShader needs no GPU |

On Linux with software rendering neither backend was measured locally, since no Linux machine or container was
available. The CI job `gallery` runs both backends on Ubuntu, the native one with `mesa-vulkan-drivers`, and records
the result there.

**Default:** the native backend. Both pass every check, and the native one is three times faster. The web backend
stays selectable with `--backend web` or `SKCOMPLEX_GALLERY_BACKEND=web`, and is the fallback where no Vulkan adapter
exists. Playwright is chosen over Selenium and pyppeteer because it is maintained by Microsoft, ships wheels with
managed browsers, and has load-state waits built in.

### Screenshots are thumbnails first

`mkdocs-gallery` makes an example's thumbnail from the first image its scrapers return, or from the image numbered by
`mkdocs_gallery_thumbnail_number`. It rescales that image to fit `thumbnail_size`, 400x280 by default, and its CSS
shows it at 160x112. The gallery grid is therefore the main place a user sees a screenshot.

The Rerun scraper returns one capture, the thumbnail, which is the example's primary view alone: the view that shows
the model's signature behaviour, such as the `v` field of Gray-Scott, the populations of Lotka-Volterra, or the
attractor of Lorenz.

`default_blueprint` orders views by importance, spatial first, and exposes `primary_view(system)`. An example names
another view as primary with `default_blueprint(system, primary=...)` or with a blueprint of its own whose first view
is the primary one, which is the curation step of the field batches. The thumbnail capture uses a blueprint with that
view alone, with no view title bar, at the 10:7 ratio. Lines are drawn thick enough to read at 160x112.
`mkdocs_gallery_thumbnail_number` is left at its default of 1.

Each example has exactly one Rerun screenshot. An earlier version also captured the full layout for the example page,
which was dropped, since the page shows the science of the model better with analysis plots, and the full layout opens
in the interactive viewer. The scraper renders its image with a single-path call of `figure_md_or_html`, whose branch
for several images in `mkdocs-gallery` 0.10.4 returns an unformatted HTML list that renders as a literal `%s`.

### Examples show analysis plots

Below the Rerun screenshot, each curated example computes from its own run the plot its model is known for and draws
it with Matplotlib. `image_scrapers` is `(RerunScraper, 'matplotlib')`, so in a block that both finishes a run and
draws a figure the screenshot is still image 1, and `reset_modules` resets Matplotlib between examples. The reference
set uses:

| Example | Analysis plot | Agreement shown |
|---|---|---|
| `ideal_gas` | histogram of the final speeds against the 2D Maxwell-Boltzmann density at the temperature measured from the kinetic energy | the histogram follows the curve, kT = 0.97 |
| `lotka_volterra` | phase portrait and relative change of `V = b x - d ln x + a y - r ln y` | one closed orbit, `V` constant to 2e-8 |
| `lorenz` | Lorenz map of successive maxima of `z` | a thin tent-shaped curve |
| `gray_scott` | final `v` field and its radial power spectrum | a peak at a spot spacing of 16 cells |
| `game_of_life` | density of live cells over generations | decay from 0.3 to about 0.05 |
| `wolf_sheep` | sheep and wolf counts | out-of-phase oscillations |
| `voter_model` | fraction of opinion 1 and interface density | the mean wanders, the interfaces decline |
| `heat_l_shape` | final temperature against `exp(x) sin(y)` and the largest error over time | nodes on the exact line, exponential decay to the mesh error |
| `boids` | polarization of the headings | rise from 0.1 to near 1 |
| `sir` | susceptible and infected curves against the final-size equation | 56 susceptibles left, 59 predicted |

The runner sets the Matplotlib backend to Agg, so running examples in parallel opens no window. Matplotlib is declared
in the `docs` group, since `mkdocs-gallery` renders its figures and it was only installed transitively.

### Checks are cheap image statistics

The checks use Pillow and NumPy on the captured image:
- **Size:** the thumbnail capture is 1200x840 pixels, at a logical size of 600x420 with density 2. That is the 10:7
  ratio of the gallery's `thumbnail_size` of 400x280, so rescaling neither crops nor letterboxes it.
- **Blank views:** every view rectangle, without its title bar, has a pixel standard deviation above 2 and at least
  0.05% of saturated pixels. The viewer's chrome is grey, and everything the visuals draw is coloured, so an empty
  plot with its axes and labels counts as blank. The rectangles come from the blueprint's containers and shares.
- **Notifications:** no more than 5% of the top right of a view has the notification background (33, 33, 33).
- **Letterboxing:** a view with a fixed aspect, such as a square lattice, leaves background bands in the 10:7 frame. The
  capture trims the bands of the corner colour and scales the content to fill the frame, cropping around the centre,
  and the check rejects a band of one colour wider than a tenth of the width or height along an edge.
- **Chrome:** no more than a fifth of the image is near-white grey, which catches a persisted light theme or open
  panels, as the baseline showed.
- **Encoding:** a PNG optimised by Pillow, with the size at most 300 KB, reduced to a palette of 256, 128, 64, or 32
  colours while it is larger. WebP was rejected because `mkdocs-gallery`
  0.10.4 finds and rescales thumbnails only from PNG, JPG, SVG, and GIF, and silently falls back to its placeholder
  image otherwise.
- **Placeholder:** the gallery build compares every generated thumbnail with the `mkdocs-gallery` placeholder
  `no_image.png`, and fails on a match.

A failed capture is retried once with twice the wait before it fails the build. The placeholder check runs as a
`mkdocs` hook after the gallery is generated: an example the build ran without a screenshot fails the build, and the
scaffolds it did not run are reported at info level, which is the warning mode. Golden-image comparison is a non-goal,
because rendering differs across GPUs and backends.

### Examples are curated scripts, scaffolded from the catalog

`python -m gallery_tools scaffold` writes a starting script per model from the catalog, with the gallery headers of
each field and subfield. The script holds:
- the module docstring with the title from the inventory, the model's description as its caption, and the sentence
  "This example is scaffolded from the catalog and awaits curation.";
- `MODEL = '<name>'`;
- `RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=...)`, with the interval that keeps
  the default run within 300 frames and logs its last frame;
- `run(MODEL, settings={'observers': [observer], 'random_state': 0})`, with `random_state=0` for seeded factories.

Contributors then tune the parameters, settings, and blueprint per model, write the caption, and remove the sentence.
Scripts are files users read and copy, so they stay plain code. Metadata on `register_model` was rejected because it
would put documentation concerns in the runtime.

The documentation build runs the curated examples only. `docs/gallery_conf.py`, the `conf_script` of `mkdocs-gallery`,
sets `filename_pattern` to the scripts without the scaffold sentence, so the scaffolds appear in the gallery with
their code and the placeholder thumbnail but do not run. `SKCOMPLEX_GALLERY=all` runs every script, which is what
`nox -s gallery` does by default.

The gallery is nine galleries, one per field, each with one subsection per subfield, because `mkdocs-gallery` nests
one level of subsections only. The `mkdocs-gallery` Rerun scraper finds the observer that finished saving a recording
among the example's variables, takes its screenshots from the cache or captures them, and copies them to the example's
images. It moves the `.rrd` and `.rbl` next to them, where the site publishes them for the interactive viewer.

The tooling lives in `docs/gallery_tools`, beside the other documentation scripts, because it needs Pillow and
Playwright, which are development dependencies, and the package keeps only `core` and `models`.

### The cache key is content-based

`sha256` of:
- the example source;
- the sources of the model's subfield package;
- `rerun.__version__`;
- the backend name;
- the capture settings, which are the window size, density, waits, encoder settings, and a version of the capture
  procedure, raised whenever the capture or its processing changes the images.

Screenshots are stored under `docs/generated/gallery/_cache/` locally and in the CI cache, and `exclude_docs` keeps the
cache out of the site. The scraper writes the key of the screenshots a page shows next to the page, and the
configuration script removes the checksum of `mkdocs-gallery` for an example whose key changed or whose screenshots
are not cached, so only those examples run again. The subfield package is the unit of the key because models share components within it, and are
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
  Measured with the ten reference examples: a cold documentation build takes 72 s, a build with every screenshot cached
  but every example run again 40 s, and a warm build 24 s, against about 20 s for the documentation alone. Running all
  301 examples with captures in six processes took 3 min 49 s.
- **Two builds at once collide.** Examples save their recording next to the script, and the scraper moves it, so a
  `properdocs serve` and a `properdocs build` running together take each other's recordings. Build one at a time.
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

- Whether the interactive embed publishes recordings within the site size limit of the hosting. If not, it links to a
  release asset per field instead.
