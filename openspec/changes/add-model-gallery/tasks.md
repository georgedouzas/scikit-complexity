# Tasks

## 1. Capture spike

- [ ] 1.1 Capture the reference set (`lotka_volterra`, `lorenz`, `gray_scott`, `game_of_life`, `wolf_sheep`,
  `ideal_gas`, `voter_model`, `heat_l_shape`, `boids`, `sir`) with the native viewer and with the web viewer in headless
  Chromium through Playwright, on macOS and on Linux with software rendering. Record each backend's share of captures
  passing the checks, time per capture, and headless support in the design, choose the default, and verify the backend
  comparison scenario. Done on macOS and recorded in the design, with the native default. Open: the Linux leg, which
  the `gallery` CI job runs with lavapipe and SwiftShader, has not been measured yet.

## 2. Layouts

- [x] 2.1 Add `views` to `Visual` and to the five built-in visuals, and verify the views of the built-in visuals
  scenario
- [x] 2.2 Add `default_blueprint(system, parameters=False)` with views ordered by importance, `primary_view`, and
  `save_blueprint`, re-export them from `skcomplex.core.viz`, and verify the ODE, field, primary view, and parameters
  scenarios
- [x] 2.3 Add the `blueprint` argument of `RerunObserver`, which sends the blueprint and saves it next to a saved
  recording, and verify the saved recording and no blueprint scenarios
- [x] 2.4 Document layouts and blueprints in the user guide with a runnable example, and verify the documentation test
- [x] 2.5 Make the visuals legible at the thumbnail size: the space-time image of one-dimensional lattices as their
  primary view, point radii that scale with the extent of the positions, trails of small groups of points, and agent
  radii that scale with a continuous space, and verify the space-time view, legible points, and agents without a
  space scenarios

## 3. Capture, checks, and cache

- [x] 3.1 Add the `gallery` dependency group with Pillow and Playwright, and the capture module with both backends, the
  wait for loading and rendering, and cropping to the viewport, and verify the field at its last frame and overlays
  excluded scenarios
- [x] 3.2 Add the checks for size, blank views, notifications, and encoding as an optimised PNG within 300 KB, with
  one retry, and the placeholder check, and verify the blank view, format and size, and no placeholder thumbnails
  scenarios
- [x] 3.3 Add the content-based cache and its key, and verify the warm build and changed model scenarios
- [x] 3.4 Verify the repeated capture scenario on both backends
- [x] 3.5 Tell letterboxing from sparse content and from coloured backgrounds, tell notifications from grey content,
  and encode with adaptive palettes, and verify the sparse content and grey content scenarios

## 4. Examples and documentation

- [x] 4.1 Add the generator that scaffolds an example per registered model, the budget check for frames, recording
  size, and run time, and the coverage check in warning mode, and verify the recording budget, over budget, missing
  example, and orphan example scenarios
- [x] 4.2 Add the Rerun image scraper to the `mkdocs-gallery` configuration, returning exactly one screenshot of the
  primary view before the Matplotlib scraper, the analysis plots of the reference examples, and the gallery navigation
  by field and subfield, and verify the thumbnail in the gallery grid, primary view of a time series model, ideal gas
  speeds, and thumbnail stays the Rerun screenshot scenarios
- [x] 4.3 Show the screenshots on the field pages and link the examples from the coverage page, and verify the field
  page scenario
- [x] 4.4 Add the lazy embedded Rerun web viewer with the published recording, and verify the open the recording
  scenario
- [x] 4.5 Add `nox -s gallery` with `--backend` and parallel captures, and a CI job with software rendering and the
  screenshot cache, and verify the selecting a backend scenario

## 5. Field batches

For every task in this section:
- curate the scaffolded example of each model of the field, by its parameters, gallery settings, primary view,
  blueprint, and caption;
- add its analysis plot, computed from its own run, that shows the behaviour the model is known for, with labelled
  axes, a title, and a sentence on the agreement to see;
- check that its thumbnail reads at the grid's display size of 160x112;
- capture its screenshot;
- review it against the caption and record the review in the batch's list;
- verify that the field's examples pass the budget, checks, and coverage.

- [x] 5.1 Physics
- [x] 5.2 Chemistry
- [x] 5.3 Biology
- [x] 5.4 Earth science
- [x] 5.5 Economics
- [x] 5.6 Social science
- [x] 5.7 Engineering
- [x] 5.8 Mathematics
- [x] 5.9 Computer science

## 6. Enforcement and gate

- [x] 6.1 Switch the coverage check and the strict documentation build to failing mode, and verify the strict build
  without a screenshot and new model without an example scenarios
- [x] 6.2 Document the gallery, how to add an example for a new model, and the review rule in the contributing guide,
  and add the `gallery` group and Playwright to the Project Profile
- [ ] 6.3 Run formatting, checks, the docs build with the gallery, and tests through nox, and verify all of them pass
