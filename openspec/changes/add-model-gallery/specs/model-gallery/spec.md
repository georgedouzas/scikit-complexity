# Spec Delta

## Purpose

Gives every registered model a runnable example and a Rerun screenshot of its known behaviour, captured
reproducibly, checked automatically, cached between builds, and shown in the documentation.

## ADDED Requirements

### Requirement: Every registered model has one example
The documentation SHALL hold exactly one example script per registered model, at
`docs/examples/<field>/<subfield>/plot_<name>.py`, in the format of `mkdocs-gallery`. Each script SHALL name its model
in a module constant, use only public names of `skcomplex`, run the model, stream it to Rerun with a blueprint, and
state in its docstring a caption of what the screenshot shows.

#### Scenario: Example of a model
- **WHEN** a user opens `docs/examples/biology/ecology/plot_lotka_volterra.py`
- **THEN** it creates `lotka_volterra` by name, runs it with a `RerunObserver` and a blueprint, and its docstring says
  that the prey and predator populations cycle out of phase

#### Scenario: Missing example
- **WHEN** a model is registered and no example names it
- **THEN** the gallery coverage check fails and names the model

#### Scenario: Orphan example
- **WHEN** an example names a model that is not registered, or two examples name the same model
- **THEN** the gallery coverage check fails and names the example

### Requirement: Examples run within a budget
Every example SHALL run its model with gallery settings that record at most 300 frames and a recording of at most
20 MB, and SHALL finish within 60 seconds on the reference machine, so the capture can load the whole recording.

#### Scenario: Recording budget
- **WHEN** the gallery runs every example
- **THEN** each recording has at most 300 frames on its time line and at most 20 MB, and the run report lists the
  slowest examples with their times

#### Scenario: Over budget
- **WHEN** an example records more frames or a larger recording than the budget
- **THEN** the gallery build fails and names the example and the exceeded limit

### Requirement: Screenshots are the gallery thumbnails
Every example SHALL produce exactly one Rerun screenshot, which its Rerun scraper returns as the first image of the
example, before any analysis figure, so that `mkdocs-gallery` makes it the example's thumbnail in the gallery grid. The
screenshot SHALL show the model's primary view alone, at the aspect ratio of the gallery's `thumbnail_size`, and read
at the size the gallery displays it.

#### Scenario: Thumbnail in the gallery grid
- **WHEN** a user opens the gallery index of `chemistry/pattern_formation`
- **THEN** the Gray-Scott thumbnail is the Rerun image of its `v` field at the last frame, with spots, filling the
  thumbnail at the 10:7 ratio of the configured `thumbnail_size` of 400x280

#### Scenario: Primary view of a time series model
- **WHEN** the `lotka_volterra` thumbnail is captured
- **THEN** it shows one time series view of the prey and predator populations, with no parameter views and no panels

#### Scenario: No placeholder thumbnails
- **WHEN** the gallery is built
- **THEN** no example's thumbnail is the `mkdocs-gallery` placeholder image, and the build fails and names any example
  whose scraper returned no image

### Requirement: Examples show analysis plots from the simulation
Every curated example SHALL show at least one Matplotlib figure computed from its own run, from the trajectory, the
observables, or the agents of the simulation, that exhibits the behaviour its model is known for. Each figure SHALL
have labelled axes and a title, and the text before it SHALL state the agreement the reader should see. The analysis
SHALL use the functions of `skcomplex.core.analysis` where they fit, and NumPy or SciPy otherwise.

#### Scenario: Ideal gas speeds
- **WHEN** a user opens the ideal gas example page
- **THEN** below the Rerun screenshot it shows the histogram of the simulated molecular speeds at the end of the run
  against the Maxwell-Boltzmann speed distribution, in the dimension of the model, at the temperature measured from the
  kinetic energy of the simulated molecules

#### Scenario: Thumbnail stays the Rerun screenshot
- **WHEN** an example draws analysis figures
- **THEN** its gallery thumbnail is still its Rerun screenshot, and the figures follow the screenshot on its page

### Requirement: Screenshots are captured from Rerun
The capture SHALL turn an example's recording and blueprint into an image of the Rerun views at the last frame. It SHALL
wait until the recording has loaded and the views have rendered, and SHALL crop to the viewport, without the side
panels, the time panel, or notifications.

#### Scenario: Field at its last frame
- **WHEN** the Gray-Scott example is captured
- **THEN** the screenshot shows the `v` field at the last recorded frame, with spots, and no empty view

#### Scenario: Overlays excluded
- **WHEN** the viewer shows a notification during capture
- **THEN** the notification is not in the screenshot

### Requirement: Screenshots are checked
Every screenshot SHALL pass automatic checks before it is kept. It SHALL be a PNG of the gallery size. No view in it
SHALL be uniform. It SHALL hold no viewer notification, and it SHALL be at most 300 KB. A failed check SHALL retry the
capture once with a longer wait, and then fail the build, naming the example and the check.

#### Scenario: Blank view
- **WHEN** a capture shows a view with a single colour, such as an image not yet uploaded
- **THEN** the capture retries once, and if the view is still blank the build fails and names the example and the view

#### Scenario: Format and size
- **WHEN** a screenshot is encoded
- **THEN** it is an optimised PNG, a format `mkdocs-gallery` turns into thumbnails, of the gallery size and at most
  300 KB

### Requirement: Captures use a chosen backend with a fallback
The capture SHALL support two backends: the native Rerun viewer in headless mode, driven by Rerun's viewer client, and
the Rerun web viewer in headless Chromium through Playwright. The design SHALL name the default chosen by a measured comparison, and the other
SHALL remain selectable. Both SHALL produce images that pass the same checks.

#### Scenario: Backend comparison
- **WHEN** the spike captures the reference set of examples with both backends
- **THEN** the design records, per backend, the share of captures that pass the checks, the time per capture, and the
  platforms where it runs headless, and names the default

#### Scenario: Selecting a backend
- **WHEN** a contributor runs `nox -s gallery -- --backend native` or `--backend web`
- **THEN** the gallery is captured with that backend

### Requirement: Captures are reproducible
A screenshot SHALL depend only on the example, the model's code, the Rerun version, and the capture settings. Seeded
models SHALL use a fixed seed in their example, the window size and pixel density SHALL be fixed, and two captures of
an unchanged example SHALL pass the same checks and differ by at most a small pixel tolerance.

#### Scenario: Repeated capture
- **WHEN** an unchanged example is captured twice on the same backend
- **THEN** both images pass the checks and their mean absolute pixel difference is below 2%

### Requirement: Screenshots are cached
The gallery SHALL cache each screenshot under a key computed from the example source, the source of the model's
subfield package, the installed Rerun version, the backend, and the capture settings. It SHALL capture only the
examples whose key changed. Screenshots SHALL NOT be committed to the repository.

#### Scenario: Warm build
- **WHEN** the gallery is built twice with no change in between
- **THEN** the second build runs no example and captures no screenshot

#### Scenario: Changed model
- **WHEN** the source of `biology/ecology` changes
- **THEN** the next build recaptures the ecology examples only

### Requirement: The gallery is part of the documentation
The documentation build SHALL show every example with its code, its caption, and its screenshot in a gallery grouped by
field and subfield. Each field page SHALL show each model's screenshot above its pattern, and the coverage page SHALL
link every implemented model to its example. A strict build SHALL fail when an implemented model has no screenshot.

#### Scenario: Field page
- **WHEN** a user opens the documentation page of the field `chemistry`
- **THEN** each model shows its screenshot and a link to its example above its factory and equations

#### Scenario: Strict build without a screenshot
- **WHEN** the strict documentation build finds a registered model without a cached or captured screenshot
- **THEN** the build fails and names the model

### Requirement: Recordings open interactively
Each example page SHALL offer the example's recording in an embedded Rerun web viewer, loaded only when the user asks
for it. The recording SHALL be published with the documentation within the recording budget.

#### Scenario: Open the recording
- **WHEN** a user clicks the interactive view on the Lorenz example page
- **THEN** the Rerun web viewer loads the example's recording with its blueprint, and the page loads no viewer before
  the click

### Requirement: Screenshots are reviewed
Every screenshot SHALL be reviewed once by a contributor against the behaviour its model is known for, and the
example's caption SHALL state that behaviour. A screenshot whose caption the image does not show SHALL be fixed by
changing the example's settings or blueprint, not by editing the image.

#### Scenario: Review record
- **WHEN** a contributor finishes a field's batch of examples
- **THEN** every example of the field has a caption, and the review list of the batch marks each screenshot as checked
  against its caption
