## MODIFIED Requirements

### Requirement: Every registered system has one example
The documentation SHALL hold exactly one example script per registered system, at
`docs/examples/<field>/<subfield>/plot_<name>.py`, in the format of `mkdocs-gallery`. Each script SHALL name its system
in a module constant, use only public names of `skcomplex`, run the system, stream it to Rerun with a blueprint, and
state in its docstring a caption of what the screenshot shows.

#### Scenario: Example of a system
- **WHEN** a user opens `docs/examples/biology/ecology/plot_lotka_volterra.py`
- **THEN** it creates `LotkaVolterraSystem` by name, runs it with a `RerunObserver` and a blueprint, and its docstring says
  that the prey and predator populations cycle out of phase

#### Scenario: Missing example
- **WHEN** a system is registered and no example names it
- **THEN** the gallery coverage check fails and names the system

#### Scenario: Orphan example
- **WHEN** an example names a system that is not registered, or two examples name the same system
- **THEN** the gallery coverage check fails and names the example

### Requirement: Screenshots are the gallery thumbnails
Every example SHALL produce exactly one Rerun screenshot, which its Rerun scraper returns as the first image of the
example, before any analysis figure, so that `mkdocs-gallery` makes it the example's thumbnail in the gallery grid. The
screenshot SHALL show the system's primary view alone, at the aspect ratio of the gallery's `thumbnail_size`, and read
at the size the gallery displays it.

#### Scenario: Thumbnail in the gallery grid
- **WHEN** a user opens the gallery index of `chemistry/pattern_formation`
- **THEN** the Gray-Scott thumbnail is the Rerun image of its `v` field at the last frame, with spots, filling the
  thumbnail at the 10:7 ratio of the configured `thumbnail_size` of 400x280

#### Scenario: Primary view of a time series system
- **WHEN** the `LotkaVolterraSystem` thumbnail is captured
- **THEN** it shows one time series view of the prey and predator populations, with no parameter views and no panels

#### Scenario: No placeholder thumbnails
- **WHEN** the gallery is built
- **THEN** no example's thumbnail is the `mkdocs-gallery` placeholder image, and the build fails and names any example
  whose scraper returned no image

### Requirement: The gallery is part of the documentation
The documentation build SHALL show every example with its code, its caption, and its screenshot in a gallery grouped by
field and subfield. Every registered system SHALL have a page with its pattern, whichever examples the build runs, and
the page of a system with an example SHALL show its screenshot above its pattern and link to the example. A default build
SHALL run a sample of three fast systems per field, and a
build of every example, `SKCOMPLEX_GALLERY=all`, SHALL run them all. A strict build of every example SHALL fail when an
implemented system has no screenshot. The page of an example SHALL show the description and the references of its system from the catalog.

#### Scenario: System page
- **WHEN** a user opens the documentation page of a system with an example
- **THEN** the page shows its screenshot and a link to its example above its class and equations

#### Scenario: System page without an example in the build
- **WHEN** a sample build leaves out the example of a system and a user opens the page of the system
- **THEN** the page shows its class and equations, without a screenshot or a link to an example

#### Scenario: Example references
- **WHEN** a user opens the gallery page of the `WolfSheepSystem` example
- **THEN** the page shows the description and the references the catalog gives for `WolfSheepSystem`

#### Scenario: Strict build without a screenshot
- **WHEN** the strict documentation build of every example finds a registered system without a cached or captured screenshot
- **THEN** the build fails and names the system
