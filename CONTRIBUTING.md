# Contributing

This guide explains how to set up the tasks of the project, add a system to the catalog with its gallery example, and
send the work as a pull request. Contributions are welcome.

## Tasks

This section lists the commands of the development tasks. The project runs them with
[nox](https://nox.thea.codes/en/stable/), and the file `noxfile.py` at the root of the project defines each session.
PDM names the common ones as scripts.

### Documentation

- `pdm docs serve` or `pdm docs`: serve the documentation.
- `pdm docs build`: build the documentation.

### Formatting

- `pdm formatting all` or `pdm formatting`: format the code and the docstrings.
  - `pdm formatting code`: format only the code.
  - `pdm formatting docstrings`: format only the docstrings.

### Checks

- `pdm checks all` or `pdm checks`: run all checks.
  - `pdm checks quality`: check only the quality of the code, with ruff and the import contracts.
  - `pdm checks types`: check only the type annotations, with mypy.
  - `pdm checks dependencies`: check only the dependencies for vulnerabilities and for unused or missing ones.
  - `pdm checks security`: run only the security checks, with bandit.
  - `pdm checks docs`: check only the coverage and the format of the docstrings, with interrogate and pydoclint.
  - `pdm checks conventions`: check only the conventions of the sources, the tests, and the documentation, with
    `openspec/scripts/check_conventions.py`.

### Tests

- `pdm tests`: run the whole suite with coverage, including the slow tests, the docstring examples, the README, and
  the user guide.
- `nox -s tests -- <expression>`: run only the tests whose names match the expression.
- `nox -s quick_tests`: run a small subset of the fast tests in a few minutes, as CI does on every push.

The tests of a module `src/skcomplex/<package>/_<module>.py` live in `tests/<package>/test_<module>.py`, and every test
file mirrors a module this way. `pdm checks conventions` checks it.

### Changelog

- `pdm changelog`: build the changelog.

### Release

Before a release, name the runtime dependencies without a release in the last 24 months. Replace each one, or record an
exception in the Project Profile of `openspec/constitution.md`. The script below prints each such dependency with the
date of its last release.

```bash
python - <<'EOF'
import json, re, tomllib, urllib.request
from datetime import UTC, datetime, timedelta

limit = datetime.now(UTC) - timedelta(days=730)
for requirement in tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']:
    name = re.match(r'[\w.-]+', requirement).group()
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/json') as response:
        releases = json.load(response)['releases'].values()
    last = max(datetime.fromisoformat(file['upload_time_iso_8601']) for files in releases for file in files)
    if last < limit:
        print(name, last.date())
EOF
```

- `pdm release`: release a new version of the package. It writes the changelog, merges a release branch into `main`,
  and pushes the tag that starts the release pipeline.

## Development

This section gives the order of work on a change.

1. `git checkout -b new-branch-name` creates a branch for the change.
2. `pdm formatting` formats the code and the docstrings.
3. `pdm checks` runs all checks.
4. `pdm tests` runs the tests.
5. `pdm docs serve` shows the documentation, when the change touches it or the dependencies of the project.

### Where a part goes

This section decides where a new part belongs. One ordered test, the placement rule, decides where any part goes, and
`openspec/scripts/check_conventions.py` checks it mechanically. The page
[Extend the library](https://georgedouzas.github.io/scikit-complexity/overview/user_guide/extend_the_library/#where-a-new-part-goes)
of the user guide gives an example of each step.

1. A structural base, such as a `Base` class, the composite system, or a base where laws run per member, edge,
   hyperedge, or pair, goes to the core.
2. A part that the engine handles by type, through a compile pass, a kernel, a solver, a backend lowering, or an
   analysis that checks the class or its spec, goes to the core, and so does an operator generated from one function.
   It is named by its mathematics and must be cross-domain in principle.
3. Every other part is a recipe, expressible with core parts. Its users place it. Several fields:
   `skcomplex.systems.general.<concept>`. One field with several systems: `systems/<field>/common/<family>`. One
   system: that system's module. A field with its own term names it, and `general` uses neutral names.
4. A recipe that needs something the core cannot express reveals a missing core primitive, which enters by step 2.
   Engine support never lives inside a recipe.
5. The core never imports `skcomplex.systems`.

### Module layout

This section gives the order of a Python file. Every file holds the docstring, the licence header, the imports, the
constants block, the type aliases, the other module statements, and then the definitions. The constitution in
`openspec/constitution.md` states the rules, and `openspec/scripts/check_conventions.py` checks them. The checks cover the constants
block and the values derived from it, the symbolic functions written as `BuiltinFunction` subclasses, the class that
owns each helper, the action word that begins the name of every method, such as `get` or `check`, and the three kinds
of module functions that stay functions.

## Adding a system to the catalog

This section adds a ready-made system to the catalog. A system of the catalog is a subclass of `BaseSystem` whose name
ends in `System`. Its `__init__` takes the parameters of the model and stores each under its own name, and its hook
`_build_system` returns the `CompositeSystem` of its parts. Its body sets these class variables:

- `field`, the field path of a field and a subfield, such as `'engineering/processes'`,
- `topics`, the words that `all_systems` filters by,
- `references`, the works the model comes from, each an author and a year,
- `until` and `dt`, the tuned run settings that `simulate` reads when a call leaves them out.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import BaseSystem, CompositeSystem
from skcomplex.core.specs import StateSpec


class TankElement(Element):
    def __init__(
        self,
        label,
        visual=None,
        *,
        level: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ):
        super().__init__(label, visual=visual)
        self.level = level


class DrainingTankSystem(BaseSystem):
    """A tank that drains at a rate proportional to its level."""

    field = 'engineering/processes'
    topics = ('ode', 'decay')
    references = ('Torricelli 1643',)
    until = 5.0
    dt = 0.1

    def __init__(self, level=2.0, *, k=0.5, label='system'):
        super().__init__(label)
        self.level = level
        self.k = k

    def _build_system(self):
        k = self.k
        drain = RateInteraction('drain', target='tank', state='level', rate=lambda s: -k * s.target.level)
        return CompositeSystem(self.label, elements=[TankElement('tank', level=self.level)], interactions=[drain])


DrainingTankSystem().get_params()
```

`get_params` returns `k=0.5`, the label, and `level=2.0`, so `set_params`, `clone`, and the experiments of the library
change them by name. `all_systems` lists a class when its module lives in the package `skcomplex.systems` and its body
sets `field`. It reads the sources without importing them, so `field`, `topics`, and `references` must be literal
values. It raises a `RegistryError` for a class of the package whose name lacks the suffix `System` or whose field is
computed at run time.

A new system adds these files:

- the module of the class in its subfield package, such as `src/skcomplex/systems/engineering/processes/_tanks.py`,
- an import of the class in the `__init__.py` of the subfield, with its name in `__all__`,
- the tests of the class in the mirrored path, such as `tests/systems/engineering/processes/test_tanks.py`,
- an entry in the inventory `docs/overview/inventory.toml`,
- a gallery example, as the next section describes.

The module holds one system and the parts that only it uses. A part that several systems of one field share goes to
the `common` package of the field, and a part that several fields share goes to `skcomplex.systems.general`, as the
placement rule says.

`tests/systems/test_catalog.py` runs every listed class through one generic check. It builds the class with its
default parameters, simulates it for ten frames while a `RerunObserver` saves the run, requires a logged frame for
every simulated frame, and clones the system. Run `pytest tests/systems/test_catalog.py -k DrainingTankSystem` to run
the check on the new class.

The inventory entry sits in the table of its field path, and its name is the class name without `System` in snake
case, so `DrainingTankSystem` is `draining_tank`.

```toml
[engineering.processes]
draining_tank = { title = "Draining tank", sources = ["Torricelli 1643"], primitives = ["rates"], system = { components = ["element"], bindings = ["members"] }, priority = "example", status = "implemented" }
```

The `title` names the system in the gallery, the `sources` cite the works, and the `primitives` name the capabilities
of the core it exercises from the table `primitives` of the inventory. The `priority` is `'core'` when the system
exercises a set of primitives that no earlier entry exercises, and `'example'` otherwise. The `status` is
`'implemented'` once the class exists. An implemented entry does not write its time, space, laws, or domains, since the
check of the inventory reads them from `get_tags()` of the class. The documentation build runs that check and fails on
an entry that disagrees with the catalog.

The guide page
[Share a system](https://georgedouzas.github.io/scikit-complexity/overview/user_guide/share/) follows these steps for
one model, with every field of the inventory entry explained.

## System gallery

This section explains the gallery of the documentation. Every system of the catalog has one example there. The example
runs the system, streams it to Rerun, and adds an analysis figure computed from its own run. A screenshot of its Rerun
view becomes the thumbnail of the example.

### Where the examples live

- The examples are scripts in the format of `mkdocs-gallery` under `docs/examples/<field>/<subfield>/plot_<system>.py`,
  one per system, named after the inventory name of the system.
- The gallery tooling is the package `gallery_tools` in `docs`. It needs the dependency group `gallery`, which holds
  Pillow and Playwright: `pdm install -dG docs -dG gallery`.
- `nox -s gallery` runs the examples, captures their screenshots, and builds the documentation with them. It takes
  `--backend native`, the default, or `--backend web`, `--jobs` for parallel captures, and `--sample` to run only three
  fast systems per field, which is what `nox -s docs` and CI build.

### How the examples are captured

The tooling runs with `docs` on the path, as `PYTHONPATH=docs python -m gallery_tools <command>`.

- `scaffold` writes a scaffold for every system without an example, with the title of its inventory entry.
- `coverage` reports the systems without an example and the examples without a system, and `--strict` fails on either.
- `run <system> ...` runs the named examples, or all selected ones, and captures their screenshots into the cache.
  The default selection is the sample of three fast systems per field, and `--all` selects every example.

The run of every example, the whole script with its analysis, must stay within the budget: at most 300 frames on the
timeline of the recording, a recording of at most 20 MB, and at most 60 s. An example over budget fails.

The capture opens the recording with its blueprint in the Rerun viewer at the last frame, crops to the views, and
checks the image for blank views, notifications, and its size. The first view of the blueprint, the primary view,
becomes the thumbnail. Screenshots are cached under `docs/generated/gallery/_cache`, keyed by the source of the example,
the source of the system, the Rerun version, and the capture settings. Each entry holds `thumbnail.png` and a
`metadata.json` with the frames, the size of the recording, and the run time. The cache is not committed, and the
documentation build captures only the stale screenshots.

### Adding an example for a new system

This section writes the example of a new system, from the scaffold to the captured screenshot.

1. Add the class and its inventory entry, then run `PYTHONPATH=docs python -m gallery_tools scaffold` to write the
   scaffold. It imports the class from its subfield, sets `SYSTEM` to the inventory name, and runs the class with
   `simulate` and a `RerunObserver`.
2. Read the docstring of the class, its parameters from `get_params()`, and its tuned run settings from
   `get_run_settings()`.
3. Write the docstring of the example: a title, and a caption of the behaviour the system is known for, which the
   thumbnail must show. Delete the sentence that marks the example as a scaffold.
4. Choose the parameters, `until`, `dt`, the `interval` of the observer, and a `random_state` for a stochastic system,
   so that the run stays within the budget. `simulate` uses the tuned solver of the class unless the call names
   another.
5. Choose the primary view. Use `blueprint='default'`, with `primary='<origin>'` to put the most telling view first. A
   custom `rrb.Blueprint` is fine when no default view shows the behaviour, with a comment on why. A custom time series
   view must show the whole run, or the view follows the latest frames and cuts the curve off.

   ```python
   import rerun.blueprint as rrb

   # The view shows the whole run, not the default window that follows the latest frame.
   whole_run = rrb.TimeAxis(
       view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
   )
   view = rrb.TimeSeriesView(origin='/<element>/<state>', name='<name>', axis_x=whole_run)
   view
   ```

6. Add one analysis figure computed from the trajectory that `simulate` returns, with labelled axes, a title, and a
   sentence in the comment above it on the agreement to expect, such as a law, a conserved quantity, or a theoretical
   curve. Prefer the functions of `skcomplex.core.analysis`.
7. Capture it with `PYTHONPATH=docs python -m gallery_tools run <system>`, and open its `thumbnail.png` in the cache.
8. Run `ruff check` and `black --check` on the example.

The example `docs/examples/biology/ecology/plot_lotka_volterra.py` shows the finished form.

### Reviewing the screenshots

This section explains how a screenshot is accepted. Every screenshot is reviewed against its caption: at the size of
the gallery grid, the thumbnail must show the behaviour the caption states. A screenshot that does not is fixed in the
example, by its run, its primary view, or its caption, never by editing the image. Each review is one row of the list
of its field, with the system, what the thumbnail shows, what the analysis checks, and the budget of the run. The lists
of the first gallery are in `openspec/changes/archive/2026-10-21-add-model-gallery/reviews/<field>.md`.

## Commit message convention

This section gives the format of a commit message. Commit messages follow conventions based on the
[Angular style](https://gist.github.com/stephenparish/9941e89d80e2bc58a153#format-of-the-commit-message).

### Structure

```text
<type>(<scope>): <subject>

<body>

<footer>
```

### Example

```text
feat(directive): A new feature of code

A description of the new feature.
It contains **important** information.

Issue #10: https://github.com/namespace/project/issues/10
Related to PR namespace/other-project#15: https://github.com/namespace/other-project/pull/15
```

#### Guidelines

- The scope and the body are optional.
- The subject and the body must be valid Markdown.
- The body ends with trailers, such as references to issues and pull requests, or co-authors.
- The subject starts with an uppercase letter where that makes sense.
- The subject has no full stop at the end and no punctuation.
- The type is one of these:
  - `feat`: a new feature.
  - `fix`: a bug fix.
  - `docs`: a change of the documentation.
  - `style`: a change of the style or the format of the code.
  - `refactor`: a change that is neither a feature nor a bug fix.
  - `tests`: tests added or corrected.
  - `chore`: maintenance of the code.

## Pull request guidelines

This section explains how to send the work. Link every related issue in the message of the pull request. Fixup commits
keep the history readable during review.

```bash
git commit --fixup=SHA
```

Once all the changes are approved, squash the commits.

```bash
git rebase -i --autosquash main
```

Then force the push.

```bash
git push -f
```
