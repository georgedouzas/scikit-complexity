# Contributing

Contributions are welcome, and they are greatly appreciated.

## Tasks

This project uses [nox](https://nox.thea.codes/en/stable/) to run development tasks. Please check the `noxfile.py` at the root of
the project for more details. You can run any of the following commands and subcommands that corresponds to a particular task:

### Documentation

- `pdm docs serve` or `pdm docs`: Serve the documentation.
- `pdm docs build`: Build locally the documentation.

### Formatting

- `pdm formatting all` or `pdm formatting`: Format both the code and docstrings.
  - `pdm formatting code`: Format only the code.
  - `pdm formatting docstrings`: Format only the docstrings.

### Checks

- `pdm checks all` or `pdm checks`: Run all checks.
  - `pdm checks quality`: Check only code quality.
  - `pdm checks types`: Check only type annotations.
  - `pdm checks dependencies`: Check only for vulnerabilities in dependencies.
  - `pdm checks security`: Run only security checks with bandit.
  - `pdm checks docs`: Check only docstring coverage with interrogate.

### Tests

- `pdm tests`: Run the tests.

### Changelog

- `pdm changelog`: Build the changelog.

### Release

Before a release, name the runtime dependencies without a release in the last 24 months. Replace each one, or record an
exception in the Project Profile of `openspec/constitution.md`:

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

- `pdm release`: Release a new Python package with an updated version.

## Development

The next steps should be followed during development:

- `git checkout -b new-branch-name` to create a new branch and then modify the code.
- `pdm formatting` to auto-format the code and docstrings.
- `pdm checks` to apply all checks.
- `pdm tests` to run the tests.
- `pdm docs serve` if you updated the documentation or the project dependencies to check that everything looks as expected.

### Adding a model to the catalog

A model of the library is a function that returns a system, registered with `register_model`. Declare the element and
interaction types, write the function, and register it with its field path, references, and default run settings.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.system import System
from skcomplex.models import create, register_model
from skcomplex.specs import ParameterSpec, StateSpec


class TankElement(Element):
    def __init__(
        self,
        label,
        *,
        level: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        k: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label)
        self.level = level
        self.k = k

    def _get_laws(self, s):
        return [RateLaw(s.level, -s.k * s.level)]


@register_model(field='examples/processes', references=('Torricelli 1643',), defaults={'until': 5})
def draining_tank(level: float = 2.0) -> System:
    """Return a tank that drains at a rate proportional to its level."""
    return System(elements=[TankElement('tank', level=level)])


print(create('draining_tank').elements[0].level)
```

Every registered function needs defaults for all of its arguments, so the catalog can create, compile, simulate,
visualize, and clone every model through the same interface.

## Model gallery

Every registered model has one example in the gallery of the documentation. The example runs the model, streams it to
Rerun, and adds an analysis figure computed from its own run. A screenshot of its Rerun view becomes the example's
thumbnail.

### Where the examples live

- The examples are scripts in the format of `mkdocs-gallery` under `docs/examples/<field>/<subfield>/plot_<model>.py`,
  one per registered model, named after the model.
- CI runs `nox -s quick_tests`, a small subset of the fast tests, in a few minutes. Run `nox -s tests` for the
  whole suite before a release.
- The gallery tooling is the `gallery_tools` package in `docs`. It needs the `gallery` dependency group, which holds
  Pillow and Playwright: `pdm install -dG docs -dG gallery`.
- `nox -s gallery` runs the examples, captures their screenshots, and builds the documentation with them. It takes
  `--backend native` (the default) or `--backend web`, `--jobs` for parallel captures, and `--sample` to run only three
  fast models per field, which is what `nox -s docs` and CI build.

### How the examples are captured

Run the tooling with `docs` on the path, `PYTHONPATH=docs python -m gallery_tools <command>`:

- `scaffold` writes a scaffold for every model without an example, from the model's catalog entry.
- `coverage` reports the models without an example and the examples without a model, and `--strict` fails on either.
- `run <model> ...` runs the named examples, or all selected ones, and captures their screenshots into the cache.
  The default selection is the sample of three fast models per field, and `--all` selects every example.

The run of every example, the whole script including its analysis, must stay within the budget: at most 300 frames on
the recording's timeline, a recording of at most 20 MB, and at most 60 s. An example over budget fails.

The capture opens the recording with its blueprint in the Rerun viewer at the last frame, crops to the views, and
checks the image for blank views, notifications, and its size. The first view of the blueprint, the primary view,
becomes the thumbnail. Screenshots are cached under `docs/generated/gallery/_cache`, keyed by the example's source, the
model's source, the Rerun version, and the capture settings. Each entry holds `thumbnail.png` and a `metadata.json` with
the frames, the recording size, and the run time. The cache is not committed, and the documentation build captures
only the stale screenshots.

### Adding an example for a new model

1. Register the model, then run `PYTHONPATH=docs python -m gallery_tools scaffold` to write its scaffold.
2. Read the model's factory and docstring, and its catalog entry with
   `python -c "from skcomplex.models import describe; print(describe('<model>'))"`.
3. Write the docstring of the example: a title, and a caption of the behaviour the model is known for, which the
   thumbnail must show. Delete the sentence that marks the example as a scaffold.
4. Create the model by name with `create`, and run it with `Simulation`, choosing `dt`, `until`, the observer's
   `interval`, and a seed so that the run stays within the budget. When the catalog entry registers a solver in its
   defaults, pass it to `Simulation`, for example `solver=ScipyODE(method='LSODA')` or `solver=RK4(dt=0.01)`, since
   `create` builds only the system.
5. Choose the primary view. Use `blueprint='default'`, with `primary='<origin>'` to put the
   most telling view first. A custom `rrb.Blueprint` is fine when no default view shows the behaviour, with a comment on why.
   A custom time series view must show the whole run, or the view follows the latest frames and cuts the curve off:

   ```python
   # The view shows the whole run, not the default window that follows the latest frame.
   whole_run = rrb.TimeAxis(
       view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
   )
   view = rrb.TimeSeriesView(origin='/<element>/<state>', name='<name>', axis_x=whole_run)
   ```

6. Add one analysis figure computed from `simulation.trajectory_`, with labelled axes, a title, and a sentence in the
   comment above it on the agreement to expect, such as a law, a conserved quantity, or a theoretical curve. Prefer the
   helpers of `skcomplex.core.analysis`.
7. Capture it with `PYTHONPATH=docs python -m gallery_tools run <model>`, and open its `thumbnail.png` in the cache.
8. Run `ruff check` and `black --check` on the example.

### Reviewing the screenshots

Every screenshot is reviewed against its caption: at the size of the gallery grid, the thumbnail must show the
behaviour the caption states. A screenshot that does not is fixed in the example, by its run, its primary view, or its
caption, never by editing the image. Each review is recorded as one row of the field's list in
`openspec/changes/add-model-gallery/reviews/<field>.md`, with the model, what the thumbnail shows, what the analysis
checks, and the budget of the run.

## Commit message convention

Commit messages follow conventions based on the [Angular
style](https://gist.github.com/stephenparish/9941e89d80e2bc58a153#format-of-the-commit-message).

### Structure

```bash
<type>(<scope>): <subject>

<body>

<footer>
```

### Example

```
feat(directive): A new feature of code

A description of the new feature.
It contains **important** information.

Issue #10: https://github.com/namespace/project/issues/10
Related to PR namespace/other-project#15: https://github.com/namespace/other-project/pull/15
```

#### Guidelines

- Scope and body are optional.
- Subject and body must be valid Markdown.
- Body must add trailers at the end, for example issues and PR references or co-authors.
- Subject must have proper casing, i.e. uppercase for first letter if it makes sense.
- Subject must have no dot at the end and no punctuation.
- Type can be:
  - `feat`: New feature implementation.
  - `fix`: Bug fix.
  - `docs`: Documentation changes.
  - `style`: Code style or format changes.
  - `refactor`: Changes that are not features or bug fixes.
  - `tests`: Test additions or corrections.
  - `chore`: Maintenance code changes.

## Pull Request guidelines

Link to any related issue in the Pull Request message. We also recommend using fixups:

```bash
git commit --fixup=SHA
```

Once all the changes are approved, you can squash your commits:

```bash
git rebase -i --autosquash master
```

And force-push:

```bash
git push -f
```
