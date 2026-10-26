# Proposal

## Why

Importing `skcomplex.models` takes 4.3 seconds, and the package's own modules account for 0.23 of them. The rest is
third-party libraries that load whether or not they are used:

| Library | Seconds |
|---|---|
| scipy | 0.81 |
| sage | 0.66 |
| qutip | 0.34 |
| pandas | 0.33 |
| scikit-learn | 0.29 |
| mesa | 0.25 |
| matplotlib | 0.21 |

They load because the model package imports every model module so the catalog can register each model. That pulls
in every formalism, and with them qutip, mesa (which brings pandas and matplotlib), LAMMPS, and deeptime.

The test session multiplies the cost:
- every xdist worker repeats the imports and the collection, which takes 10 to 12 seconds;
- coverage is always on through `addopts`;
- a few tests run for minutes, one of them for 652 seconds, so the suite takes about 26 minutes.

## What Changes

- **Lazy catalog:** the catalog lists models from their registration declarations without importing their modules.
  `create`, `run`, and `describe` import only the module of the model they name. Importing `skcomplex.models` no
  longer imports the subfield packages.
- **Coverage:** `addopts` no longer turns it on. The `tests` session of nox turns it on, so the gate keeps branch
  coverage, and a plain `pytest` run starts without it.
- **Slow tests:** they carry a `slow` marker and are skipped unless `--slow` is given. The `tests` session of nox
  passes `--slow`, so the gate still runs every test.
- **The slowest tests:** they are trimmed to the smallest size that still tests their behaviour, starting with the
  652-second avalanche test.
- **BREAKING:** `skcomplex.models` no longer re-exports its field packages as attributes. Code that reached a factory
  through `skcomplex.models.physics.mechanics` without importing it must import the package or use `create`.

## Capabilities

### New Capabilities

- `model-catalog`: how models are listed, described, created, and run by name, and what each of these imports.
- `test-suite`: the default test run, the slow marker, and the run the gate makes.

### Modified Capabilities

None.

## Impact

- **Code:** `engine/runtime/_catalog.py`, `models/__init__.py`, `pyproject.toml`, `noxfile.py`, `tests/conftest.py`, and
  the slow tests.
- **Behaviour:** model behaviour is unchanged, and every registered model is still listed and runs.
