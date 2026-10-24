# Design

## Context

- **Registration:** `engine/runtime/_catalog.py` keeps a `MODELS` registry. `register_model(field=..., tags=...,
  references=..., defaults=..., observables=...)` decorates every factory. `catalog`, `describe`, `create`, and `run`
  read the registry.
- **Model package:** `models/__init__.py` imports all nine field packages. Each field package imports its subfield
  packages, and each subfield package imports its modules, which registers every model.
- **Import cost:** importing `skcomplex.models` loads every formalism and its libraries: qutip, mesa (with pandas and
  matplotlib), LAMMPS, and deeptime.
- **Test session:**
  - pytest runs with `addopts = "--cov=src --cov=tests --doctest-modules --markdown-docs"`;
  - `nox -s tests` runs `pytest -n auto`;
  - collection takes 10 to 12 seconds in every worker;
  - the slowest test, `test_avalanches_same_statistics_on_a_sandpile_and_an_earthquake_model`, takes 652 seconds.

## Goals / Non-Goals

**Goals:**
- `import skcomplex.models` and `catalog()` load no formalism.
- A plain `pytest` run starts fast and skips the slow tests.

**Non-Goals:**
- Deferring imports inside modules. The constitution keeps lazy imports for optional dependencies only, and the
  maintainer chose the lazy catalog instead.
- Faster full collection. The full suite imports every module, through its tests and its doctests, so its collection
  stays bounded by the imports.

## Decisions

### The catalog reads registrations from the model sources

The first call to `catalog`, `describe`, `create`, or `run` scans the model sources with `ast` and builds one listing
entry per factory decorated with `register_model`. Each entry holds the name, field, tags, references, the
description (the first line of the factory's docstring), and the dotted module path.
- **Reading:** the scan reads the decorator's literal keyword arguments, and it rejects a computed one with a named
  error. `defaults` and `observables` may be computed, because only the imported registration uses them.
- **Using:** `create`, `run`, and `describe` import the entry's module, which registers the full entry in `MODELS`, and
  then work as today.
- **Cost:** the scan parses about 400 files once per process, about 0.5 seconds, measured during the implementation.
  That is much less than importing every model.
- **Alternative: a generated index file.** It was rejected because the file would duplicate the registrations, and a
  forgotten regeneration would leave it stale.

`models/__init__.py` stops importing the field packages, and it re-exports only the catalog functions. A field or
subfield package still imports its own modules, so `from skcomplex.models.physics.mechanics import pendulum` works.

### Test run options

- **`addopts`:** keeps `--doctest-modules --markdown-docs` and drops coverage.
- **`tests` session:** runs `pytest -n auto --slow --cov=src --cov=tests --cov-branch`, and then combines and reports
  coverage as it does today.
- **`tests/conftest.py`:**
  - adds `--slow`, next to the `--benchmarks` option it has;
  - registers the `slow` marker;
  - skips marked tests without the option.
- **Finding slow tests:** a full run with `--durations=0` on an idle machine lists every test above 10 seconds, which
  then gets the marker.

### Trimming the slowest tests

A test above 60 seconds is cut to a smaller system that still shows the behaviour it tests.
- **Avalanche test:** the sandpile shrinks from 128 by 128 for 150 000 steps to a size that still gives the
  avalanche exponents within the test's tolerance. Run time grows faster than linearly in the step count, and 5 000
  steps take 0.7 seconds.

## Risks / Trade-offs

- [The scan misses an unusual registration] → The listing-agreement test imports every module and compares.
- [A smaller avalanche system blurs the exponents] → The tolerance stays, and the size is chosen by measurement.
- [Contributors skip slow tests by default] → The gate runs them, and the skip reason names `--slow`.
