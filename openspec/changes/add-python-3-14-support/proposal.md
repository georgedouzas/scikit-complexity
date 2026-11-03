# Proposal

## Why

The package refuses Python 3.14 (`requires-python = ">=3.12, <3.14"`), so `pip install scikit-complexity` fails with "no
matching distribution" on it. Only `deeptime` stands in the way: 0.4.5 has no 3.14 wheels and does not compile here, and
it has published no release since April 2025, which falls short of the project's aim of frequently updated dependencies.

## What Changes

- Declare Python 3.14 as supported in the manifest, the classifiers, the lock file, the README, and the Project Profile
  of the constitution.
- **BREAKING for code that reaches into the library**: replace `deeptime` with `pydtmc` as the library behind
  `MarkovAnalysis`, `PassageAnalysis`, and `EmergenceAnalysis`. The public attributes of these analyses keep their names
  and meaning, and their results keep the values `deeptime` gives within numerical tolerance.
- Add the one method `pydtmc` lacks, the reversible maximum-likelihood fit, as a private method of `MarkovAnalysis`.
- Keep feature parity: no model or analysis is disabled by Python version.
- Cap `xgi` below 1.0. A fresh install today gets 1.0.0, under which `hypergraph_kuramoto` raises a `TypeError`, on the
  published 0.2.0 too, while the lock file hides it by holding 0.10.2.
- Run the fast tests on 3.14 in CI on both platforms, in addition to 3.12 on Linux and 3.13 on macOS.
- Keep the documentation build on Python 3.13, because the latest `mkdocs-gallery` reads `ast.Str`, which 3.14 removed.

## Capabilities

### New Capabilities

- `python-support`: the Python versions the package declares, what must hold for a version to be declared, and how each
  declared version is verified.

### Modified Capabilities

- `model-catalog`: listing models must not load the new Markov library, in place of `deeptime`.
- `analysis-adapters`: the default-install requirement no longer names `deeptime`.

## Impact

- Source: `src/skcomplex/analysis/_markov.py` and `_passage.py`, with their tests and reference values.
- Packaging: `pyproject.toml` (`requires-python`, classifiers, `deeptime` out, `pydtmc` in, `xgi`, the deptry list) and a
  relocked `pdm.lock`, in which the versions of other dependencies may move to reach 3.14 wheels.
- Tooling: `noxfile.py` (`PYTHON_VERSIONS`, the Python of the documentation sessions), `.github/workflows/ci.yml` (test
  matrix, installed Python versions), and `release.yml` (the Python of the documentation job). Two more test jobs per CI
  run.
- Documentation: `README.md`, `docs/overview/user_guide.md`, and the Project Profile in `openspec/constitution.md`.
- Release: the next release follows this change, and ships the Models tab and the `xgi` cap with it.
