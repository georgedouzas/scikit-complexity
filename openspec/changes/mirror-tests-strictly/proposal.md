# Proposal

## Why

The constitution says the test tree mirrors the source tree, with no exceptions. The test suite broke that rule: 20
files tested things other than a module of `skcomplex`. These were:
- the layout, docstring, documentation, and agnostic-core checks at the root of `tests`;
- `tests/docs`, which tested the gallery tooling, the inventory, and the compatibility page.

The maintainer deleted them. What they enforced must now come from the tools built for each job, and the specs must
stop naming checks that no longer exist.

## What Changes

- **Mirroring:** every test module tests one module of `skcomplex` and lives at the mirrored path. No test module
  checks repository conventions, documentation tooling, or documentation data.
- **Docstring examples:** pytest's doctest collection runs them, configured in `pyproject.toml` with no test module.
- **Documentation examples:** the Python blocks of `README.md` and the user guide run in the test session through
  `pytest-markdown-docs`, a development dependency. Blocks that build on earlier ones are tagged as continuations.
- **Import rules:** `import-linter` enforces them through contracts in `pyproject.toml`, run by the `checks` session:
  - the order of the core packages;
  - the core never importing the models;
  - the shared core never importing a formalism;
  - formalisms importing only the formalisms they declare;
  - models importing no private core module;
  - model packages staying self-contained.
- **Inventory check:** the docs build runs it, and the build fails when the inventory disagrees with the catalog or
  breaks the inventory rules.
- **Removed from the specs:** the requirements for definition order and export hygiene. No mature tool checks them, and
  the constitution's "Modules & Surface" rules already bind them.
- **Project Profile:** the gate's tools and the test-session sources are updated.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `package-layout`:
  - import rules enforced by import-linter;
  - test tree mirroring without exceptions;
  - examples run by doctest collection and `pytest-markdown-docs`;
  - two convention requirements removed.
- `formalism-plugins`: formalism import rules enforced by import-linter.
- `model-inventory`: the docs build runs the inventory check.

## Impact

- **Configuration:** `pyproject.toml`, which gains the pytest doctest options, the `pytest-markdown-docs` settings, the
  import-linter contracts, and the development dependencies.
- **Sessions and docs:** `noxfile.py`, the user guide and README code fences, which gain continuation tags,
  `docs/generate_models.py`, and the constitution's Project Profile.
- **Dependencies:** `pytest-markdown-docs` joins the `tests` group, and `import-linter` joins the `checks` group. No
  runtime dependency changes.
