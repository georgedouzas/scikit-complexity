# Tasks

## 1. Documentation decisions

- [x] 1.1 Say "in any field of science" in the README, the guide, and the package docstring
- [x] 1.2 Replace every semicolon of prose in the README, the guide, docstrings, comments, and messages
- [x] 1.3 Set `relaxed_headers` on superfences so `python continuation` blocks render on the site
- [x] 1.4 Keep the README as code only, with no output and no `print`
- [x] 1.5 Run the guide blocks at the documentation build and show the value of each top-level expression
- [x] 1.6 Remove hard-coded counts of the catalog from prose

## 2. Catalog functions

- [x] 2.1 Rename `create` to `create_system` across the code, tests, gallery, guide, and README
- [x] 2.2 Add the `until` argument to `Simulation`, used by `run()` when it gets none
- [x] 2.3 Replace `run` with `create_simulation`, which returns the unprepared simulation with the registered settings
- [x] 2.4 Rewrite every call of `run` in tests, examples, doctests, the guide, the README, and the gallery scaffold

## 3. Package layout

- [x] 3.1 Move the old `core` to `skcomplex.core.components`, and `engine`, `analysis`, `experiments`, `specs`, `base`,
  `errors`, and `types` into `skcomplex.core`, with relative imports recomputed
- [x] 3.2 Move `skcomplex.models` to `skcomplex.systems`, and the tests to `tests/core` and `tests/systems`
- [x] 3.3 Map absolute module names and paths in code, docs, configuration, and workflows
- [x] 3.4 Check that the import contracts of the checks session name the new packages and pass

## 4. Model to system

- [x] 4.1 Rename the catalog identifiers: `register_model`, `ModelEntry`, `ModelCatalog`, `ModelListing`, `MODELS`, and
  the private modules that hold them
- [x] 4.2 Rename "model" to "system" in every other identifier, skipping identifiers of other libraries
- [x] 4.3 Rename the Models section and pages of the documentation, and its generator
- [x] 4.4 Rename "model" to "system" in all prose, docstrings, comments, and messages, keeping cited titles
- [x] 4.5 Review every changed line that touches `mesa`, `cobra`, `pysd`, and SBML

## 5. User guide

- [ ] 5.1 Reorder the guide by task: select, change, build, simulate, observe, analyse, experiment, extend
- [ ] 5.2 Rewrite each section to open with what the reader achieves, then short code, keeping every topic
- [ ] 5.3 Compare the topics before and after, and keep every executed block passing

## 6. Verification

- [x] 6.1 Ruff, black, mypy, and pydoclint pass
- [ ] 6.2 Every module imports, and the structural, catalog, and runtime tests pass
- [ ] 6.3 The documentation builds with the executed guide
- [ ] 6.4 The full suite passes in CI

## Workflow follow-up

- Sync the deltas at archive, then map the old paths and the word model in the main specs with the same pass.
