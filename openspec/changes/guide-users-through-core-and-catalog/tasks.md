# Tasks

## 1. Declarations

- [ ] 1.1 Add `elements` and `states` to `RoleSpec`, with doctests, defaulting to any element and no states; verify `tests/test_specs.py` and that every existing model still compiles through `tests/models/test_catalog.py`.
- [x] 1.2 Add `get_requirements()` to `BaseInteraction` and `get_states()` to `BaseElement`, settled against how `add-declared-state-elements` holds the states; verify the reading requirements scenario.
- [x] 1.3 Declare the roles of the 42 interactions of `skcomplex.core.interactions`; verify the every core interaction declares scenario.

## 2. Validation

- [x] 2.1 Report `role-element` from the declarations, in the one pass; verify the wrong kind and declared missing state cases.
- [x] 2.2 Catch a missing attribute of a role while building laws and report it as the same issue; verify the missing state scenario with an undeclared interaction, and the other errors are kept scenario.
- [ ] 2.3 Validate every model of the catalog; verify the prebuilt models stay valid scenario, correcting any declaration that rejects a valid use.

## 3. Generated pages

- [ ] 3.1 Add elements and interactions to `docs/generate_compatibility.py`; verify the interactions in the table scenario with a strict build.
- [ ] 3.2 Add the catalog index and the selection and component blocks of the model pages to `docs/generate_models.py`, and the navigation; verify the index follows the catalog and model page scenarios with a strict build and the gallery sample.

## 4. README and guide

- [x] 4.1 Add the documentation test that reads the Python blocks of the README and the guide and fails on an import from a field module, expected to fail at this point; verify it names the 62 current imports, 3 in the README and 59 in the guide.
- [x] 4.2 Rewrite the usage section of the README around the two paths, with the custom model from the core; verify the first screen and core alone scenarios.
- [ ] 4.3 Rewrite the concepts opening of the guide around the two paths, and add the section on choosing compatible components, using `get_requirements()`, `get_states()`, and the report; verify the asking before building and getting it wrong scenarios.
- [ ] 4.4 Replace the imports from field modules in the guide, section by section: a small `Element` from the core where a model's part was only a convenience, and `create` by name where a section documents a family; verify the test of 4.1 passes and all guide blocks pass.
- [ ] 4.5 Review the 31 gallery examples that import from field modules; keep those whose subject is a model's own analysis or parts, and list them in the review; verify the gallery coverage check and the sample build.

## 5. Gate

- [ ] 5.1 Run formatting, checks, the docs build, and tests through CI and the full suite on Linux; verify all pass.
