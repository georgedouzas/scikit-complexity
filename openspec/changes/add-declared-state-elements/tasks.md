# Tasks

## 1. Primitive

- [x] 1.1 Find every caller of `_get_specs` and every reader of the signature of an element, and settle the reader of instance specs as the design says; verify the annotated elements give the same specs, by running `tests/core` and `tests/engine/compile`.
- [x] 1.2 Add the declaration from data to `skcomplex.core.elements`, with doctests; verify the same results and cloning scenarios.

## 2. Move the five places

- [x] 2.1 Record reference values of the five places as literal test values while the old code runs; verify the new tests pass on the current code.
- [x] 2.2 Move `_surrogate_system` of `SparseIdentificationAnalysis`; verify its reference values and `tests/analysis/test_surrogate_systems.py`.
- [x] 2.3 Move `_element_type` of `next_generation_reservoir`; verify its reference values and its test file.
- [ ] 2.4 Move `_culture`, `_metabolic_network`, and `_community` of the metabolism family; verify their reference values and `tests/models/biology/common/metabolism`.

## 3. Gate

- [ ] 3.1 Add the source check test; verify it finds nothing and fails on a planted `__signature__`.
- [ ] 3.2 Document the declaration in the user guide; verify the guide's blocks pass.
- [ ] 3.3 Push to `development`; verify every CI job passes and the full suite passes on Linux on a temporary branch.
