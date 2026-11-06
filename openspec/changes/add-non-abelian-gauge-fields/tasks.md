# Tasks

## 1. Core primitives

- [x] 1.1 Add `MatrixGroupSpec` with components, Haar and near-identity draws, projection, and `holonomy_of`, verified by
  `tests/core/specs/test_matrix_group_spec.py` and its doctests
- [x] 1.2 Add `StateSpec(group=...)`, carry it through `VariableSpec` and the layout pass, verified by the matrix group
  tests in `tests/core/specs/test_state_spec.py`
- [x] 1.3 Add `boundaries_` and `boundary_signs_` to `CellComplexSpace`, verified by the boundary tests in
  `tests/core/components/spaces/test_cell_complex.py`
- [x] 1.4 Let drives change states of several components with the space by keyword, and fix the shape of drives on
  network edges, verified by the matrix tests in `tests/core/components/laws/test_drive.py`
- [x] 1.5 Add `HolonomyScoreSpec` and its sampling in `SamplingInteraction`, verified by
  `tests/core/specs/test_holonomy_score_spec.py` and the SU(2) and SU(3) tests in
  `tests/core/components/interactions/test_sampling.py`

## 2. Physics realization

- [x] 2.1 Add `IsospinLinkElement`, `ColorLinkElement`, the matrix Wilson action in `WilsonSamplingInteraction`, and
  matrix loops in `GaugeAnalysis`, verified by `tests/systems/physics/waves/test_yang_mills.py` and doctests
- [x] 2.2 Register `lattice_yang_mills` with its observables, verified by `tests/systems/test_catalog.py -k yang_mills`
- [x] 2.3 Add the inventory entry, the `matrix-groups` primitive, and the `matrix` domain, verified by
  `inventory_problems()` returning no problem
- [x] 2.4 Add the curated gallery example, verified by `gallery_tools coverage --strict` and a run of the example

## 3. Documentation and checks

- [x] 3.1 Document matrix links in the cell complexes and link fields guide, verified by running its code blocks
- [x] 3.2 Run ruff, black, docformatter, mypy, and pydoclint on the changed files, and the tests of the touched modules
