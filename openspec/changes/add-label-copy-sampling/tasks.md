# Tasks

## 1. Core primitives

- [x] 1.1 Add `RegionScoreSpec`, verified by `tests/core/specs/test_region_score_spec.py` and its doctest
- [x] 1.2 Add the move `'copy'` and the region sampling path with its Numba kernel to `SamplingInteraction`, verified by
  `tests/core/components/interactions/test_region_scores.py` and its doctests

## 2. Biology realization

- [x] 2.1 Add `TissueSiteElement`, `CellAdhesionInteraction`, and the disordered lattice, verified by
  `tests/systems/biology/physiology/test_cellular_potts.py` and doctests
- [x] 2.2 Register `cellular_potts` with its observables, verified by `tests/systems/test_catalog.py -k cellular_potts`
- [x] 2.3 Add the inventory entry and the `label-copies` primitive, verified by `inventory_problems()` returning no
  problem
- [x] 2.4 Add the curated gallery example, verified by `gallery_tools coverage --strict` and a run of the example

## 3. Documentation and checks

- [x] 3.1 Document label copies in the lattice systems guide, verified by running its code blocks
- [x] 3.2 Run ruff, black, docformatter, mypy, and pydoclint on the changed files, and the tests of the touched modules
