## 1. One branch

- [x] 1.1 Merge `engine-tree` and `engine-catalog` into `development`, rerun the root-module scripts on the merged tree, and delete both branches and worktrees. Verify with ruff, lint-imports, and the full test session.

## 2. One root

- [x] 2.1 Put `BaseBackend`, `BaseSolver`, and `BaseProblem` under `BaseConfig`, delete `engine/base.py` and any `BaseValue`, and add a test that every class named `Base*` outside the models derives from `BaseConfig`. Verify with `pytest tests/engine` and the new test.

## 3. Specs package

- [x] 3.1 Turn `skcomplex/specs.py` into a package with one module per spec, add `BaseSpec(BaseConfig)` as a frozen dataclass base whose `check()` validates the fields, and derive the existing specs from it. Set `exclude_type_checking_imports = true` in the import-linter contracts. Verify with `pytest tests/test_specs.py` and lint-imports.
- [ ] 3.2 Move the dataclasses of `core.laws.utils`, `analysis`, and `experiments` into `skcomplex.specs` as `*Spec` under `BaseSpec`, moving any method that does work into the class that uses the spec. Verify with the tests of those packages.
- [x] 3.3 Do the same for `engine.problems`. Verify with `pytest tests/engine/problems`.
- [ ] 3.4 Do the same for `engine.compile` and its passes, making `Block`, `Declarations`, `TableValues`, `FiniteVolume`, and `Topology` frozen and having the passes return copies. Verify with `pytest tests/engine/compile`.
- [ ] 3.5 Do the same for `engine.solvers`, `engine.backends`, `engine.runtime`, and `engine.viz`. Verify with `pytest tests/engine`.
- [ ] 3.6 Turn the acting mutable dataclasses (`Observer`, `Kernel`, `PairSearch`, `EdgeSearch`, `SplitOperator`, `Lowering`, `ReductionSchedule`, `_Printer`, and the rest of the audit) into classes under their package's base. Add a test that no module outside the models and `skcomplex.specs` defines a dataclass. Verify with the full test session.

## 4. Types

- [ ] 4.1 Move every enum and protocol outside the models to `skcomplex.types`, renaming enums to end in `Kind` and protocols to end in `Protocol`, and add a test that no other module defines one. Verify with the full test session.

## 5. Integration

- [ ] 5.1 Update the models, the gallery, the user guide, and the specs of `restructure-downstream-packages` that name the renamed classes. Verify with the full test session with doctests, the checks session, and the gallery build.
