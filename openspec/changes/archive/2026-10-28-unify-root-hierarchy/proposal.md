# Proposal

## Why

The library is meant to be one hierarchy of classes that work together. Today it is not:
- Four bases sit outside the root class `BaseConfig`: `BaseBackend`, `BaseSolver`, `BaseProblem`, and the engine's own `BaseEstimator`.
- About 145 dataclasses are scattered over analysis, the core, and the engine. They are specs in all but name, and 37 of them are mutable.
- Eight enums and ten protocols are types, yet they live in the modules that use them.

The root modules `skcomplex.base`, `skcomplex.specs`, and `skcomplex.types` now exist, so every class can be placed by what it is.

## What Changes

- Every base class of the library derives from `BaseConfig` in `skcomplex.base`. `BaseBackend`, `BaseSolver`, and `BaseProblem` move under it. `engine/base.py` with its `BaseEstimator` is deleted, and so is the `BaseValue` root of the engine branch.
- `skcomplex.specs` becomes a package with one module per spec. Its import path stays the same.
- A new `BaseSpec(BaseConfig)` is the base of every spec: a frozen dataclass whose fields are what it is and whose `check()` validates them. Every dataclass of the library becomes a `*Spec` under `BaseSpec` in `skcomplex.specs`:
  - declarations a class reads, such as `SupportSpec`, `KernelSpec`, `LammpsFix`, `Dialect`, and the `Call`, `Draw`, `Reduction`, and `Table` of `core.laws.utils`;
  - run records, such as `Frame`, `EventRecord`, `Census`, `AgentSnapshot`, `ControlSolution`, `ValidationReport`, and the `Series`, `Samples`, `Snapshots`, `JumpPath`, and `Signature` of analysis;
  - compiled parts, such as `Problem`, `Reference`, `Variable`, and the `Compiled*` classes;
  - mutable working objects that only hold data, such as `Block`, `Declarations`, and `TableValues`, which become frozen.
  **BREAKING**: every one of these classes is renamed with the suffix `Spec` and imported from `skcomplex.specs`.
- Mutable dataclasses that act, such as `Observer`, `Kernel`, `PairSearch`, `EdgeSearch`, `SplitOperator`, and `Lowering`, become classes under the base of their package, which derives from `BaseConfig`. They are not specs.
- A method of a dataclass that does work is moved to the class that uses the spec. A spec keeps only its fields, `check()`, and readers of its own fields.
- The enums and protocols move to `skcomplex.types`. Enums take the suffix `Kind`, so `Form` becomes `FormKind`, `Source` becomes `SourceKind`, and `DataKind` stays `DataKind`. Protocols take the suffix `Protocol`, so `Record` becomes `RecordProtocol` and `Jacobian` becomes `JacobianProtocol`. **BREAKING** for the renamed names.
- Before the sweep, the engine branches `engine-tree` and `engine-catalog` are merged into `development`, and the branches are deleted.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `base-interfaces`: the configurable bases derive from one root, `BaseConfig`, and values are specs under `BaseSpec` in `skcomplex.specs`. Types are kinds and protocols in `skcomplex.types`.
- `package-layout`: every class derives from the base it is, and every base derives from `BaseConfig`.

## Impact

- **Code:**
  - every module of `skcomplex.engine`, plus `core.laws.utils`, `analysis._base`, a few private analysis helpers, and `experiments`;
  - `skcomplex.specs` turns from a module into a package;
  - `skcomplex.types` gains the kinds and protocols.
- **Public names:** the dataclasses gain the suffix `Spec` and move to `skcomplex.specs`, and some enums and protocols are renamed. Models, tests, the gallery, and the user guide are updated in the same sweep.
- **Layering:** a spec whose fields name a class of a package imports that class only under `TYPE_CHECKING`, and import-linter excludes type-checking imports, so the root modules still import no package at run time.
- **Work in flight:** the engine agent's branches are merged first, so the sweep runs on one branch.
