# Design

## Context

`BaseConfig` in `skcomplex.base` is the root that stores settings and speaks `check`, `prepare`, and `run`, and the root modules `base`, `types`, `specs`, and `errors` sit below every package. An audit of the library outside the models found three breaks in the hierarchy:
- four bases outside `BaseConfig`;
- 145 dataclasses spread over 60 modules;
- 18 enums and protocols beside their users.

The engine agent's branch made it worse by adding `BaseValue`, a second root for frozen values.

## Goals / Non-Goals

**Goals:**
- One root: every base class derives from `BaseConfig`.
- One kind of value: every dataclass is a frozen spec under `BaseSpec` in `skcomplex.specs`.
- One place for types: aliases, kinds, and protocols live in `skcomplex.types`.

**Non-Goals:**
- Changing what any class computes.
- The models library: its own dataclasses and enums are examples of the field and stay there, apart from those that are copies of a root one.
- Exceptions, which derive from `SkcomplexError` in `skcomplex.errors`.

## Decisions

### `BaseSpec` derives from `BaseConfig`
`BaseSpec` is a frozen dataclass and a `BaseConfig`:
- `get_params` reads the dataclass constructor, so `clone` works.
- `check()` validates the fields through `_check()` and returns the spec. It cannot set `_checked` on a frozen instance, so `BaseSpec` overrides `_is_checked` to pass.
- `prepare` and `run` keep the trivial hooks of the root: a spec is prepared against nothing and runs nothing.

The alternative was a separate value root, the `BaseValue` of the engine branch. It was rejected because it gives the library two roots.

### `skcomplex.specs` is a package
Moving 145 specs into one file would break the rule of one class per module. Instead, `skcomplex/specs/` holds one module per spec, and its `__init__` exports every spec, so `from skcomplex.specs import StateSpec` keeps working. The 20 existing specs are split out of `specs.py` the same way.

### What a spec may hold
A spec holds:
- its fields;
- `check()`;
- properties and methods that only read its own fields.

A method that does work for another class moves to that class as a private method, following the rule that helpers are methods of the class using them. An example is `KernelSpec._target_references`, which reads a problem.

### Mutable dataclasses
A mutable dataclass becomes either a frozen spec or a class, depending on whether it acts:
- **Holds data.** Passes edit it in place, as with `Block`, `Declarations`, `TableValues`, `FiniteVolume`, and `Topology`. It becomes a frozen spec, and the pass returns a copy made with `dataclasses.replace`.
- **Acts.** It keeps state between calls and runs, as with `Observer`, `Kernel`, `PairSearch`, `EdgeSearch`, `SplitOperator`, `Lowering`, `ReductionSchedule`, and `_Printer`. It becomes a class under its package's base. Its constructor stores settings, and its state goes into attributes that end in `_`.

### Names
- Every spec ends in `Spec`: `Problem` becomes `ProblemSpec`, `Frame` becomes `FrameSpec`, `CompiledRule` becomes `CompiledRuleSpec`.
- A private dataclass keeps its leading underscore, as in `_SegmentSpec`, and moves to `skcomplex.specs` only when it is a spec. A private record that only one method builds and reads is a tuple instead.
- Enums end in `Kind` and protocols in `Protocol`.

### Layering
Root modules import no package at run time. A spec whose field names a package class, such as a `ProblemSpec` holding a solver's form, imports that class under `TYPE_CHECKING`. The import-linter contracts set `exclude_type_checking_imports = true`.

### Order of work
1. Merge `engine-tree` and `engine-catalog` into `development`, rerunning the root-module scripts on them, and delete the branches.
2. Put the bases under `BaseConfig`.
3. Turn `specs.py` into a package and add `BaseSpec`.
4. Move the specs package by package, in the order core, analysis, experiments, then the engine's `problems`, `compile`, `solvers`, `backends`, `runtime`, and `viz`.
5. Move the kinds and protocols.
6. Update the models, the gallery, and the user guide.

Each step ends with ruff, lint-imports, and the tests of the packages it touched. The last step ends with the full session.

## Risks / Trade-offs

- **Broken imports in models and the gallery from the renames.** A script maps every old name to its new one and rewrites every import, with the full session and the gallery as the check.
- **Slower passes from frozen copies.** Copying is cheap next to compiling, and the full session's run time is the check.
- **`TYPE_CHECKING` imports hide a dependency of a spec on a package.** This is acceptable: the dependency exists only for type checkers, and a run-time import is still caught by the contracts.

## Migration Plan

This is a single sweep on `development`, committed per step. There is no deprecation alias for renamed names, since the library has no released users of these internals.
