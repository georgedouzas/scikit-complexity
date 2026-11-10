## Context

See proposal.md for the motivation. Today `src/skcomplex/core/components/system.py` holds `RunSettings`, `BaseSystem`,
and `System`, and `src/skcomplex/core/components/mass_action.py` holds `MassActionSystem`. The siblings `elements`,
`interactions`, `laws`, `spaces`, and `specs` are packages with `_base.py`, one private module per class, and an
`__init__` with an explicit `__all__`.

The constraints that shape the move:

- The constitution asks that a package's modules import one another downward, and the project rule is that a base
  depends only on bases. `BaseSystem` today names its subclass `System` twice. It annotates `system_`,
  `_build_system`, and `_pass_parameters` with `System`, and `_pass_parameters` checks `isinstance(system, System)` at
  run time. `System` derives from `BaseSystem`, so the two cannot sit in two modules that import downward unless the
  base stops naming `System`.
- The code never uses `if TYPE_CHECKING` imports, lazy imports other than for optional dependencies, or suppressions.
- mypy checks `src`, `tests`, and `docs`. Many of them read the parts of a composite through `.system_`, such as
  `system.prepare().system_.elements`, so typing `system_` as a bare `BaseSystem` would fail on all of them.
- `skcomplex.core.components` re-exports nothing, as the Project Profile states.
- The contract "Systems import only public modules" forbids `skcomplex.core.components.**` and allows
  `skcomplex.core.components.*`. So a catalog module may import from `skcomplex.core.components.systems` and not from
  its `_base` or `_system`.

## Goals / Non-Goals

**Goals:**

- One package for the systems of the core, with the layout of its siblings.
- A `_base` module that holds the base of systems alone and imports no other module of its package.
- No alias, no shim module, and every import site moved in the same change.

**Non-Goals:**

- No change to what a system does, to its public methods, or to its tags.
- No change to the catalog package `skcomplex.systems` beyond its imports.
- No fix of unrelated stale text in the main specs, such as the modules of `skcomplex.core.base` in `package-layout`.

## Decisions

### Name the package `systems`

The package is `skcomplex.core.components.systems`. Every sibling is plural, `elements`, `interactions`, `laws`,
`spaces`, and `specs`, and each holds many classes of one concept. The new package holds `BaseSystem`, `System`, and
`MassActionSystem`, and later general systems, so the plural describes it and mirrors its siblings.

The catalog `skcomplex.systems` shares the last name. The two do not collide. They differ in depth, Python resolves
them by full path, and both mean "a package of systems". The core one holds the bases and general systems, and the
catalog holds ready-made systems built on them. A catalog module writes `from ....core.components.systems import
BaseSystem`, and a user writes `from skcomplex.core.components.systems import System` beside `from
skcomplex.systems.biology.ecology import LotkaVolterraSystem`, which reads naturally.

Alternative considered: the singular `system`. It would keep `from skcomplex.core.components.system import System`
working and halve the import edits. It breaks the plural pattern of the siblings, and it gives a package whose name
reads as one system while it holds several. The user confirmed this name.

### `_base.py` holds `BaseSystem` and `RunSettings`

`RunSettings` is the typed mapping that `BaseSystem.get_run_settings` returns, and catalog systems that override that
method annotate their return with it. It belongs to the base contract and to nothing else, so it lives in `_base.py`.
It is a `TypedDict`, not a frozen spec, so it does not move to `skcomplex.core.specs`, and it differs from
`RunSettingsSpec`, which a translator writes. The package re-exports it, since the catalog uses it.

### The base takes the composite type as a type parameter

`BaseSystem` becomes `BaseSystem[CompositeT: BaseSystem]`, a PEP 695 generic like `BasePart[ContextT, InputT]`.
`system_`, `_build_system`, and `_pass_parameters` are typed with `CompositeT`. `System` derives from
`BaseSystem['System']` and `MassActionSystem` from `BaseSystem[System]`. So `_base.py` names no concrete system, and
`MassActionSystem().prepare().system_` stays typed as `System`.

A subclass that writes the bare `BaseSystem`, as most catalog systems and user code do, gets an unchecked composite,
the same way the docstring examples subclass the bare `BasePart` and `BaseConfig` today. So no catalog class must
change its bases, and no test or document that reads `.system_.elements` fails to type-check.

The run-time check in `_pass_parameters` becomes "the built value is a `BaseSystem` that is its own composite", that
is `isinstance(system, BaseSystem)` and `system._build_system() is system`. Only a composite builds itself, so the
check rejects what it rejects today: a value that is no system, and a system class that still builds its parts. The
message keeps naming the class of the system and of the value it built.

Alternatives considered:

- Keep `System` in `_base.py` beside `BaseSystem`. This keeps full typing with no type parameter, but goes against
  the request that `_base` hold the base alone.
- Type the composite as `BaseSystem`. This breaks type-checking of every read of the parts of `system_` in the
  source, the tests, and the documents, so each would need a narrowing `isinstance`.
- A typing protocol for the composite. The compilers and solvers take `System`, so a protocol would not be accepted
  where the composite goes next, and it would duplicate the attributes of `System`.
- An import of `_system` inside `_base`, or a cycle resolved by import order. Both break the downward rule and are
  the machinery the project avoids.

### One module per class

- `_base.py`: `RunSettings`, `BaseSystem`.
- `_system.py`: `System`, with its private methods, such as `_get_composite`, `_check_roles`, and `_join_tag`.
- `_mass_action.py`: `MassActionSystem`.

A search for subclasses of `BaseSystem` and `System` under `src/skcomplex/core` finds only `System` and
`MassActionSystem`. The class templates in `skcomplex.core.engine.translators._module_writer` and the doctest of
`_system_dynamics.py` write source text and define no class of the core. Their import strings move to the new path.

### `skcomplex.core.components` keeps re-exporting nothing

Its `__init__` holds only its docstring today, and the Project Profile says the package re-exports nothing so every
public name is imported from the subpackage that owns it. It stays that way. The docstring keeps listing systems
among the parts.

### Import contracts

The components layers contract lists `specs`, `mass_action`, `system`, `elements | interactions`, `laws`, `spaces`,
and `base`. It becomes `specs`, `systems`, `elements | interactions`, `laws`, `spaces`, and `base`. The contracts of
the catalog need no change, since `skcomplex.core.components.systems` matches the allowed `*` and its private modules
match the forbidden `**`.

A new layers contract with `containers = ["skcomplex.core.components.systems"]` and the layers `_mass_action`,
`_system`, `_base` enforces the downward rule inside the package. The sibling packages have no such contract, but
this one is the package whose base used to name its subclass, so the contract keeps the fix from regressing.

### Tests mirror the source

`tests/core/components/test_system.py` becomes `tests/core/components/systems/test_base.py` for the tests of
`BaseSystem` and `RunSettings` and `tests/core/components/systems/test_system.py` for the tests of `System`.
`tests/core/components/test_mass_action.py` becomes `tests/core/components/systems/test_mass_action.py`. The new
directory gets an `__init__.py` like its siblings. The test that `_base` imports no other module of its package is
the layers contract, not a test module, since the test tree checks no repository convention.

## Risks / Trade-offs

- [Every import site of a system class changes] → The edit is a mechanical rewrite of two module paths. The checks
  session, the doctests, and the markdown doctests of the user guide catch any site left behind, since the old modules
  no longer exist.
- [Users' code breaks] → The project makes breaking changes in 0.x without shims. The changelog entry of the
  conventional commit names the new path.
- [A bare `BaseSystem` subclass has an unchecked composite type] → It is the behaviour of a bare generic in mypy's
  default mode, and the run-time check still rejects a wrong composite. A subclass that wants the check names
  `BaseSystem[System]`.
- [The check builds a wrong composite's parts once more] → It only runs on the error path, where the system is
  about to fail anyway.
- [Two packages named `systems`] → The design states the difference, and the user guide names the full path wherever
  it imports either.

## Migration Plan

Replace `skcomplex.core.components.system` and `skcomplex.core.components.mass_action` with
`skcomplex.core.components.systems` in every import. Rolling back is reverting the change's commits.

## Decisions by the user

- Package name: `systems`, plural like its siblings (2026-11-09).
- Typing of the composite: `BaseSystem` takes the composite type as a type parameter (2026-11-09).
