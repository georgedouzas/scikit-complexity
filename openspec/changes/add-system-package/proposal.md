## Why

Every concept of `skcomplex.core.components` is a package with a `_base` module, one private module per class, and a
public `__init__`. Systems are the exception. `components/system.py` holds `RunSettings`, `BaseSystem`, and the
composite `System` in one module, and `components/mass_action.py` holds `MassActionSystem` beside it. So the base of
systems shares a file with a concrete system, and a general system sits loose at the top of `components`. Systems
should follow the layout of their siblings, so the base of a system is found where every other base is.

## What Changes

- **BREAKING** New package `skcomplex.core.components.systems`, laid out like `elements`, `interactions`, `laws`, and
  `spaces`.
  - `_base.py` holds `BaseSystem` and `RunSettings`, the typed mapping its `get_run_settings` returns.
  - `_system.py` holds the composite `System`.
  - `_mass_action.py` holds `MassActionSystem`.
  - `__init__.py` re-exports `BaseSystem`, `MassActionSystem`, `RunSettings`, and `System` with an explicit `__all__`.
- **BREAKING** The modules `skcomplex.core.components.system` and `skcomplex.core.components.mass_action` are removed,
  with no alias and no shim module. Every import site in the source, the tests, the documents, the README, the
  docstring examples, and the generated module template moves to `skcomplex.core.components.systems`.
- `BaseSystem` no longer names `System`, so `_base.py` imports no module of its own package. `BaseSystem` takes the
  type of the composite it builds as a type parameter, as `BasePart` already takes its context and inputs. It checks
  at run time that the built composite is a `BaseSystem` that is its own composite. `System` and `MassActionSystem`
  name `System` as their composite. A subclass that names no type parameter still works, as a bare `BasePart` does
  today.
- `skcomplex.core.components` keeps re-exporting nothing, as the constitution asks.
- The layers contract of `skcomplex.core.components` in `pyproject.toml` replaces the layers `mass_action` and `system`
  with the one layer `systems`, and a new layers contract keeps the modules of the package importing downward.
- The tests move to `tests/core/components/systems/`, one test module per module of the package.
- The Project Profile of the constitution names `systems` in the order of the packages of `components`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `package-layout`: systems of the core become a package of `components` with a `_base` module, like the other parts,
  and the base of systems imports nothing else of its package.
- `base-interfaces`: general systems of the core, such as `MassActionSystem`, live in the systems package of the core.

## Impact

- Code: `src/skcomplex/core/components/system.py` and `mass_action.py` become the package
  `src/skcomplex/core/components/systems/`. Every module of `skcomplex.core` and `skcomplex.systems` that imports a
  system class changes its import. The module template of `skcomplex.core.engine.translators._module_writer` writes
  the new path.
- Tests: `tests/core/components/test_system.py` and `test_mass_action.py` move under
  `tests/core/components/systems/`, and every test that imports a system class changes its import.
- Documents: the user guide under `docs/overview`, the gallery examples and tools, `docs/generate_parts.py`,
  `docs/generate_systems.py`, `docs/inventory_checks.py`, `README.md`, and `CONTRIBUTING.md`.
- Configuration: the import contracts in `pyproject.toml`.
- Users: code that imports from `skcomplex.core.components.system` or `skcomplex.core.components.mass_action` must
  import from `skcomplex.core.components.systems`. The project makes breaking changes in 0.x without shims.
