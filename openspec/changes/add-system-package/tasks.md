## 1. Systems package

- [x] 1.1 Create `src/skcomplex/core/components/systems/_base.py` with `RunSettings` and `BaseSystem[CompositeT: BaseSystem]` moved from `components/system.py`, typed with `CompositeT` for `system_`, `_build_system`, and `_pass_parameters`, and with the check that the built value is a `BaseSystem` that is its own composite, and verify with `grep` that it imports no module of its package
- [x] 1.2 Create `systems/_system.py` with `System` deriving from `BaseSystem['System']`, and `systems/_mass_action.py` with `MassActionSystem` deriving from `BaseSystem[System]`, each importing downward, and update their module docstrings and doctests to import from `skcomplex.core.components.systems`
- [x] 1.3 Create `systems/__init__.py` with a one-line docstring and an explicit `__all__` that re-exports `BaseSystem`, `MassActionSystem`, `RunSettings`, and `System`, delete `components/system.py` and `components/mass_action.py`, and verify that importing either old module raises `ModuleNotFoundError`
- [x] 1.4 Move the tests of `BaseSystem` from `tests/core/components/test_system.py` to `tests/core/components/systems/test_base.py`, the tests of `System` to `tests/core/components/systems/test_system.py`, and `test_mass_action.py` to `tests/core/components/systems/test_mass_action.py`, add `tests/core/components/systems/__init__.py`, add a test that a subclass building a system class that is not its own composite raises `ValidationError`, and verify `pytest tests/core/components/systems` passes

## 2. Import sites

- [x] 2.1 Replace every import of `skcomplex.core.components.system` and `skcomplex.core.components.mass_action`, relative or absolute, in `src/skcomplex/core`, including the docstring examples, and the import strings of the module template in `engine/translators/_module_writer.py`, and verify `pytest --doctest-modules src/skcomplex/core` and `pytest tests/core` pass
- [x] 2.2 Replace the same imports in `src/skcomplex/systems` and `tests/systems`, so the catalog imports `BaseSystem`, `RunSettings`, `System`, and `MassActionSystem` from `skcomplex.core.components.systems`, and verify `pytest tests/systems` and the catalog doctests pass
- [x] 2.3 Replace the same imports in `tests/conftest.py` and every other test module, and verify `grep -rn "components\.system\b\|components\.mass_action" src tests` finds nothing
- [x] 2.4 Replace the same imports in `docs/overview`, `docs/examples`, `docs/gallery_tools`, `docs/generate_parts.py`, `docs/generate_systems.py`, `docs/inventory_checks.py`, `README.md`, and `CONTRIBUTING.md`, and verify the markdown doctests of `README.md` and the user guide pass and `grep` over `docs`, `README.md`, and `CONTRIBUTING.md` finds no old path

## 3. Contracts and profile

- [x] 3.1 In `[tool.importlinter]` of `pyproject.toml`, replace the layers `mass_action` and `system` of the components contract with `systems`, and add a layers contract for the container `skcomplex.core.components.systems` with `_mass_action`, `_system`, and `_base`, and verify `lint-imports` passes with `PYTHONPATH=src`
- [x] 3.2 Update the Layout and Extension points of the Project Profile in `openspec/constitution.md` to name the package `systems` in the order of the packages of `components`, and verify `grep` finds no `system` package or `mass_action` module left in the profile
- [x] 3.3 Commit the work as a conventional `refactor!` commit whose body names the new import path, and verify `nox -s changelog` lists it as a breaking change

## 4. Verification

- [x] 4.1 Run `nox -s checks` for `src`, `tests`, and `docs`, including ruff, mypy, and the import contracts, the full test session with doctests, and the strict docs build, and fix every finding without a suppression
