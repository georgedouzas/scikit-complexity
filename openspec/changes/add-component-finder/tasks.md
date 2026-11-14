# Tasks

## 1. The finder

- [ ] 1.1 Add a private reader of part classes to `src/skcomplex/systems/_catalog.py`. It reads `__all__` of every package `__init__` under `skcomplex/core` and `skcomplex/systems` statically, keeps names with a kind suffix, and resolves each class to the public package nearest its defining module. Verify with tests in `tests/systems/test_catalog.py` for `MassPointElement` under `skcomplex.systems.physics.common.particles` and for one `GrowthInteraction` per field.
- [ ] 1.2 Add `all_components(type_filter=None, package=None)` next to `all_systems`, export it from `skcomplex.systems`, and give it a docstring with examples for the core and for a field. Verify with tests for selecting by kind and by package, the sorted import paths, the `ValidationError` for an unknown kind or package, and the `RegistryError` for a suffixed name that is not a part.
- [ ] 1.3 Test that selecting `package='systems/chemistry'` in a fresh interpreter imports no module of another field, and that every exported subclass of a part base appears in `all_components()`. Verify with `pytest tests/systems/test_catalog.py`.

## 2. Docstrings

- [ ] 2.1 Rewrite the docstring of `Element` in `src/skcomplex/core/components/elements/_element.py`: what an element is, how a subclass declares its states, parameters, and laws, a paragraph that sends the user to `all_components` first with an example that finds `MassPointElement`, and the `Unlike`, `Choose it when`, and `It appears in` paragraphs. Remove the paragraphs on ports and agents. Verify with the docstring tests and the doctests of the module.
- [ ] 2.2 Add the same pointer to `all_components` in the docstring of `Interaction` in `src/skcomplex/core/components/interactions/_interaction.py`, with an example that finds `ConstantForceInteraction`. Verify with the docstring tests and the doctests of the module.

## 3. Guide

- [ ] 3.1 Add to `docs/overview/user_guide/build_a_custom_system/agent_based_systems.md` the two rules that leave the `Element` docstring: when a member is removed, and the state names an agent cannot take. Verify that the page builds and its examples run.
- [ ] 3.2 Show `all_components` in `docs/overview/user_guide/core_components/choosing_components.md` and in `docs/overview/user_guide/build_a_custom_system/defining_elements.md`, as the first step before defining an element, with the particle example. Verify that the examples run in the documents test group.

## 4. Checks

- [ ] 4.1 Run ruff, mypy, the convention check, and the tests of `tests/systems/test_catalog.py`, `tests/core/components/elements`, and `tests/core/components/interactions`, plus the documents test group. Verify that all pass, apart from the mypy errors that predate this change.
