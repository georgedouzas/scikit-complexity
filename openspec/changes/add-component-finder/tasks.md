# Tasks

## 1. The finder

- [ ] 1.1 Add a private reader of component classes to `src/skcomplex/systems/_catalog.py`. It reads `__all__` of every
  package `__init__` under `skcomplex/core` and `skcomplex/systems` statically, keeps names with the suffix
  `Element`, `Interaction`, `Space`, or `Neighborhood`, and resolves each class to the public package nearest its
  defining module. Verify with tests in `tests/systems/test_catalog.py` for `MassPointElement` under
  `skcomplex.systems.physics.common.particles` and for the `GrowthInteraction` of `biology.common.populations` and of
  `biology.common.metabolism`.
- [ ] 1.2 Add `all_components(type_filter=None, package=None)` next to `all_systems`, export it from
  `skcomplex.systems`, and give it a docstring in the scikit-learn layout with examples for the core and for a field.
  Verify with tests for selecting by kind and by package, the sorted import paths, the `ValidationError` for an unknown
  kind, for `'law'`, or for an unknown package, and the `RegistryError` for a suffixed name that is not a component.
- [ ] 1.3 Test that selecting `package='systems/chemistry'` in a fresh interpreter imports no module of another field,
  and that every exported subclass of `BaseElement`, `BaseInteraction`, `BaseSpace`, and `BaseNeighborhood` appears in
  `all_components()`. Verify with `pytest tests/systems/test_catalog.py`.

## 2. Docstrings

- [ ] 2.1 Add to the body of the docstring of `Element` in `src/skcomplex/core/components/elements/_element.py` one
  sentence that sends the user to `all_components` before writing a subclass, and name `all_components` in its See
  Also section. Verify with the docstring tests and the doctests of the module.
- [ ] 2.2 Do the same in the docstring of `Interaction` in
  `src/skcomplex/core/components/interactions/_interaction.py`. Verify with the docstring tests and the doctests of the
  module.

## 3. Guide

- [ ] 3.1 Add to `docs/overview/user_guide/build_a_custom_system/agent_based_systems.md` the rules the agent solver
  enforces: when a member can be removed, given its values tagged with a quantity, and the names an agent value cannot
  take, read from the engine. Verify that the page builds and its examples run.
- [ ] 3.2 Show `all_components` in `docs/overview/user_guide/core_components/choosing_components.md` and in
  `docs/overview/user_guide/build_a_custom_system/defining_elements.md`, as the first step before defining an element,
  with the particle example. Verify that the examples run in the documents test group.

## 4. Checks

- [ ] 4.1 Run ruff, mypy, the convention check, and the tests of `tests/systems/test_catalog.py`,
  `tests/core/components/elements`, and `tests/core/components/interactions`, plus the documents test group. Verify
  that all pass.
