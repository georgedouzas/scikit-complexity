# Proposal

## Why

A user who wants a particle has no way to find `MassPointElement`. Systems have `all_systems`, but the element,
interaction, space, and neighbourhood classes of the core and of the systems packages have no finder. Since the core
API redesign, the core holds only `Element`, `Interaction`, and a few spaces and neighbourhoods, and every
field-specific component lives in `skcomplex.systems.<field>.common` or `skcomplex.systems.general`, so the parts a
user wants are spread over many packages. Users then write `Element` and `Interaction` subclasses by hand for parts that
already exist.

## What Changes

- A new public function `all_components` finds component classes, in the role of `sklearn.utils.all_estimators`. It
  returns the import path and the class of every public element, interaction, space, and neighbourhood, from the core
  and from the systems packages. It selects by kind and by package, so a user can ask for the elements of the core, of
  `systems/physics`, or of `systems/physics/common/particles`.
- Laws are values that interactions state, not components, and times are no longer public, so neither is a kind of
  the finder.
- The import path it returns is the line the user writes, such as
  `skcomplex.systems.physics.common.particles.MassPointElement`, so two classes with the same name in different
  fields stay apart.
- The docstrings of `Element` and `Interaction` say in their body to look for an existing part first and name
  `all_components` in their See Also section. They keep the scikit-learn layout that the redesign gave them.
- The guide page on choosing components and the guide page on defining elements show the finder.
- The guide page on agent based systems states the rules the agent solver enforces on members, the removal of a member
  that holds a quantity and the names an agent value cannot take, which no page states yet.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-catalog`: adds the requirement that component classes are discovered with `all_components`, beside the
  existing requirement for `all_systems`.
- `user-documentation`: adds the requirement that the docstrings of the core classes users subclass point to the
  finder, and that the guide states the rules of agent members.

## Impact

- New public function in `skcomplex.systems`, next to `all_systems`, in `src/skcomplex/systems/_catalog.py`. No
  breaking change.
- Docstrings of `Element` and `Interaction` in `skcomplex.core.components`.
- Guide pages `core_components/choosing_components.md`, `build_a_custom_system/defining_elements.md`, and
  `build_a_custom_system/agent_based_systems.md`.
- New tests in `tests/systems/test_catalog.py`, which mirrors `src/skcomplex/systems/_catalog.py`.
- The redesign `redesign-core-api` parked this change until the components settled. They have, and this change is
  rebased onto them.
