# Proposal

## Why

A user who wants a particle has no way to find `MassPointElement`. Systems have `all_systems`, but the hundreds of
element, interaction, law, and space classes of the core and the systems packages have no finder, so users subclass
`Element` by hand for parts that already exist. The docstring of `Element` makes it worse: it reads as a manual for
implementers, with ports, agent rules, and names that Mesa reserves, and never tells the user to look for an existing
part first.

## What Changes

- A new public function `all_components` finds part classes, in the role of `sklearn.utils.all_estimators`. It
  returns the import path and the class of every public element, interaction, law, space, neighbourhood, and time,
  from the core and from the systems packages. It selects by kind and by package, so a user can ask for the elements of
  the core, of `systems/physics`, or of `systems/physics/common/particles`.
- The import path it returns is the line the user writes, such as
  `skcomplex.systems.physics.common.particles.MassPointElement`, so two classes with the same name in different
  fields stay apart.
- The docstring of `Element` becomes short: what an element is, how a subclass declares its states and parameters,
  and a pointer to `all_components` before writing a subclass. The paragraphs on ports and agents leave the docstring,
  since the guide pages on ports and connections and on agent based systems already cover them. The docstring of
  `Interaction` gets the same pointer.
- The guide page on choosing components and the guide page on defining elements show the finder.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-catalog`: adds the requirement that part classes are discovered with `all_components`, beside the existing
  requirement for `all_systems`.
- `user-documentation`: adds the requirement that the docstrings of the core classes users subclass to define parts
  point to the finder first and leave usage details to the guide.

## Impact

- New public function in `skcomplex.systems`, next to `all_systems`. No breaking change.
- Docstrings of `Element` and `Interaction` in `skcomplex.core.components`.
- Guide pages `core_components/choosing_components.md` and `build_a_custom_system/defining_elements.md`.
- New tests in `tests/systems/test_catalog.py`, which mirrors `src/skcomplex/systems/_catalog.py`.
