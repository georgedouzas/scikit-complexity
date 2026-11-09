## Why

The two core system classes do not say what they are. `System` is the general composite of elements, interactions, and
nested systems, but its name reads as the base of every system. `MassActionSystem` is a system given as species and
reactions, but it is named after one rate law, and "mass" is a physics term in a core that serves every field. The
user decided to name each class after what it is, and to describe the rate law of reactions in mathematical words.

## What Changes

- **BREAKING** `System` in `core/components/systems/_system.py` becomes `CompositeSystem` in `_composite.py`. It is the
  composite of elements, interactions, and nested systems on a space and in a time. `BaseSystem[CompositeT]` keeps its
  type parameter, which every subclass now binds to `CompositeSystem`.
- **BREAKING** `MassActionSystem` in `_mass_action.py` becomes `ReactionNetworkSystem` in `_reaction_network.py`. It
  is a system given as species and reactions, with stoichiometric coefficients and rate constants, and it builds a
  `CompositeSystem` as rate equations, jumps, or Langevin equations. "Reaction network" is the mathematical name of
  the formalism. `ReactionNetworkSystem` stays in the core only until `move-recipes-to-systems` moves it, with
  `ReactionInteraction`, to `skcomplex.systems.general.reactions`.
- The core describes the rate law of a reaction mathematically. The propensity is a rate constant times the product of
  the reactant amounts, each raised to its stoichiometric coefficient. `ReactionNetworkSystem` and
  `ReactionInteraction` say so, and name field terms only as examples: the law of mass action in chemistry,
  homogeneous mixing in epidemiology, and the encounter term of Lotka-Volterra in ecology.
- **BREAKING** The private method `ReactionInteraction._get_mass_action` becomes `_get_product_propensity`. Its
  subclasses in the systems and the tests call the new name.
- "Mass action" and "mass-action" leave the wording of the core, of the user guide, and of the specs, except as a named
  field example. System modules of chemistry and biology keep the term, since it is their own.
- Every use is renamed: the import and the calls of the core class in the systems, the tests, the gallery, the user
  guide, the generators of the documentation, the README, CONTRIBUTING, the source text that the translators write,
  error messages, the import contract of the systems package, and the constitution profile.
- Test modules mirror the new module names: `test_system.py` becomes `test_composite.py` and `test_mass_action.py`
  becomes `test_reaction_network.py`.
- No alias and no shim module remains, as the project makes breaking changes in 0.x.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `base-interfaces`: the system hierarchy names `CompositeSystem`.
- `package-layout`: the systems package re-exports `CompositeSystem` and `ReactionNetworkSystem`, and the bases and
  concrete roots name `CompositeSystem`.
- `system-modeling`: a system is built with `CompositeSystem`, and the core packages and nested systems name it.
- `general-mechanisms`: reactions follow a product propensity, and the general elements name `ReactionNetworkSystem`.
- `capability-tags`: a `CompositeSystem` derives its tags from its parts.
- `agent-based-modeling`: agent systems use `CompositeSystem` with a discrete time.
- `group-cellular-automata`: a cellular automaton is a `CompositeSystem` with a local rule.
- `learning-dynamics`: adaptive parameters need no subclass of `CompositeSystem`.
- `formalism-plugins`: the logistic growth scenario writes `CompositeSystem`.
- `data-driven-dynamics`: a surrogate result holds a `CompositeSystem`.
- `model-library`: ready-made systems subclass `BaseSystem` or `ReactionNetworkSystem`, and the agent economies are
  `CompositeSystem` instances.
- `user-documentation`: the systems page explains the composite `CompositeSystem`.

## Impact

- Core code: `core/components/systems` (two modules renamed, `__init__.py`, `_base.py`), `interactions/_reaction.py`,
  the `isinstance` checks in `core/analysis`, `engine/runtime/_simulate.py`, `engine/compile/_base.py`, and
  `engine/compile/passes/_flatten.py`, and the translators `engine/translators/_module_writer.py` and
  `_system_dynamics.py`.
- Systems: every module of `skcomplex.systems` that imports `System`, and the subclasses of `MassActionSystem` in
  `chemistry/kinetics/_reactions.py` and `biology/epidemiology/_factories.py`.
- Tests: the mirrored test modules of the systems package, `tests/core/components/interactions/test_reaction.py`, and
  every test that imports `System` or matches its name in an error message.
- Documents: the user guide, the gallery examples, `docs/generate_parts.py`, `docs/generate_systems.py`, the README,
  and CONTRIBUTING.
- Configuration: the layers contract on `skcomplex.core.components.systems` in `pyproject.toml`, and the extension
  points in the profile of `openspec/constitution.md`.
- Ordering: lands after `slim-core-elements-and-spaces`, `slim-core-interactions`, and `slim-core-laws`, alone, with
  no other change running in parallel. The design gives the reason. `move-recipes-to-systems` lands right after it.
- Users: code that imports `System` or `MassActionSystem` imports `CompositeSystem` or `ReactionNetworkSystem`. A
  subclass of `ReactionInteraction` that calls `_get_mass_action` calls `_get_product_propensity`.
