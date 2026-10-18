# Proposal

## Why

The core already describes models from several sciences through elements, interactions, laws, and systems, but it does
not yet state that interface as the one way every model is built. Defining a component always needs a subclass, state
domains stop at real and integer numbers, macro variables exist only as co-simulation aggregates, and the library mixes
reusable component classes with a few factory functions that nothing lists or checks. The references describe any
complex system in the same field-agnostic terms: components with typed states, typed interactions, update laws, and
observables at several levels (Thurner et al. 2018 §1.5, Mobus 2022 §4.3.3, Dilão 2023 ch. 1, Datseris and Parlitz 2022
ch. 1). The core should express exactly that, and every ready-made model should be a factory of it.

## What Changes

- Define component types without subclassing, from declared states, parameters, roles, and a laws function.
- Extend state domains with booleans, finite alphabets, and angles, so discrete and periodic models need no encoding.
- Add observables: named symbolic expressions of states or reductions over groups, declared on a system, recorded in
  trajectories, logged to Rerun, and readable by laws of other partitions as aggregates are today.
- Add a model catalog: factories register with their field, description, tags, references, and default run settings,
  and the catalog lists, describes, and creates models by name, with the formalism inferred by compiling.
- Rebuild the library as registered factories over reusable components, add factories for the models that exist today
  only as components, and check that the library uses only the public interface.
- Run every registered model through one conformance suite: create with defaults, compile, simulate, log to Rerun
  without a viewer, and clone.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `system-modeling`: components defined without subclassing, new state domains, and observables.
- `model-library`: the catalog, factories for every ready-made model, and the conformance suite.
- `simulation`: trajectories record observables.
- `rerun-visualization`: observables are logged.

## Impact

- `skcomplex.core` gains the new domains, `skcomplex.model` gains `Element.define`, `Interaction.define`, and
  `Observable`, and `skcomplex.models` gains the catalog.
- Existing component classes and factories keep working, so the change is additive.
- Every later change in this series adds its models as registered factories and is covered by the conformance suite.
