# Proposal

## Why

An element declares its states as annotated constructor arguments, so a model whose states depend on its data has no way to
write its class. Five places work around that by building a class at runtime with `type()` and a hand-set `__signature__`:
`_surrogate_system` of `SparseIdentificationAnalysis`, `_element_type` of `next_generation_reservoir`, and three in the
metabolism family (`_culture`, `_metabolic_network`, `_community`). That is the machinery the project's design rules
reject, and each copy repeats how `BasePart` reads its specs. The core lacks a general primitive, and these models reveal it.

## What Changes

- Add a way to declare an element's states and parameters from data, without generating a class or a signature.
- Rewrite the five places on it, keeping their results and public names.
- Add a test that no module of the library sets `__signature__` or builds a part with `type()`.

## Capabilities

### New Capabilities

- `declared-state-elements`: elements whose states and parameters are given as data at construction.

### Modified Capabilities

None. The models that move keep their behaviour.

## Impact

- `skcomplex.core.base` and `skcomplex.core.elements`, and the compiler passes that read specs through `_get_specs`.
- `skcomplex.analysis._sparse_identification`, `models.computer_science.neural_networks._next_generation`, and
  `models.biology.common.metabolism`.
- No new package and no new dependency.
