# Design

## Context

`BasePart._get_specs` is a classmethod that reads `inspect.signature(__init__)` over the MRO and caches per class, and
scikit-learn's `get_params` and `clone` read the same signature. So a spec is a fact of a class, and the five workarounds
make one class per set of states so that the fact holds.

## Goals / Non-Goals

**Goals:** one general primitive, no `type()` or `__signature__` in the library, unchanged results, and working `clone`.

**Non-Goals:** changing how ordinary elements declare states, or changing any model's public name.

## Decisions

### States as one data argument

An element takes one constructor argument holding its declared variables by name, each a value with its spec. This keeps a
trivial `__init__`, and `get_params` and `clone` work because the argument is an ordinary one. Fields are read by the laws
through the same namespace as annotated states.

The alternative, a factory that returns a class, is what the code does now, tidied. It keeps one class per dataset and
leaves the machinery in the core, so it is rejected.

### Specs of an instance

`_get_specs` is a classmethod read while compiling. The first task finds every caller and decides between an instance method
that falls back to the class specs and a new reader beside it. The rule is that annotated elements give the same specs as
today. If this needs a change across the compiler that the tasks do not name, the change pauses for a decision.

### Attributes

The declared values are also read as `x.name` by laws and views, so an attribute is set per name. The rule against
generated attributes holds for class machinery; if this conflicts with `__getattr__` being banned as a magic method, the
values go in a namespace on the element, and the compiler's symbol names stay the same.

## Risks / Trade-offs

- [Compiled systems depend on specs per class through a cache] → Reference results for all five places are recorded before
  any move, and the conformance suite covers the models.
- [A subclass of `MetabolicNetworkElement` carries `reactions` as class data] → It becomes constructor data, and
  `metabolic_network_type` stays as a function if a public name depends on it.

## Migration Plan

Add the primitive with its tests first, then move the five places one commit each, so any one can be reverted.
