# Spec Delta

## ADDED Requirements

### Requirement: The engine names no realization of a model package
The engine SHALL NOT list a compiler, a form, or a law kind other than those of its own packages. A form, its laws,
its compiler, and its solver that serve the models of one field SHALL live in that field of the models library.

#### Scenario: No realization in the engine
- **WHEN** a test searches the public names of `skcomplex.core`, `skcomplex.engine`, `skcomplex.specs`, and `skcomplex.types`
  for `KochElement`, `CantorElement`, `SierpinskiElement`, and `grigorchuk_group`
- **THEN** it finds none

#### Scenario: Rewriting in the core and the engine
- **WHEN** a user imports `WordElement` and `PatchElement` from `skcomplex.core.elements`, `ProductionLaw` and
  `SubstitutionLaw` from `skcomplex.core.laws`, and `RewritingCompiler` from `skcomplex.engine.compile`
- **THEN** every import succeeds, and none of these modules imports `skcomplex.models`

## REMOVED Requirements

### Requirement: The engine names no form, compiler, or law of a model package
**Reason**: Its only scenario states that rewriting is absent from the core and the engine, and rewriting now lives there.
**Migration**: The requirement "The engine names no realization of a model package" keeps its text, and its scenarios name
the realizations that stay in the models.
