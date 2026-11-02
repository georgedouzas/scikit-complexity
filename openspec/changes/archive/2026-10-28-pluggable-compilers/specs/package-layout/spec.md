## ADDED Requirements

### Requirement: The engine names no form, compiler, or law of a model package
The engine SHALL NOT list a compiler, a form, or a law kind other than those of its own packages. A form, its laws,
its compiler, and its solver that serve the models of one field SHALL live in that field of the models library.

#### Scenario: No rewriting in the engine
- **WHEN** a test searches `skcomplex.core`, `skcomplex.engine`, `skcomplex.specs`, and `skcomplex.types` for
  `WordElement`, `PatchElement`, `ProductionLaw`, `SubstitutionLaw`, `RewritingCompiler`, and the rewriting form
- **THEN** it finds none
