## MODIFIED Requirements

### Requirement: Rewriting systems are a form
A system whose elements hold a word or a patch of tiles rewritten by productions SHALL compile to the rewriting form
with the rewriting compiler of `skcomplex.models.mathematics.fractals`. Each simulation step SHALL apply one
generation, every part of the structure rewritten in parallel, and the time between frames SHALL be a positive integer
number of generations.

#### Scenario: Generations as steps
- **WHEN** an L-system with the axiom `A` and the productions `A → AB` and `B → A` runs with `until=5`
- **THEN** it compiles to the rewriting form, the trajectory has 6 frames at times 0 to 5, and the word lengths are
  1, 2, 3, 5, 8, and 13

#### Scenario: Fractional generation
- **WHEN** a rewriting system runs with `dt=0.5`
- **THEN** the run raises a validation error that names `dt` and asks for a positive integer

## REMOVED Requirements

### Requirement: Graph productions rewrite host graphs
**Reason**: No model of the library uses graph rewriting, so it does not belong in the core.
**Migration**: None. Define a host graph element, its productions, and their kernel in a model package, through a compiler of its own.
