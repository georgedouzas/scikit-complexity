## MODIFIED Requirements

### Requirement: Rewriting systems are a form
A system whose elements hold a word or a patch of tiles rewritten by productions SHALL compile to the rewriting form
with the private compiler of that form, which `simulate` chooses automatically. Each simulation step SHALL apply one
generation, every part of the structure rewritten in parallel, and the time between frames SHALL be a positive integer
number of generations.

#### Scenario: Generations as steps
- **WHEN** an L-system with the axiom `A` and the productions `A → AB` and `B → A` runs with `until=5`
- **THEN** it compiles to the rewriting form, the trajectory has 6 frames at times 0 to 5, and the word lengths are
  1, 2, 3, 5, 8, and 13

#### Scenario: Fractional generation
- **WHEN** a rewriting system runs with `dt=0.5`
- **THEN** the run raises a validation error that names `dt` and asks for a positive integer

### Requirement: Rewriting parts live in the core
`WordElement` and `PatchElement` SHALL live in `skcomplex.core.components.elements`, and `ProductionLaw` and `SubstitutionLaw` in
`skcomplex.core.components.laws`. Their specs SHALL live in `skcomplex.core.specs`, and `RewritingError` in `skcomplex.core.errors`.
Field-specific elements, such as `KochElement`, and their system classes SHALL stay in `skcomplex.systems.mathematics.fractals`.

#### Scenario: A rewriting system from the core alone
- **WHEN** a user writes an L-system from a `WordElement` and a `ProductionLaw`, and runs it with `simulate` and no compiler
- **THEN** it runs without importing a field module of `skcomplex.systems`

### Requirement: Rewriting runs in the engine
The compiler and the problem of the rewriting form SHALL be private modules of `skcomplex.core.engine.compile` and
`skcomplex.core.engine.problems`, and `RewritingSolver` SHALL live in `skcomplex.core.engine.solvers` and
`GeometryVisual` in `skcomplex.core.engine.viz`.

#### Scenario: The core imports no engine
- **WHEN** the import contracts run
- **THEN** no module of `skcomplex.core.components` imports `skcomplex.core.engine`, although a spec of a word names its visual
