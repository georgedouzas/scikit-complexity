# Spec Delta

## ADDED Requirements

### Requirement: Rewriting parts live in the core
`WordElement` and `PatchElement` SHALL live in `skcomplex.core.elements`, and `ProductionLaw` and `SubstitutionLaw` in
`skcomplex.core.laws`. Their specs SHALL live in `skcomplex.specs`, and `RewritingError` in `skcomplex.errors`.
Field-specific elements, such as `KochElement`, and their factories SHALL stay in `skcomplex.models.mathematics.fractals`.

#### Scenario: A rewriting system from the core alone
- **WHEN** a user writes an L-system from a `WordElement` and a `ProductionLaw`, and runs it with `RewritingCompiler`
- **THEN** it runs without importing a field module of `skcomplex.models`

### Requirement: Rewriting runs in the engine
`RewritingCompiler` SHALL live in `skcomplex.engine.compile`, `RewritingSolver` in `skcomplex.engine.solvers`,
`RewritingProblemSpec` in `skcomplex.engine.problems`, and `GeometryVisual` in `skcomplex.engine.viz`.

#### Scenario: The core imports no engine
- **WHEN** the import contracts run
- **THEN** no module of `skcomplex.core` imports `skcomplex.engine`, although a spec of a word names its visual

## MODIFIED Requirements

### Requirement: Rewriting systems are a form
A system whose elements hold a word or a patch of tiles rewritten by productions SHALL compile to the rewriting form
with `RewritingCompiler` of `skcomplex.engine.compile`. Each simulation step SHALL apply one
generation, every part of the structure rewritten in parallel, and the time between frames SHALL be a positive integer
number of generations.

#### Scenario: Generations as steps
- **WHEN** an L-system with the axiom `A` and the productions `A → AB` and `B → A` runs with `until=5`
- **THEN** it compiles to the rewriting form, the trajectory has 6 frames at times 0 to 5, and the word lengths are
  1, 2, 3, 5, 8, and 13

#### Scenario: Fractional generation
- **WHEN** a rewriting system runs with `dt=0.5`
- **THEN** the run raises a validation error that names `dt` and asks for a positive integer

### Requirement: Turtles interpret words as geometry
A `GeometrySpec` of `skcomplex.specs`, the turtle interpretation of L-systems, SHALL interpret a word as line
segments in two or three dimensions from declared symbol meanings: moving with or without drawing, turning by a
parameter angle, pushing and popping its state with brackets, and scaling its step per generation. A module's first
parameter SHALL override the step or the angle. A word SHALL take it as its `geometry`.

#### Scenario: Koch curve length
- **WHEN** the word of the Koch L-system `F → F+F--F+F` with a 60° angle and a step of `3^-n` is interpreted at
  generation `n = 4`
- **THEN** the geometry has 256 segments, its total length is `(4/3)^4` within 1e-12, and it starts at `(0, 0)` and
  ends at `(1, 0)`

#### Scenario: Branches return
- **WHEN** the word `F[+F]F` is interpreted with a 90° angle and a unit step
- **THEN** the geometry has 3 segments, and the third starts where the second segment's branch began

### Requirement: Rewriting rules are laws of the rewritten component
`ProductionLaw` and `SubstitutionLaw` SHALL derive from `BaseLaw`, and a word or a patch SHALL return its rewriting
rules from `_get_laws`. No rewriting rule SHALL be an interaction.

#### Scenario: Rule of a word
- **WHEN** a word returns the `ProductionLaw`s of the productions `A -> AB` and `B -> A` from `_get_laws`, and runs for
  five steps
- **THEN** the word is `ABAABABAABAAB`, and `ProductionLaw` is not a subclass of `BaseInteraction`
