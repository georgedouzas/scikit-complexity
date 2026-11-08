# rewriting-systems Specification

## Purpose
Lets a user build systems that grow by replacing their parts in parallel, words by L-system productions, segments and
polygons by affine substitution, and graphs by graph productions, with one generation per simulation step.

## Requirements

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

### Requirement: Symbol counts are states
The rewriting form SHALL expose the count of every symbol of the alphabet, every prototile, or every node label whose
name is a word of letters, digits, and underscores, and the size of the structure as states, so the trajectory,
observables, and analyses read them by dotted name.

#### Scenario: Parikh vector
- **WHEN** the Fibonacci L-system above has run 10 generations
- **THEN** `trajectory['word.A']` and `trajectory['word.B']` are consecutive Fibonacci numbers at every frame, and
  `trajectory['word.size']` is their sum

### Requirement: String productions rewrite words in parallel
String productions SHALL rewrite every symbol of a word in parallel, and SHALL keep a symbol without a matching
production unchanged. Stochastic productions of one symbol SHALL be chosen per symbol by their probabilities, with
draws from the simulation's random stream.

#### Scenario: Stochastic choice
- **WHEN** a word of 10000 symbols `F` is rewritten once by `F → F+F` with probability 0.3 and `F → F` with 0.7
- **THEN** the share of rewritten symbols lies within 0.3 ± 0.015, and two runs with the same seed give the same word

#### Scenario: Probabilities that do not sum to one
- **WHEN** the stochastic productions of one symbol have probabilities that sum to 0.9
- **THEN** compiling raises a named rewriting error that names the symbol and the sum

### Requirement: Productions can depend on context
A production SHALL accept a left context, a right context, or both, skip the declared ignored symbols, and treat
bracketed branches as Prusinkiewicz and Lindenmayer (1990 §1.8) do. A context-sensitive production SHALL take
precedence over a context-free one for the same symbol.

#### Scenario: Context with ignored symbols
- **WHEN** the word `BAA+A` is rewritten by `A → B` with left context `B`, ignoring `+`, for 3 generations
- **THEN** the words are `BBA+A`, `BBB+A`, and `BBB+B`

### Requirement: Parametric productions rewrite modules
A production SHALL rewrite modules with numeric parameters, guarded by a condition and giving successor parameters,
both written as symbolic expressions of the predecessor's parameters and the system's parameters.

#### Scenario: Parametric growth
- **WHEN** the module `A(1)` is rewritten by `A(x) → F(x) A(x·R)` with condition `x > 0.1` and `R = 0.5`
- **THEN** after 5 generations the word is `F(1) F(0.5) F(0.25) F(0.125) A(0.0625)`, and the module `A(0.0625)` is no
  longer rewritten because its condition is false

### Requirement: Turtles interpret words as geometry
A `GeometrySpec` of `skcomplex.core.specs`, the turtle interpretation of L-systems, SHALL interpret a word as line
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

### Requirement: Substitution replaces segments and polygons by affine copies
A patch SHALL hold tiles, each a prototile with an affine placement, and substitution laws SHALL replace every tile of
a prototile by its pieces, each a prototile with an affine map relative to the parent tile. Prototiles SHALL be
segments or polygons, and tiles whose prototile has no substitution SHALL be kept.

#### Scenario: Cantor intervals
- **WHEN** the unit segment is substituted by the maps `x ↦ x/3` and `x ↦ x/3 + 2/3` for 5 generations
- **THEN** the patch holds 32 intervals, their total length is `(2/3)^5` within 1e-12, and their left endpoints are
  the multiples of `3^-5` whose five ternary digits are all 0 or 2

#### Scenario: Sierpiński triangle area
- **WHEN** the unit triangle is substituted by its three half-scale corner copies for 6 generations
- **THEN** the patch holds 729 triangles, and their area is `(3/4)^6` times the triangle's area within 1e-12

### Requirement: Geometric measures are recorded
The rewriting layer SHALL record the total length and the area of a patch or a turtle geometry at every frame under
`<element>.length` and `<element>.area`, and every structure SHALL give its number of parts, its bounding box, and a
boolean raster at a given resolution, which the box-counting dimension of the analysis layer accepts.

#### Scenario: Box-counting dimension of a gasket
- **WHEN** the box-counting dimension is computed from the raster at resolution 1024 of the Sierpiński triangle at
  generation 8, over boxes of 4 to 128 cells
- **THEN** it is `log 3 / log 2 ≈ 1.585` within 0.05

### Requirement: Growth is bounded
A rewriting system SHALL accept a maximum structure size, and a generation that would exceed it SHALL raise a named
rewriting error before allocating it, naming the element, the generation, and the size it would reach. One generation
of a word of 10⁶ symbols SHALL take under 0.1 s on an idle reference machine.

#### Scenario: Exponential growth stopped
- **WHEN** a Koch L-system with a maximum size of 10⁵ symbols runs for 10 generations
- **THEN** the run raises a rewriting error at the generation whose word would exceed 10⁵ symbols, and the frames
  before it are recorded

### Requirement: Rewriting rules are laws of the rewritten component
`ProductionLaw` and `SubstitutionLaw` SHALL derive from `BaseLaw`, and a word or a patch SHALL return its rewriting
rules from `_get_laws`. No rewriting rule SHALL be an interaction.

#### Scenario: Rule of a word
- **WHEN** a word returns the `ProductionLaw`s of the productions `A -> AB` and `B -> A` from `_get_laws`, and runs for
  five steps
- **THEN** the word is `ABAABABAABAAB`, and `ProductionLaw` is not a subclass of `BaseInteraction`

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
