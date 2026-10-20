# Spec Delta

## Purpose

Lets a user model systems that grow by replacing their parts in parallel, words by L-system productions, segments and
polygons by affine substitution, and graphs by graph productions, with one generation per simulation step.

## ADDED Requirements

### Requirement: Rewriting systems are a form
A system whose elements hold a word, a patch of tiles, or a host graph rewritten by productions SHALL compile to the
rewriting form. Each simulation step SHALL apply one generation, every part of the structure rewritten in parallel,
and the time between frames SHALL be a positive integer number of generations.

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
A turtle SHALL interpret a word as line segments in two or three dimensions from declared symbol meanings, moving
forward with or without drawing, turning by a parameter angle, pushing and popping its state with brackets, and
scaling its step by a factor per generation. A module's first parameter SHALL override the step or the angle where
the module has one.

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

### Requirement: Graph productions rewrite host graphs
A host graph SHALL be rewritten by double-pushout graph productions, each a left graph, an interface, and a right
graph with optional node and edge labels. Each generation SHALL apply a production at a maximal set of parallel
independent matches, which overlap only in nodes and edges that every one of them preserves, chosen in a reproducible
order from the simulation's random stream. It SHALL skip a match whose deletion would leave a dangling edge.

#### Scenario: Edge subdivision
- **WHEN** a cycle of 4 nodes is rewritten by a production that replaces each edge by a path of two edges for 3
  generations
- **THEN** the host graph is a cycle of 32 nodes, and the state `graph.size` is 32 at the last frame

#### Scenario: Dangling edge
- **WHEN** a production deletes a node of its left graph that has an edge outside the match
- **THEN** that match is not applied, and the other matches are

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
