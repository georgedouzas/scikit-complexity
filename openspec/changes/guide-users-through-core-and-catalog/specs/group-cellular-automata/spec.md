# Spec Delta

## ADDED Requirements

### Requirement: A cellular automaton is a system
A cellular automaton SHALL be a `System` with a `LocalRuleInteraction`, and no class SHALL duplicate it. The `run` of a
`LocalRuleLaw` SHALL be one step of the map on configurations, and several steps SHALL be a loop over that `run` or a
`Simulation` of the system.

#### Scenario: A step and a run agree
- **WHEN** a user applies the `run` of the `LocalRuleLaw` of a shift twice to a configuration, and simulates the system
  of the same rule from that configuration until 2
- **THEN** the last frame of the simulation equals the result of the loop

#### Scenario: No class for an automaton
- **WHEN** a user lists the public names of `skcomplex`
- **THEN** none is `Automaton`, `LinearAutomaton`, or `Subshift`

## MODIFIED Requirements

### Requirement: Automata run on Cayley graphs of groups
`LocalRuleLaw` laws on site groups SHALL run under discrete time on group spaces, the Cayley graphs of finitely presented
groups built on finite quotients or on finite balls, with memory sets given as group elements. A group space SHALL be a
space of the lattice kind, so the solvers and backends that declare lattice support run it.

#### Scenario: Free group majority
- **WHEN** the majority automaton on the free group with two generators runs on a ball of radius 6
- **THEN** every interior cell takes the majority value of itself and its four neighbours given by the generators and
  their inverses

#### Scenario: Orientation is kept
- **WHEN** a rule copies the value at the generator `a` on the Heisenberg group
- **THEN** each cell takes the value of `g a`, not of `a g`

### Requirement: Named groups live in the automata models
The core SHALL name no specific group, and SHALL take finitely presented, automaton, permutation, and matrix groups in
their general form, as native GAP or Sage groups.
Named groups, such as the free, Heisenberg, lamplighter, Baumslag-Solitar, and Grigorchuk groups, SHALL be realizations
in `skcomplex.models.computer_science.automata`.

#### Scenario: Grigorchuk group from the library
- **WHEN** a user imports `grigorchuk_group` from `skcomplex.models.computer_science.automata`
- **THEN** it returns an automaton group built by `libgap.AutomatonGroup`, and no public name of `skcomplex.core`
  names a specific group

#### Scenario: Grigorchuk group in the general form
- **WHEN** a user writes the Grigorchuk group with `libgap.AutomatonGroup` from its wreath recursion
- **THEN** `CayleySpace` takes it, and no field module of `skcomplex.models` is imported

### Requirement: Injectivity and surjectivity are decided on finite universes
`AutomatonAnalysis` SHALL prepare on a `System` with a `LocalRuleInteraction`, and SHALL decide injectivity, surjectivity,
and pre-injectivity of the automaton on finite quotients and
windows, find Garden of Eden patterns, and find the inverse of a reversible automaton.

#### Scenario: Life has a Garden of Eden
- **WHEN** a user checks the Game of Life on a 6 × 6 torus
- **THEN** it is reported not surjective, with a configuration that has no preimage

#### Scenario: Reversible rule
- **WHEN** a user asks for the inverse of the elementary shift automaton
- **THEN** the result is the shift in the opposite direction, and their composition is the identity

### Requirement: Linear automata are decided algebraically
A linear automaton SHALL be a `LocalRuleLaw` whose rule is a `LinearRuleSpec`, a group-ring element whose invertibility and
surjectivity `AutomatonAnalysis` decides algebraically, with its matrix and its Laurent polynomial as the kinds `'matrix'` and
`'polynomial'`.

#### Scenario: Laplacian over a finite field
- **WHEN** a user checks the discrete Laplacian on the cycle of length 5 over the field with 2 elements
- **THEN** the answer matches the invertibility of the corresponding circulant matrix

### Requirement: Subshifts constrain configurations
A subshift of finite type SHALL be a `SubshiftSpec`, the domain of a state. The kinds `'admissible'`, `'violations'`, `'count'`,
and `'sample'` of `AutomatonAnalysis` SHALL check configurations against forbidden patterns, count admissible patterns, and
sample admissible configurations. A run SHALL check that every cell holds a value of the alphabet, and SHALL NOT enforce
admissibility.

#### Scenario: Golden mean shift
- **WHEN** a user counts admissible words of length n in the subshift that forbids `11`
- **THEN** the counts follow the Fibonacci numbers, and the entropy estimate approaches the log of the golden ratio

#### Scenario: A run does not enforce admissibility
- **WHEN** a system whose state has the domain of the subshift that forbids `11` runs from a configuration that holds `11`
- **THEN** the run completes, and the kind `'admissible'` reports that configuration as not admissible
