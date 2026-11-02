# constraint-satisfaction Specification

## Purpose
Lets a user reveal clauses and cardinality constraints on the Boolean states of a group step by step and decide their
satisfiability incrementally.

## Requirements

### Requirement: Constraints are declared on the members of a group
`SatisfiabilityInteraction` SHALL bind a group whose state is Boolean and SHALL take a sequence of constraints, each a
`ClauseSpec` of literals or a `CardinalitySpec` that at least or at most a bound of its literals hold, where a literal
is a member index from 1, negative for its negation. A state that is not Boolean SHALL be reported with the code
`law-domain`.

#### Scenario: Perceptron pattern as a constraint
- **WHEN** a binary perceptron of 21 weights `w_i = ±1` must give the output `sigma` for a pattern `xi`
- **THEN** the pattern becomes the `CardinalitySpec` that at least 11 of the literals `w_i = sigma xi_i` hold

#### Scenario: Real state
- **WHEN** a `SatisfiabilityInteraction` binds a group whose state is real
- **THEN** the report of `PassCompiler` holds an issue with the code `law-domain` that names the interaction and the
  state

### Requirement: Satisfiability is decided incrementally
`SatisfiabilityInteraction` SHALL reveal a set number of further constraints per step of discrete time, decide whether
the revealed constraints are satisfiable with one solver that keeps what it learned, set the states of the members to a
satisfying assignment when one exists, and report `satisfiable` and `revealed` every step.

#### Scenario: Small perceptron
- **WHEN** the perceptron of Bertin (2021 Ex. 6.4) with `N = 3` reveals its patterns one per step
- **THEN** the reported `satisfiable` at every step equals the result of enumerating all `2^3` weight vectors, and every
  assignment it sets satisfies the revealed patterns

#### Scenario: Capacity of the binary perceptron
- **WHEN** random patterns of 21 components are revealed one per step until the constraints become unsatisfiable, over
  20 replicates
- **THEN** the mean load `p/N` at the first unsatisfiable step lies between 0.7 and 1.0, near the critical load 0.83
