## MODIFIED Requirements

### Requirement: Monte Carlo moves are accepted by row scores
The biology `SequenceSamplingInteraction`, a subclass of the core `SamplingInteraction` in
`skcomplex.systems.biology.common.sequences`, SHALL take a `RowScoreSpec` as its objective, a function of the values of
one row of a lattice state along its last axis and the target's parameters that returns one number. It SHALL propose a
declared number of moves per row and unit of time through its move, accept each by its rule on the change of the score
of the row, give every row its own draws from the simulation's stream, and record the score of every row in the
trajectory. It SHALL return an `ArrayUpdateLaw`.

#### Scenario: Neutral walk
- **WHEN** a row score is 0 for sequences whose phenotype equals a target and 1 otherwise, and a walk starts on the
  target at temperature zero
- **THEN** every recorded sequence has the target phenotype

#### Scenario: Domain-agnostic score
- **WHEN** a binary row of length 20 is scored by its number of ones at temperature zero, with one proposal per unit of
  time and a move that flips one random site
- **THEN** the score never increases, and it reaches 0 before time 2000 in more than 99 of 100 seeds

#### Scenario: Score at a positive temperature
- **WHEN** the same binary row is sampled at temperature 1 for 20000 units of time
- **THEN** the mean number of ones matches `20 / (1 + e)` within 3%

#### Scenario: Point mutation as a function move
- **WHEN** a move function replaces the value at one random site of an RNA row of length 30 by one of the other three
  letters of `ACGU`
- **THEN** each proposal differs from the row at exactly one site, and each other letter is drawn with frequency 1/3
  within 0.01 over 30000 proposals
