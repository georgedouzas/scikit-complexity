# molecular-structures Specification

## Purpose

Lets a user sample sequences and chains held as lattice rows by Monte Carlo with a score of a whole row, replay an
ensemble one member per step, and measure the geometry of point chains. The biological pieces live in
`skcomplex.systems.biology`.

## Requirements

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

### Requirement: Sampling accepts an infinite intensity
The intensity of `SamplingInteraction` SHALL take an infinite value, the zero temperature of its thin subclass in
physics, at which the Metropolis rule accepts a move only when the objective does not grow.

#### Scenario: Zero temperature on a local energy
- **WHEN** an Ising energy on a lattice is sampled at temperature zero
- **THEN** the total energy never increases from one frame to the next

### Requirement: Drives read the step they run in
`ArrayUpdateLaw` SHALL pass the time of the step to a function that names `time` in its signature, so an array update
can set a state to member `k` of an ensemble at step `k`.

#### Scenario: Replay of an ensemble
- **WHEN** an array update sets the positions of a 20-site chain to member `min(time, 37)` of an ensemble of 38
  members, and the system runs to time 40
- **THEN** the positions at frame 10 are member 10, and the positions at frame 40 are member 37

### Requirement: Chain geometry is analysed
`chain_shape` SHALL give, per frame of a group's positions, the radius of gyration, the end-to-end distance, and the
root mean square deviation from a reference after optimal superposition.

#### Scenario: Freely rotating chain
- **WHEN** `chain_shape(positions, kind='end-to-end')` reads an ensemble of 2000 chains of 200 points with bond length 1, bond
  angle 112°, and uniform dihedral angles
- **THEN** the mean squared end-to-end distance matches `n b² (1 + cos θ) / (1 − cos θ)`, with `θ` the supplement of the
  bond angle, within 5%

#### Scenario: Superposition
- **WHEN** `chain_shape(positions, kind='rmsd')` compares a conformation rotated and translated by a random rigid motion with the
  original
- **THEN** the deviation is below 1e-6

### Requirement: Missing extras raise a named error
`skcomplex.core.errors` SHALL define `MissingExtraError`, which a system raises when an optional extra it needs is not
installed, naming the system, the extra, and the install command. The default install SHALL NOT import Biopython,
ViennaRNA, or MDTraj.

#### Scenario: Default install stays importable
- **WHEN** the package and its system catalog are imported without the extras
- **THEN** no module of Biopython, ViennaRNA, or MDTraj is imported, and every system is listed in the catalog

### Requirement: The core stays free of molecular biology
`skcomplex.core.components`, `skcomplex.core.engine`, and `skcomplex.core.analysis` SHALL export no structure, RNA, hydrophobicity, genotype,
or protein name, and SHALL import neither Biopython, ViennaRNA, nor MDTraj.

#### Scenario: Field-term and import contracts
- **WHEN** the field-term test and the import contracts run after this change
- **THEN** both pass, and the biological pieces import from `skcomplex.systems.biology`
