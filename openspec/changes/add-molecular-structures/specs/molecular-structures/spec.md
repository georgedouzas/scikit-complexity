# Spec Delta

## Purpose

Lets a user model sequences, chains, and molecular structures as lattice states. A user declares sequences over an
alphabet and chains with internal coordinates, samples them by Monte Carlo with scores of a whole chain or sequence,
maps sequences to structures, reads and replays structure files and ensembles, and analyses neutral networks and chain
geometry. The general pieces live in the core, and the biological ones in `skcomplex.models.biology`.

## ADDED Requirements

### Requirement: Sequences are categorical states along a lattice axis
The modelling layer SHALL declare a sequence as a categorical state over a declared alphabet in a site group of a
lattice, along its last axis, so that a lattice of shape `(copies, length)` holds one sequence per row. The sampling
law SHALL offer a point mutation move that replaces the value at one site by a different value of the alphabet, drawn
uniformly. Sequences SHALL convert to and from strings of the alphabet's letters.

#### Scenario: Point mutation
- **WHEN** a point mutation is proposed on an RNA sequence of length 30
- **THEN** the proposal differs from the sequence at exactly one site, and its letter there is one of the three other
  letters of `ACGU`, each with frequency 1/3 within 0.01 over 30000 proposals

#### Scenario: Letter outside the alphabet
- **WHEN** a sequence over `ACGU` is initialised from the string `ACGT`
- **THEN** construction raises a named validation error that names the letter `T`, its position 3, and the alphabet

### Requirement: Chains derive positions from internal coordinates
The core SHALL declare a chain as a site group on a one-dimensional lattice whose sites hold bond lengths, bond angles,
and dihedral angles, and SHALL derive the Cartesian positions of its sites from them with a bead geometry or with any
geometry object given to it. A rotation move SHALL change one dihedral angle by a uniform amount in `[-width, width]`.

#### Scenario: Freely rotating chain
- **WHEN** an ensemble of 2000 chains of 200 sites with bond length 1 and bond angle 112° is sampled with uniform
  dihedral angles
- **THEN** the mean squared end-to-end distance matches the freely rotating chain value
  `n b² (1 + cos θ) / (1 − cos θ)`, with `θ` the supplement of the bond angle, within 5%

### Requirement: The library provides a protein backbone geometry
The library SHALL provide, in `skcomplex.models.biology.realizations`, a protein backbone geometry for chains that
takes `phi`, `psi`, and `omega` per residue and derives the backbone atoms with standard geometry.

#### Scenario: Round trip through internal coordinates
- **WHEN** the internal coordinates of a read protein structure are computed and its positions are derived from them
  with the library's backbone geometry
- **THEN** the derived backbone positions match the read ones with an RMSD below 0.01 Å after superposition

### Requirement: Chains are validated against their space
A chain SHALL declare that it requires discrete time and a lattice space, and `validate()` SHALL report a chain outside
a one-dimensional lattice with its path and the fix of a one-dimensional `LatticeSpace`.

#### Scenario: Chain on a square lattice
- **WHEN** a chain is placed on a two-dimensional lattice
- **THEN** validation reports one issue that names the chain and asks for a one-dimensional lattice

### Requirement: Monte Carlo moves are accepted by global scores
The sampling law SHALL take a score, a function of the whole sequence or chain of one lattice row, in place of a local
energy, and SHALL accept moves by the change of that score with Metropolis acceptance at a temperature that changes
live. At temperature zero, a move SHALL be accepted only when it does not raise the score. Each row SHALL be scored
separately, each unit of time SHALL propose a declared number of moves per row, and the score of every row SHALL be
recorded in the trajectory.

#### Scenario: Neutral walk
- **WHEN** a score is 0 for sequences whose phenotype equals a target and 1 otherwise, and a walk starts on the target
  at temperature zero
- **THEN** every recorded sequence has the target phenotype, and the sequence changes at every accepted move

#### Scenario: Domain-agnostic score
- **WHEN** a binary sequence of length 20 is scored by its number of ones at temperature zero, with one proposal per
  unit of time
- **THEN** the score never increases, and it reaches 0 before time 2000 with probability above 0.99 over 100 seeds

#### Scenario: Score at a positive temperature
- **WHEN** the same binary sequence is sampled at temperature 1 for 20000 units of time
- **THEN** the mean number of ones matches `20 / (1 + e)` within 3%

### Requirement: Global scores are validated
`validate()` SHALL report a sampling law that gives both a local energy and a score, or a local energy and a score that
target one lattice state, and a score of an uncompiled Python function with the Numba backend, each with its path and a
fix, before compilation.

#### Scenario: Local energy and score on one state
- **WHEN** one lattice state is targeted by a sampling law with a local energy and another with a score
- **THEN** validation reports one issue that names both laws and suggests keeping one of them

#### Scenario: Python score on Numba
- **WHEN** a score wraps an uncompiled Python function and the system is validated with the Numba backend
- **THEN** validation reports that the backend does not support the score and suggests the NumPy backend or a
  Numba-compiled function

### Requirement: Ensembles are replayed one member per step
The core SHALL provide a replay law of the discrete class that sets a state to member `k` of an ensemble at step `k`
and holds the last member after the end. It SHALL require discrete time, and validation SHALL report it under any other
time.

#### Scenario: Replay under continuous time
- **WHEN** a replay law is placed in a system with `ContinuousTime()`
- **THEN** validation reports the law and suggests `DiscreteTime()`

### Requirement: Sequences map to structures
The library SHALL provide, in the `biology/evolution` model package, maps from a sequence to a phenotype key and named
numeric features, memoised so that a repeated sequence is not computed again. A user SHALL define a map from any
function of a sequence. The RNA folding map SHALL return the minimum free energy secondary structure in dot-bracket
notation, its free energy in kcal/mol, and the base-pair distance to a given structure, and SHALL lay out a structure in
the plane.

#### Scenario: Hairpin fold
- **WHEN** the sequence `GGGGAAACCCC` is folded at 37 °C with the default energy parameters
- **THEN** the structure is `((((...))))` and its free energy is negative

#### Scenario: Memoised map
- **WHEN** a walk evaluates the same sequence twice
- **THEN** the map computes it once, and the cache statistics report one hit

#### Scenario: Base-pair distance
- **WHEN** the base-pair distance between `((((...))))` and `.(((...))).` is computed
- **THEN** it is 1

### Requirement: Neutral networks are analysed
The library SHALL compute, in the `biology/evolution` model package, for a map over sequences of an alphabet:
- the neutral network of a phenotype, enumerated or sampled by neutral walks, as a graph of point mutations;
- genotype robustness, the number of neutral neighbours of a sequence;
- genotype evolvability, the number of other phenotypes among its neighbours;
- phenotype robustness and evolvability over the neutral network, as Wagner defines them.

#### Scenario: Toy map of the first letter
- **WHEN** the binary sequences of length 4 are mapped to their first letter and enumerated
- **THEN** each phenotype's neutral network has 8 sequences in one component, every genotype robustness is 3, and every
  genotype evolvability is 1

#### Scenario: Toy map of the number of ones
- **WHEN** the binary sequences of length 4 are mapped to their number of ones
- **THEN** the neutral network of phenotype 2 has 6 sequences and no edges, its phenotype robustness is 0, and its
  phenotype evolvability is 2

#### Scenario: Neighbours of an RNA sequence
- **WHEN** the genotype robustness and evolvability of an RNA sequence of length 30 are computed
- **THEN** the robustness and the number of non-neutral neighbours sum to 90

### Requirement: Structures are read, written, and replayed
The library SHALL read and write, in `skcomplex.models.biology.realizations`, PDB and mmCIF files as structures that
hold every model of an ensemble, with chains, residues, atoms, coordinates, and effective atoms. The core's replay law
SHALL set a chain's positions to member `k` of a structure's ensemble at frame `k`. With the extra `trajectories`, the
library SHALL also read molecular dynamics trajectories, such as DCD and XTC, with a topology file.

#### Scenario: NMR ensemble
- **WHEN** a user reads the bundled Trp-cage NMR structure 1L2Y
- **THEN** it holds 38 models of 20 residues and 304 atoms each

#### Scenario: Replay of an ensemble
- **WHEN** a chain replays the 38 models of 1L2Y with one frame per unit of time from time 0 to 37
- **THEN** the trajectory has 38 frames, and the positions at frame 10 are the effective atoms of model 11

#### Scenario: Round trip
- **WHEN** a structure is written to mmCIF and to PDB and read back
- **THEN** both copies have the same residues, atom names, and coordinates within 0.001 Å

### Requirement: Chain geometry is analysed
The analysis layer SHALL compute the radius of gyration, the end-to-end distance, and the RMSD of two conformations
after optimal superposition, per frame of a trajectory. The library SHALL compute, in
`skcomplex.models.biology.realizations`, the dihedral angles of a protein backbone and the angle between successive
peptide bond planes.

#### Scenario: Ideal helix
- **WHEN** a polyalanine chain of 20 residues is built with `phi = −57°` and `psi = −47°`
- **THEN** its computed dihedral angles return those values within 0.01°, and its rise per residue along the helix axis
  is 1.5 Å within 0.1 Å

#### Scenario: Superposition
- **WHEN** a conformation is rotated and translated by a random rigid motion
- **THEN** its RMSD to the original after superposition is below 1e-6 Å

### Requirement: Hydrophobicity is compared with a fuzzy oil drop
The library SHALL compute, in `skcomplex.models.biology.realizations`, for a structure's effective atoms and a
hydrophobicity scale, the Gaussian distribution `T` along the principal axes, the observed distribution `O` from
Levitt's function with a 9 Å cut-off, the uniform distribution `R`, `D_KL(O|T)` and `D_KL(O|R)` in bits, the relative
distance `RD`, and the environment factor `K` that best fits `O`. A residue that the scale does not cover SHALL raise a
named error.

#### Scenario: Ideal micelle
- **WHEN** the observed distribution of a structure equals its theoretical distribution
- **THEN** `D_KL(O|T)` is 0 within 1e-12, and `RD` is 0 within 1e-12

#### Scenario: Uniform hydrophobicity
- **WHEN** the observed distribution is uniform
- **THEN** `RD` is 1 within 1e-12

#### Scenario: Native against extended
- **WHEN** `RD` is computed for the bundled native ubiquitin structure and for its chain rebuilt with all `phi = −120°`
  and `psi = 130°`
- **THEN** the native `RD` is lower than the extended `RD`

#### Scenario: Unknown residue
- **WHEN** a structure holds a residue `SEP` that the scale does not cover
- **THEN** the computation raises a named error that names `SEP`, its chain and number, and the scale

### Requirement: Optional extras are named
The default install SHALL NOT install or import ViennaRNA or MDTraj. RNA folding SHALL need the extra `rna`, and reading
molecular dynamics trajectory formats SHALL need the extra `trajectories`. Every capability that needs a missing extra
SHALL raise a named error that names the capability and the extra, and gives the install command.

#### Scenario: Missing RNA extra
- **WHEN** a user without the extra `rna` folds a sequence
- **THEN** a named error says that RNA folding needs `pip install 'scikit-complexity[rna]'`

#### Scenario: Default install stays importable
- **WHEN** the package and its model catalog are imported without the extras
- **THEN** no module of ViennaRNA or MDTraj is imported, and every model is listed in the catalog

### Requirement: The core stays free of molecular biology
`skcomplex.core` SHALL export no structure, RNA, hydrophobicity, genotype, or protein name, and SHALL import neither
Biopython, ViennaRNA, MDTraj, nor `skcomplex.models`.

#### Scenario: Field-term and import-graph tests
- **WHEN** the field-term and import-graph tests of the core run after this change
- **THEN** both pass, and the biological pieces of this change import from `skcomplex.models.biology`
