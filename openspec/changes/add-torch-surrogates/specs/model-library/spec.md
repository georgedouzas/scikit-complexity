## ADDED Requirements

### Requirement: Gradients of equivariant network potentials
`EquivariantPotentialInteraction` of `skcomplex.systems.general.mechanics` SHALL move the positions of every member of a
group down the gradient of a many-body network potential of all members, invariant to rotations, translations, and permutations of members of one
kind, given by a saved or published network and an integer kind per member, divided by an optional divisor.

#### Scenario: Gradient of the network
- **WHEN** the interaction acts on a group of 8 members with a small untrained network
- **THEN** the rate of every member equals minus the finite-difference gradient of the network potential, divided by
  its divisor, within 1e-5 relative error

#### Scenario: Rotation equivariance
- **WHEN** every state of the group is rotated by one rotation in three dimensions
- **THEN** the rates rotate by the same rotation within 1e-6 relative error

#### Scenario: Wrong number of kinds
- **WHEN** the kinds differ in length from the size of the group
- **THEN** preparing the system raises `ValidationError`, whose message names both lengths

### Requirement: Network potentials respect periodic spaces
On a periodic continuous space, the network potential of `EquivariantPotentialInteraction` SHALL read the members by
the minimum image within the cutoff of the network, and its potential SHALL be recorded as an observable of the run.

#### Scenario: Members across the boundary
- **WHEN** two members of a periodic box lie on opposite faces, closer by the minimum image than the cutoff
- **THEN** their rates equal those of the same pair placed side by side in the middle of the box

#### Scenario: Recorded potential
- **WHEN** a group with the interaction is prepared with `VelocityVerletSolver` and no thermostat and simulated for
  1000 steps
- **THEN** the trajectory holds the potential at every frame, and the total of the potential and the kinetic term drifts
  by less than 1e-3 of its initial value

### Requirement: Water with a machine-learned potential
`MaceWaterSystem` of `physics/statistical_mechanics` SHALL be water molecules in a periodic cubic box at a given
density and temperature, with a Langevin thermostat, moved by a pretrained equivariant network potential of the atoms
after Batatia et al. (2023, arXiv 2401.00096), in ångström, electronvolt, and femtosecond units.

#### Scenario: Structure of the system
- **WHEN** the system is built with 8 molecules and a saved small network
- **THEN** it holds 24 atoms with the kinds of oxygen and hydrogen, in a box whose side gives the requested density

#### Scenario: Liquid structure
- **WHEN** 32 molecules run for 2 picoseconds at 300 K with the default pretrained model
- **THEN** the first peak of the oxygen-oxygen radial distribution lies between 2.6 and 3.0 ångström

### Requirement: Coarse-grained protein folding
`CgProteinFoldingSystem` of `physics/soft_matter` SHALL be the chignolin chain at two resolutions after Charron et al.
(2023, arXiv 2310.18278): a fine native-centric model with two beads per residue, and a coarse model of one bead per
residue with bond, angle, and repulsion priors plus an optional fitted network potential.

#### Scenario: Fine model folds
- **WHEN** the fine model runs from an extended chain at its folding temperature minus 20%
- **THEN** the mean fraction of native contacts over the second half of the run exceeds 0.8

#### Scenario: Coarse model from force matching
- **WHEN** the coarse model takes the network fitted by `force_matching` with `kind='equivariant'` on mapped frames of
  the fine model, and both run at the folding temperature
- **THEN** the folded fraction of the coarse run lies within 0.15 of that of the fine run

#### Scenario: Chain geometry
- **WHEN** `chain_shape` reads a run of either resolution
- **THEN** it returns the radius of gyration of the residue beads at every frame

### Requirement: Torch surrogate probes are recorded in the system inventory
The inventory SHALL record `mace_water_md` and `cg_protein_folding` as implemented with their sources and primitives,
and the gallery SHALL hold their examples.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for the two systems
