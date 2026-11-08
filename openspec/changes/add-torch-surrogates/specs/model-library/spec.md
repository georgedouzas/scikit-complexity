## ADDED Requirements

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
