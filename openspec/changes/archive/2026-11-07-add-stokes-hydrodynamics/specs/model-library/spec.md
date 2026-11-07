## ADDED Requirements

### Requirement: Squirmers swim and stir through the mobility

The catalog SHALL include `squirmers`, spherical squirmers in an unbounded Stokes fluid with the surface slip modes `B1`
and `B2`, which swim at `2 B1 / 3` along their orientation and stir the fluid as pushers (`B2 < 0`) or pullers
(`B2 > 0`) through the stresslet `4 pi eta a^2 B2 (e e - I / 3)` and the potential dipole `B1 a^3 e / 3`, with a soft
repulsion between overlapping spheres.

#### Scenario: A lone squirmer swims at its speed

- **WHEN** one squirmer runs alone
- **THEN** it moves along its orientation at the speed `2 B1 / 3`

#### Scenario: Pullers and pushers reorient each other differently

- **WHEN** two squirmers swim side by side
- **THEN** the flows of pullers and pushers turn them in opposite senses

### Requirement: Hydrodynamically coupled particles in blinking traps

The catalog SHALL include `hydrodynamically_coupled_traps`, two Brownian beads in optical traps that switch on and off
periodically, coupled by the Rotne-Prager-Yamakawa or Oseen mobility, after Argun et al. 2021, Problems 7.4 and 7.5.

#### Scenario: Free motion is correlated

- **WHEN** the traps are off
- **THEN** the displacements of the two beads along their line of centres are positively correlated as the mobility
  predicts, and the correlation vanishes without the coupling

### Requirement: Hydrodynamic systems are recorded in the system inventory

The system inventory SHALL list `squirmers` and `hydrodynamically_coupled_traps` as implemented, with their sources and
the `hydrodynamic-interactions` primitive, and the gallery SHALL hold an example of each.

#### Scenario: Inventory and gallery checks pass

- **WHEN** the inventory check and the gallery coverage check run
- **THEN** both systems are found implemented, with gallery examples
