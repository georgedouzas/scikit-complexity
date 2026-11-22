## ADDED Requirements

### Requirement: Network potentials run in LAMMPS
`LammpsSolver` SHALL run a group moved by an equivariant network potential through its `fix external`, evaluating the
network outside LAMMPS at every step and passing its gradient to LAMMPS, together with the pair energies and fixes it
already translates, without naming the interaction.

#### Scenario: LAMMPS agrees with the native solver
- **WHEN** a group of 64 members with a small untrained network potential runs for 100 steps with velocity Verlet and
  with `LammpsSolver` from the same state and step
- **THEN** the positions agree within 1e-6

#### Scenario: Network and thermostat together
- **WHEN** the MACE water system is prepared with `LammpsSolver` and its Langevin thermostat and simulated
- **THEN** the run starts without an unsupported-law error, and its trajectory has the shapes of a native run
