## ADDED Requirements

### Requirement: Ready-made systems name their seed random_state
Every ready-made system that draws values when prepared SHALL take its seed as `random_state`, and its docstring SHALL
say that it seeds what the system draws when prepared while `simulate(random_state=)` seeds the dynamics.

#### Scenario: One name
- **WHEN** the signatures of all ready-made systems are read
- **THEN** none has a parameter named `seed`

#### Scenario: Reproducible flock
- **WHEN** a `VicsekSystem(random_state=4)` simulates twice with `random_state=1`
- **THEN** both trajectories are identical

### Requirement: Lattice samplers choose a schedule that fits the lattice
A ready-made lattice system whose solver is `'auto'` SHALL run on periodic lattices of odd side, and an explicit
schedule that cannot run SHALL raise an error that names how to choose another one through the `solver` setting.

#### Scenario: Odd Ising lattice
- **WHEN** `IsingSystem(81, temperature=2.0)` is prepared and simulated
- **THEN** it runs with the random sequential schedule

#### Scenario: Explicit checkerboard on an odd lattice
- **WHEN** an Ising system of side 81 sets a Metropolis solver with the checkerboard schedule and is prepared
- **THEN** a validation error names `solver=MetropolisSolver(schedule='random-sequential')`
