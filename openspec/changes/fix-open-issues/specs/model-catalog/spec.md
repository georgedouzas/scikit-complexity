## ADDED Requirements

### Requirement: Ready-made systems name their seed random_state
Every ready-made system that draws values when prepared SHALL take the seed of those draws as `random_state`, or as a
name that says what it fixes, such as `landscape_seed` for a fitness landscape. The docstring of
`simulate(random_state=)` and the user guide SHALL say that it seeds the dynamics only, while the system's own
`random_state` seeds what it draws when prepared.

#### Scenario: One name
- **WHEN** the signatures of all ready-made systems are read
- **THEN** none has a parameter named `seed`

#### Scenario: Reproducible flock
- **WHEN** a `VicsekSystem(random_state=4)` simulates twice with `random_state=1`
- **THEN** both trajectories are identical

### Requirement: Coevolving networks start with the degrees of their initial network
The coevolving voter and join-a-club systems SHALL start the degree of every agent at its degree in the initial
network, so the first frame records it.

#### Scenario: Initial degrees
- **WHEN** a `CoevolvingVoterSystem(200, mean_degree=4)` simulates
- **THEN** the degrees of frame 0 sum to twice the number of initial edges

### Requirement: Lattice samplers choose a schedule that fits the lattice
A ready-made lattice system whose solver is `'auto'` SHALL run on periodic lattices of odd side, and an explicit
schedule that cannot run SHALL raise an error that names how to choose another one through the `solver` setting.

#### Scenario: Odd Ising lattice
- **WHEN** `IsingSystem(81, temperature=2.0)` is prepared and simulated
- **THEN** it runs with the random sequential schedule

#### Scenario: Explicit checkerboard on an odd lattice
- **WHEN** an Ising system of side 81 sets a Metropolis solver with the checkerboard schedule and is prepared
- **THEN** a validation error names `solver=MetropolisSolver(moves=..., schedule='random-sequential')`
