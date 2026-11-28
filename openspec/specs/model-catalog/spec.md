# system-catalog Specification

## Purpose
Lists the registered systems and creates, describes, and runs them by name, importing only the systems that are used.

## Requirements

### Requirement: The documentation holds a browsable catalog
The documentation SHALL hold an index of every registered system by field and tag, with its description, generated from
`all_systems()`. Each system page SHALL show how to import the system class and list the element and interaction classes its system
uses.

#### Scenario: Index follows the catalog
- **WHEN** a system is registered
- **THEN** the next documentation build lists it in the index under its field and its tags

#### Scenario: System page
- **WHEN** a reader opens the page of `LotkaVolterraSystem`
- **THEN** it shows `from skcomplex.systems.biology.ecology import LotkaVolterraSystem`, and lists the element and interaction classes of its system

### Requirement: Ready-made systems are imported from their subfield
Every ready-made system SHALL be a public class of its subfield package, imported as
`from skcomplex.systems.<field>.<subfield> import <Name>System`, and SHALL carry its field, topics, references, and
tuned run settings as class attributes. `simulate` SHALL use the tuned run settings for every argument the call leaves
out.

#### Scenario: Direct use
- **WHEN** a user runs `SIRSystem().prepare().simulate()` after
  `from skcomplex.systems.biology.epidemiology import SIRSystem`
- **THEN** the run uses the class's tuned end time, solver, and observables

### Requirement: Systems are discovered with all_systems
`all_systems(field=None, topics=(), tags=None)` SHALL return the name and class of every ready-made system, in the role
of `sklearn.utils.all_estimators()`. Selecting by field or topics SHALL read the class definitions statically and import
only the modules of the systems it returns. Filtering by capability tags SHALL import the systems it filters.

#### Scenario: Listing by field without imports
- **WHEN** a user calls `all_systems(field='physics/mechanics')` in a fresh interpreter
- **THEN** the systems of that field are returned, and no module of another subfield is imported

#### Scenario: Filtering by capability
- **WHEN** a user calls `all_systems(tags={'randomness': 'stochastic', 'space': 'network'})`
- **THEN** every listed system has those tags

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

### Requirement: Pólya urns draw once per step in discrete time
`PolyaUrnSystem(time='discrete')` SHALL draw exactly one ball in every urn per step, and its docstring SHALL say that
in continuous time the draws happen at the times of a Poisson process of rate 1.

#### Scenario: Two reds in two draws
- **WHEN** many urns with one red and one blue ball run two discrete steps
- **THEN** the share with two reds is close to 1/3

### Requirement: Galton Watson offspring laws and generation sizes
`GaltonWatsonSystem` SHALL take a Poisson, geometric, or finite offspring law, and SHALL track the size of every
generation as one count when `individuals=False`.

#### Scenario: Geometric offspring
- **WHEN** a Galton Watson process with geometric offspring of mean 0.8 runs many times
- **THEN** the mean size of generation 3 is close to 0.8 cubed

#### Scenario: Cheap supercritical runs
- **WHEN** 400 runs with Poisson offspring of mean 2.5 over 12 generations run with `individuals=False`
- **THEN** they finish in a few seconds and their mean sizes match those of the individuals

#### Scenario: Finite law
- **WHEN** the offspring probabilities are `[0.25, 0.5, 0.25]`
- **THEN** the mean size of generation 1 from one individual is close to 1

### Requirement: West Brown Enquist runs scale with the vessels
`WestBrownEnquistSystem` SHALL run in a time that grows roughly with the number of vessels as its depth grows.

#### Scenario: One level deeper
- **WHEN** a network of branching ratio 2 runs at depth 7 and at depth 8
- **THEN** the second run takes less than four times as long as the first
