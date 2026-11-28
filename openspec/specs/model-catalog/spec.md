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
- **WHEN** many urns with `balls=(1, 1)` run two discrete steps
- **THEN** the share with two balls added to the first colour is close to 1/3

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

### Requirement: Lattice spin systems record their energy
`IsingSystem`, `PottsSystem` and `XYSystem` SHALL record the observable `energy` at every frame, the total energy of
the configuration over the neighbour pairs of the lattice and, for Ising, the field term.

#### Scenario: Energy of the all up configuration
- **WHEN** `IsingSystem(8, J=1.0, h=0.3, initial='up')` runs at a low temperature
- **THEN** the energy at frame 0 is `-2 * 64 - 0.3 * 64`

#### Scenario: Energy of a flipped configuration
- **WHEN** the energy of an Ising configuration is compared with that of its global flip at `h=0`
- **THEN** the two energies are equal

### Requirement: Galam majority with sampled groups
`GalamMajoritySystem(groups='sample')` SHALL let every agent draw `group_size` agents uniformly with replacement each
round and adopt their majority, while `groups='partition'` SHALL keep the disjoint groups as the default.

#### Scenario: Mean of the sampled recursion
- **WHEN** many agents with share `p` of one opinion run one round with groups of three sampled with replacement
- **THEN** the new share is close to `3 p^2 - 2 p^3`

### Requirement: Logistic map ensembles
`LogisticMapSystem` SHALL take an array of initial values and iterate one map per value in one run, with the state
`map.x` of the shape of the array, and a scalar SHALL keep the state of shape `()`.

#### Scenario: Five maps at r of 2
- **WHEN** `LogisticMapSystem(x=np.linspace(0.01, 0.99, 5), r=2.0)` runs 50 steps
- **THEN** every map is close to the fixed point 0.5

### Requirement: Discrete-time random walks with boundaries
`RandomWalkSystem(time='discrete')` SHALL step every walker by `+1` with probability `bias` or `-1` otherwise once per
unit of time, starting at `start`, SHALL freeze a walker that reaches either end of `absorbing=(a, b)`, and SHALL map
a position to `max(floor, x + step)` when `floor` is given. `absorbing` and `floor` SHALL raise in continuous time.

#### Scenario: Gambler's ruin
- **WHEN** many fair walkers start at 3 with `absorbing=(0, 10)` and run until absorbed
- **THEN** the share absorbed at 10 is close to 0.3

#### Scenario: Floored walk
- **WHEN** walkers with `bias=0.4` and `floor=0` run long
- **THEN** no position is below 0 and the share at 0 is close to `1 - 0.4 / 0.6`

#### Scenario: Boundaries need discrete time
- **WHEN** `RandomWalkSystem(absorbing=(0, 10))` is prepared in continuous time
- **THEN** a `ValidationError` names `time='discrete'`

### Requirement: Pólya urns of any number of colours
`PolyaUrnSystem(balls=...)` SHALL hold one count per colour for two or more colours, with the state `urns.balls` of
shape `(n_urns, n_colours)`, and SHALL draw a colour with probability proportional to its count. `red` and `blue`
SHALL no longer be arguments.

#### Scenario: Dirichlet limit of three colours
- **WHEN** many urns with `balls=(1, 1, 2)` and `time='discrete'` run many draws
- **THEN** the share of the third colour has a mean close to 1/2 and a variance close to 1/20

### Requirement: Discrete laws of the Gibrat factor
`GibratProcessSystem` SHALL draw the growth factor from `factors` with probabilities `weights` when `factors` is
given, and from the normal law of `mu` and `sigma` otherwise.

#### Scenario: Multiplicative gamble
- **WHEN** many firms run 20 steps with `factors=(1.5, 0.6)` and `weights=(0.5, 0.5)`
- **THEN** the mean size grows close to `1.05^20` and the median size close to `0.9^10`

### Requirement: Langton's ant on given colours and on the plane
`LangtonsAntSystem` SHALL take initial colours, a start site and a heading, SHALL record the ant's position and
heading at every step, SHALL update only the ant's cell per step, and SHALL run on the unbounded plane when
`shape=None`.

#### Scenario: Highway on the plane
- **WHEN** the ant runs 12000 steps on the unbounded plane from a blank start
- **THEN** it runs in a few seconds and its last 104 steps move it by the period of the highway

#### Scenario: Plane and large torus agree
- **WHEN** the ant runs 5000 steps on the plane and on a torus too large to wrap
- **THEN** its positions are the same

### Requirement: Vectorised and variant Boolean networks
`RandomBooleanNetworkSystem` SHALL hold its nodes in one group with the state `nodes.x` of shape `(n_nodes,)`, SHALL
update them asynchronously, one random node at a time, when `update='asynchronous'`, and SHALL redraw the inputs and
tables every step when `annealed=True`.

#### Scenario: Large network
- **WHEN** a network of `10_000` nodes with `k=4` runs 50 steps
- **THEN** it runs in seconds and `nodes.x` has shape `(10_000,)`

#### Scenario: Annealed overlap
- **WHEN** two copies of an annealed network with `k=2` and `bias=0.5` start at a small distance
- **THEN** their distance follows the annealed approximation of Derrida and Pomeau
