## ADDED Requirements

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
