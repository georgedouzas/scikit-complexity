# Spec Delta

## Purpose

Lets a user model networks whose structure and edge states evolve together with the states of their nodes, across
several relation layers, with neighbourhood reductions in any space.

## ADDED Requirements

### Requirement: Generators create reproducible networks
The space layer SHALL generate Erdős-Rényi, Watts-Strogatz, Barabási-Albert, configuration-model, stochastic block,
lattice, and complete network spaces from the simulation's random stream.

#### Scenario: Same seed, same network
- **WHEN** a Watts-Strogatz network is generated twice with the same seed
- **THEN** both networks have identical edges

### Requirement: Interactions read and change edge states
An interaction with the binding `'neighbours'` SHALL bind an edge group of the same network, run its laws once per
edge, and read and target the states of the edge member and of the members at its source and target.

#### Scenario: Hebbian weights
- **WHEN** an interaction grows the weights of an edge group with the product of the activities at both ends
- **THEN** edges between co-active nodes end with higher weights than edges between inactive ones

### Requirement: Layers separate relation types
A network space SHALL hold several named layers over the same nodes, and each interaction and edge group SHALL bind to
one layer.

#### Scenario: Two relations
- **WHEN** a model has a contact layer for infection and a work layer for wages
- **THEN** infection spreads only along contact edges, and wages flow only along work edges

#### Scenario: Missing layer
- **WHEN** an interaction names a layer the network space does not have
- **THEN** validation reports an issue with the code `binding` that lists the layers of the space

### Requirement: Laws change the topology
Update, jump, and rule laws SHALL request edge and node additions, removals, and rewirings, which add and remove members
of the variable groups on the edges and sites of the network, so groups on network nodes resize with the nodes.

#### Scenario: Preferential attachment
- **WHEN** a Barabási-Albert growth model adds 10,000 nodes with two edges each
- **THEN** the degree distribution's tail exponent lies within 0.3 of 3

#### Scenario: Adaptive epidemic
- **WHEN** susceptible nodes rewire away from infected neighbours at a high rate
- **THEN** the epidemic threshold rises above its value without rewiring

#### Scenario: Request on a fixed group
- **WHEN** a law requests a new edge on a network whose edge group is not variable
- **THEN** validation reports an issue with the code `group-fixed` whose fix is `variable=True`

### Requirement: Variable groups run on compiled backends
The NumPy and Numba backends SHALL run variable groups with capacity layouts and active masks, and SHALL declare their
support for variable and edge groups, so validation accepts them with solvers that support variable groups.

#### Scenario: Growth outside Mesa
- **WHEN** a network that grows by rule laws from 10 to 1,000 nodes runs under discrete time on the Numba backend
- **THEN** the run completes, every frame holds the active nodes only, and it agrees with the NumPy backend within 1e-12

### Requirement: Laws reduce over neighbourhoods
Laws SHALL use reductions of expressions over neighbours: network neighbours of a layer weighted by an edge state,
lattice neighbours, neighbours within a radius, the k nearest neighbours, or neighbours within a vision cone. Validation
SHALL report a neighbourhood that the system's space does not offer.

#### Scenario: Kuramoto synchronisation
- **WHEN** 1000 Kuramoto oscillators with Lorentzian frequencies of half-width γ run on a complete graph with coupling
  above and below `2 * γ`
- **THEN** the order parameter stays near 0 below the critical coupling and grows above it

#### Scenario: Flocking
- **WHEN** the Vicsek model runs at low noise and high density
- **THEN** the polar order parameter exceeds 0.9

#### Scenario: Metric neighbourhood on a network
- **WHEN** a law reduces over `Radius(1.0)` in a system on a network space
- **THEN** validation reports an issue with the code `law-space` that suggests a continuous space or a layer

### Requirement: Networks are recorded and drawn
Trajectories SHALL hold network snapshots whenever the topology changes, and Rerun SHALL draw them as graphs with
node and edge colors from states.

#### Scenario: Snapshot by frame
- **WHEN** a user asks a trajectory of a growing network for frame 10
- **THEN** it returns the nodes and edges present at frame 10
