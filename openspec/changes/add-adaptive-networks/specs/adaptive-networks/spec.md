# Spec Delta

## Purpose

Lets a user model networks whose structure and edge states evolve together with the states of their nodes, across
several relation layers, with neighbourhood reductions in any space.

## ADDED Requirements

### Requirement: Generators create reproducible networks
The modelling layer SHALL generate Erdős-Rényi, Watts-Strogatz, Barabási-Albert, configuration-model, stochastic block,
lattice, and complete networks from the simulation's random stream.

#### Scenario: Same seed, same network
- **WHEN** a Watts-Strogatz network is generated twice with the same seed
- **THEN** both networks have identical edges

### Requirement: Edges carry states with laws
Edges SHALL carry states with their own laws, which read the states of their source and target nodes.

#### Scenario: Hebbian weights
- **WHEN** edge weights grow with the product of their nodes' activities
- **THEN** edges between co-active nodes end with higher weights than edges between inactive ones

### Requirement: Layers separate relation types
A network SHALL hold several named layers over the same nodes, and each interaction SHALL bind to one layer.

#### Scenario: Two relations
- **WHEN** a model has a contact layer for infection and a work layer for wages
- **THEN** infection spreads only along contact edges, and wages flow only along work edges

### Requirement: Laws change the topology
Update, jump, and rule laws SHALL add and remove edges and nodes and rewire edges, and groups on network nodes SHALL
resize with the nodes.

#### Scenario: Preferential attachment
- **WHEN** a Barabási-Albert growth model adds 10,000 nodes with two edges each
- **THEN** the degree distribution's tail exponent lies within 0.3 of 3

#### Scenario: Adaptive epidemic
- **WHEN** susceptible nodes rewire away from infected neighbours at a high rate
- **THEN** the epidemic threshold rises above its value without rewiring

### Requirement: Laws reduce over neighbourhoods
Laws SHALL use reductions of expressions over neighbours: network neighbours weighted by edges, neighbours within a
radius, the k nearest neighbours, or neighbours within a vision cone.

#### Scenario: Kuramoto synchronisation
- **WHEN** 1000 Kuramoto oscillators with Lorentzian frequencies of half-width γ run on a complete graph with coupling
  above and below `2 * γ`
- **THEN** the order parameter stays near 0 below the critical coupling and grows above it

#### Scenario: Flocking
- **WHEN** the Vicsek model runs at low noise and high density
- **THEN** the polar order parameter exceeds 0.9

### Requirement: Networks are recorded and drawn
Trajectories SHALL hold network snapshots whenever the topology changes, and Rerun SHALL draw them as graphs with
node and edge colors from states.

#### Scenario: Snapshot by frame
- **WHEN** a user asks a trajectory of a growing network for frame 10
- **THEN** it returns the nodes and edges present at frame 10
