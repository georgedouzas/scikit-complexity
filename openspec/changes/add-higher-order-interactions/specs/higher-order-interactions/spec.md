# Spec Delta

## Purpose

Lets a user run laws over hyperedges of more than two members, on hypergraphs and simplicial complexes, let nodes
regulate links, and couple populations through random tensors of any order.

## ADDED Requirements

### Requirement: Network spaces with hyperedges come from XGI
The space layer SHALL build a network space with hyperedges of any size from an XGI hypergraph, an XGI simplicial
complex, or a list of node groups, with its incidence matrix and the hyperedges around each node.

#### Scenario: Triangle hypergraph
- **WHEN** a network space with the triangles of a complete graph on five nodes as hyperedges is built
- **THEN** it has 10 hyperedges of size 3 and every node lies in 6 of them

### Requirement: Laws act on hyperedges
An interaction with the binding `'hyperedges'` SHALL run its laws once per hyperedge of its declared size, its laws
SHALL read all members of the hyperedge, and a law SHALL reduce over the hyperedges of a node, compiled to gathers on
NumPy and Numba.

#### Scenario: Three-body Kuramoto
- **WHEN** phase oscillators couple only through the triangles of a hypergraph with the three-body term
  `sin(theta_j + theta_k - 2 theta_i)`
- **THEN** the compiled drift equals a direct loop over the triangles within 1e-12 on both backends

#### Scenario: Missing hyperedge size
- **WHEN** an interaction over hyperedges of size 4 is placed in a network space whose hyperedges all have size 3
- **THEN** validation reports an issue with the code `binding` that names the size and the sizes the space has

### Requirement: Nodes regulate links
An interaction over hyperedges SHALL bind an edge group of the same network through an edge role, whose member is the
edge between two declared members of the hyperedge, so a regulating node switches the activity of a link with an
update law each step.

#### Scenario: Triadic percolation
- **WHEN** links are regulated by the activity of their regulating nodes as in Millán et al. (2023)
- **THEN** the giant component shows period two above the bifurcation point of the paper within the finite-size error

### Requirement: Couplings can be random tensors
The component layer SHALL provide quenched random coupling tensors of order k over the members of a group, symmetric or
general, with a given mean and variance, as parameters of groups.

#### Scenario: Tensor statistics
- **WHEN** a symmetric order-3 tensor over 100 species with variance 1/N^2 is drawn
- **THEN** its entries are symmetric under permutation and their sample variance is within 5% of 1/N^2

#### Scenario: Coupling tensor on an element
- **WHEN** a coupling tensor is given as a parameter of a single element
- **THEN** validation reports an issue with the code `parameter-shape` that suggests a group
