# Spec Delta

## Purpose

Lets a user declare interactions among groups of more than two components, on hypergraphs and simplicial complexes, let nodes regulate links, and couple populations through random tensors of any order.

## ADDED Requirements

### Requirement: Hypergraphs are spaces
The modelling layer SHALL provide hypergraph spaces whose hyperedges have any size, built from an XGI hypergraph or a list of node groups, with incidence matrices and the hyperedges around each node.

#### Scenario: Triangle hypergraph
- **WHEN** a hypergraph of the triangles of a complete graph on five nodes is built
- **THEN** it has 10 hyperedges of size 3 and every node lies in 6 of them

### Requirement: Laws act on hyperedges
An interaction role SHALL bind to every hyperedge of a size, its laws SHALL read all members, and a law SHALL reduce over the hyperedges of a node, compiled to gathers on NumPy and Numba.

#### Scenario: Three-body Kuramoto
- **WHEN** phase oscillators couple only through the triangles of a hypergraph with the three-body term `sin(theta_j + theta_k - 2 theta_i)`
- **THEN** the compiled drift equals a direct loop over the triangles within 1e-12 on both backends

### Requirement: Nodes regulate links
A regulation SHALL switch a link on or off as a function of the state of a regulating node, applied between steps.

#### Scenario: Triadic percolation
- **WHEN** links are regulated by the activity of their regulating nodes as in Millán et al. (2023)
- **THEN** the giant component shows period two above the bifurcation point of the paper within the finite-size error

### Requirement: Couplings can be random tensors
The modelling layer SHALL provide quenched random coupling tensors of order k, symmetric or general, with a given mean and variance, as parameters of groups.

#### Scenario: Tensor statistics
- **WHEN** a symmetric order-3 tensor over 100 species with variance 1/N^2 is drawn
- **THEN** its entries are symmetric under permutation and their sample variance is within 5% of 1/N^2
