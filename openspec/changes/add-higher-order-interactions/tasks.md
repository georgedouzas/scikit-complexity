# Tasks

## 1. Hypergraphs

- [ ] 1.1 Survey XGI and HyperNetX, record the result, and add the chosen one as a runtime dependency
- [ ] 1.2 Build network spaces with hyperedges from XGI and node groups, with incidence matrices, and verify the triangle
  hypergraph scenario
- [ ] 1.3 Run the laws of interactions over hyperedges by size, and add reductions over hyperedges, with their compiled
  gathers, and verify the three-body Kuramoto scenario on both backends
- [ ] 1.4 Add the edge role of interactions over hyperedges, and verify the triadic percolation scenario
- [ ] 1.5 Add `CouplingTensor`, and verify the tensor statistics scenario

## 2. Declarations and validation

- [ ] 2.1 Declare the support of the NumPy and Numba backends for hyperedge groups and the requirements of hyperedge
  reductions, edge roles, and coupling tensors, and verify the missing size and coupling tensor on an element scenarios

## 3. Library

- [ ] 3.1 Add the four probe models with curated gallery examples and review rows, verify the explosive synchronization
  and bistable contagion scenarios, and mark their inventory entries implemented

## 4. Gate

- [ ] 4.1 Document hypergraphs and higher-order interactions in the user guide, and run formatting, checks, the docs
  build, and tests through nox
