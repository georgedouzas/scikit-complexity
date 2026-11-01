# Tasks

## 1. Hyperedges

- [x] 1.1 Survey XGI and HyperNetX, record the result in design.md, and add the chosen library as a runtime dependency
  in `pyproject.toml` and the Project Profile; verify `deptry` passes
- [x] 1.2 Let `NetworkSpace` take XGI hypergraphs and simplicial complexes as `hyperedges`, deriving `incidence_` and
  `memberships_`, with tests in `tests/core/spaces/test_network.py`; verify the triangle hypergraph scenario

## 2. Laws per hyperedge

- [x] 2.1 Add the `size` argument and the namespace of one hyperedge to `HyperedgeInteraction`, the hyperedge pass in
  `engine/compile/passes/_hyperedges.py`, and its gather kernels on the NumPy and Numba backends; verify the hyperedge
  law against a loop and the missing size scenarios
- [x] 2.2 Bind a group on hyperedges or edges through the second role of `HyperedgeInteraction`, with the `binding`
  issue of an edge role that joins no edge; verify a hyperedge group update and the issue
- [x] 2.3 Add the reduction `hyperedges` to `skcomplex.core.laws.utils` and the mode `'hyperedges'` of
  `CouplingInteraction`; verify the three-body Kuramoto and the hyperedge reduction outside a hypergraph scenarios

## 3. Symbolic groupings

- [ ] 3.1 Add `form`, `write`, `function`, and `reduce` to `GroupingInteraction`, lowering the symbolic form to an
  `UpdateLaw` on the hyperedge path with per-step tables for pairs and partitions, fixed tables for neighbourhoods, and
  keyed reductions for groups by state; verify the majority in partitions and the agent setting scenarios

## 4. Tensors and growth

- [ ] 4.1 Add the reduction `contraction` with its NumPy and Numba lowering, on the shaped parameters of
  `add-learning-dynamics`, and the `parameter-shape` issue of a tensor on an element; verify the contraction and tensor
  on an element scenarios
- [x] 4.2 Add the `hyperedge` argument of `TopologyInteraction(action='add-node')` with growing hyperedge tables;
  verify the triangle attachment scenario

- [ ] 4.3 Add the connected-components reduction to `skcomplex.core.laws.utils` with its NumPy and Numba lowering, giving component labels and the largest component of the active links of a network; verify it against networkx on random graphs

## 5. Models

- [ ] 5.1 Add `hypergraph_kuramoto`, `simplicial_contagion`, `triadic_percolation`, and
  `higher_order_random_lotka_volterra` with curated gallery examples; verify the explosive synchronization and
  bistable contagion scenarios
- [ ] 5.2 Add `hypergraph_consensus`, `simplicial_social_dilemmas`, `hypergraph_public_goods`,
  `hypergraph_exposure_sis`, `hypergraph_random_walk`, and `growing_simplicial_complex`; verify each model's tests
- [ ] 5.3 Rebuild `galam_majority` and `public_goods_game` on the symbolic grouping, `DiscussionsInteraction` as a thin
  subclass; verify the Galam scenario and the public goods game over 20 seeds
- [ ] 5.4 Mark the ten probe entries implemented in `docs/overview/inventory.toml`; verify the inventory check passes

## 6. Documentation and gate

- [ ] 6.1 Document hypergraphs, laws per hyperedge, and symbolic groupings in the user guide, and add the changelog
  entry; verify the documentation test passes
- [ ] 6.2 Run formatting, checks, the docs build, and tests through nox; verify all of them pass
