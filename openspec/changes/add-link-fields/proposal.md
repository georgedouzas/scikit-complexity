# Proposal

## Why

Schwichtenberg (2020 §2.1, §4.4) builds gauge theory from exchange rates. A good sits at the sites of a lattice, the
exchange rate between neighbouring sites lives on the link that joins them, and an arbitrage opportunity is the product
of the rates around a closed loop, a plaquette. The same structure gives lattice Maxwell theory, with the vector
potential on links and the field strength on plaquettes, and Wilson's lattice gauge theory, with angle-valued link
variables sampled by Monte Carlo. Millán, Torres, and Bianconi (2203.06601) couple phases on the nodes and links of a
simplicial complex through its boundary operators, and Mihaylov and Cacciatori (2509.01581) treat agents on a network as
a gauge theory whose frustrated loops block agreement. The inventory records `lattice_gauge_fields`,
`topological_kuramoto`, and `gauge_arbitrage_market` as blocked on this change, and `gauge_network_agents` as pending
it.

The current core already holds states on the parts of a space and samples them. `GroupElement(on='edges')` and
`GroupElement(on='hyperedges')` hold one member per edge or hyperedge of a `NetworkSpace(hyperedges=)`. Angle states,
`StateSpec(period=2 * pi)`, give U(1) values, and integer states with a period give Z_N values. `SamplingInteraction`
already samples a state by Metropolis, Glauber, or heat-bath acceptance with the `'rotate'` and `'random-value'` moves
and the `'colored'` schedule over a greedy coloring of the conflict graph. `CouplingInteraction(mode='edges')` already
reads edge states. What is missing is a space whose hyperedges are the oriented faces of a lattice or a network, with
the incidence between its cells, the exterior calculus that maps site states to link states to face states, a
reduction over the faces that contain a link, sampling of edge groups, the analysis of edge flows and complexes, and a
visual of cells. This change adds those and the four models.

## What Changes

- Add `CellComplexSpace` to `skcomplex.core.spaces`, a `NetworkSpace` derived from a `LatticeSpace` or a `NetworkSpace`:
  its nodes are the sites or nodes, its edges the oriented links or edges, and its hyperedges the oriented plaquettes or
  faces in cyclic node order, taken from the lattice, from given node cycles, from the hyperedges of the network, from a
  minimum cycle basis, or from the cliques of the network. It holds the oriented incidence between consecutive degrees.
- Add the exterior calculus to `skcomplex.core.laws.utils`: `exterior_derivative`, `codifferential`, and `hodge_star`,
  which act on the states of groups on the sites, edges, and hyperedges of a cell complex, lowered to sparse incidence
  products on NumPy and gather loops on Numba.
- Add `CofaceNeighborhood` to `skcomplex.core.spaces`, the faces that contain an edge or the edges that contain a site,
  so `neighbors` reduces over them and `exterior_derivative` inside such a reduction reads the oriented sum around one
  face.
- Let `SamplingInteraction` target a group on the edges of a cell complex, with the `'colored'` schedule over edges that
  share no face.
- Add the compile issue `law-degree` for an exterior operation or a sampling on a state of the wrong degree.
- Add `CellComplexAnalysis` to `skcomplex.analysis`, with the kinds `'hodge'`, the Hodge decomposition of an edge flow,
  `'betti'`, the Betti numbers of a complex, and `'curvature'`, the faces whose oriented sum of an angle field is not a
  multiple of its period.
- Add `CellVisual` to `skcomplex.engine.viz`, which streams edge groups as colored segments and hyperedge groups as an
  image on a lattice complex or as colored polygons on a network complex.
- Add the models `lattice_gauge_fields` in `physics/waves`, `topological_kuramoto` in `physics/synchronization`,
  `gauge_arbitrage_market` in `economics/finance`, and `gauge_network_agents` in `social/opinion_dynamics`, with their
  field cases as thin subclasses in their packages.

## Capabilities

### New Capabilities

- `link-fields`: cell complexes of lattices and networks, exterior calculus, coface reductions, sampling of edge
  groups, the degree check, the analysis of edge flows and complexes, and the cell visual.

### Modified Capabilities

- `model-library`: the lattice gauge, topological Kuramoto, gauge arbitrage, and gauge network models.

## Impact

- `skcomplex.core.spaces` gains `CellComplexSpace` and `CofaceNeighborhood`.
- `skcomplex.core.laws.utils` gains `exterior_derivative`, `codifferential`, and `hodge_star`.
- `skcomplex.core.interactions.SamplingInteraction` accepts edge groups of a cell complex.
- `skcomplex.engine.compile` lowers the exterior calculus and coface reductions, and `PassCompiler` gains the issue
  `law-degree`. `skcomplex.engine.backends` gains their NumPy and Numba kernels.
- `skcomplex.analysis` gains `CellComplexAnalysis`, and `skcomplex.engine.viz` gains `CellVisual`.
- `skcomplex.models` gains four models, and `docs/overview/inventory.toml` marks them implemented.
- No new runtime dependency. The feature builds on NumPy, SciPy sparse matrices, NetworkX, and Numba.
