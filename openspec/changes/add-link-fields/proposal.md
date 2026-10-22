# Proposal

## Why

Schwichtenberg (2020 §2.1, §4.4) builds gauge theory from exchange rates. A good sits at the sites of a lattice, the
exchange rate between neighbouring sites lives on the link that joins them, and an arbitrage opportunity is the product
of the rates around a closed loop, a plaquette. The same structure gives lattice Maxwell theory, with the vector
potential on links and the field strength on plaquettes, and Wilson's lattice gauge theory, with group-valued link
variables sampled by Monte Carlo. The inventory records `lattice_gauge_fields` as out of scope because the framework
places every state on the sites of a lattice or the nodes of a network, and has no state on edges or faces, no
group-valued state, no product around a loop, and no exterior calculus that maps sites to links to faces.

The missing feature is general. Flows on the edges of transport, trade, and power networks, edge weights that change
over time, and the Hodge decomposition of edge flows into gradient, curl, and harmonic parts all need states on edges
and faces. The user directive puts every surveyed model in scope, so this change adds the capability and makes the
lattice gauge model a first-class registered model.

**Rebased on `redesign-system-model`.** The redesign already places state on the parts of a space: a group with
`on='edges'` or `on='hyperedges'` holds one member per edge or hyperedge of a `NetworkSpace(hyperedges=)`, and
`Sample` is the one Monte Carlo law, with its energy, move, and acceptance as parts. This change therefore no longer
adds a `LinkField` component or a `degree` keyword on states: link and plaquette states are edge and hyperedge groups
on the cell complex of a lattice or a network. It still adds the cell complex, the exterior calculus, compact
group-valued domains, holonomies, link moves and schedules of `Sample`, the general analysis of edge flows and
complexes, and the link visual. The physics-named pieces, the Wilson action and the gauge observables, move to the
model package `skcomplex.models.physics.waves` instead of the core.

## What Changes

- Add cell complexes in `core.space`: `CellComplex(space, faces=None, dimension=2)` derives, from a `LatticeSpace` or a
  `NetworkSpace`, a network space whose edges are the oriented links or edges and whose hyperedges are the oriented
  plaquettes or faces, given in cyclic node order, with the incidence matrices between consecutive degrees and the
  cells around each cell. Cubes and larger simplices are held for incidence and topology, without states.
- Link and plaquette states are ordinary groups, `Group(on='edges')` and `Group(on='hyperedges')`, on a cell complex.
  No new component kind is added.
- Add discrete exterior calculus in `core.laws`: the exterior derivative, the Hodge star of a lattice with spacings or
  of declared network weights, and the codifferential. The existing `gradient`, `divergence`, and `laplacian` lower to
  these on edge and hyperedge groups, and a new `curl` maps links to plaquettes.
- Add the compact group domain `CompactGroup` in `core.components`, with `DomainKind.COMPACT_GROUP`: U(1) angles,
  Z_N integers, and SU(2) unit quaternions, with the group product, the inverse, the trace, the projection, and the
  exact heat-bath draw of each group.
- Add holonomies in `core.laws`: the ordered product of link variables around a plaquette or any closed path, and the
  staple sum of a link, as symbolic functions that the compiler lowers to index tables.
- Extend `Sample` to groups on edges: a local energy over the plaquettes that contain a link, Metropolis, heat-bath,
  and overrelaxation moves on compact groups, and the `'colored'` schedule over links that share no plaquette.
- Add the general analysis of complexes in `core.analysis`: the Hodge decomposition of an edge flow into its gradient,
  curl, and harmonic parts, the frustration of a link field, and the Betti numbers of a complex.
- Add clique complexes of networks, after Mihaylov and Cacciatori (2025, arXiv 2509.01581): the clique complex of a
  network up to a chosen dimension, so triangles are faces of agent interaction networks.
- Add a link visual that streams edge groups to Rerun as colored segments or arrows and hyperedge groups as images or
  colored faces.
- Declare every new primitive for validation: the cell complex as a network space kind, the compact group domain, the
  exterior calculus and holonomies with the parts of the space they read, and the link support of the Monte Carlo
  solver and of the NumPy and Numba backends, so `validate()` reports a degree mismatch, a compact group under a rate
  law, or a colored schedule on an odd lattice before compilation.
- Add the library model `lattice_gauge_fields` in `physics/waves`, with Monte Carlo sampling of the Wilson action for
  U(1), Z_N, and SU(2), and the real-time evolution of lattice Maxwell fields, with a gallery example. Its package
  holds the Wilson action and the gauge observables: gauge transformations, the mean plaquette, Wilson and Polyakov
  loops, the topological charge, and the Gauss law residual.
- Add the library model `gauge_network_agents` in `social/opinion_dynamics`, agents with internal U(1) phases on a
  network whose edges carry connections, after Mihaylov and Cacciatori, with a gallery example.

## Capabilities

### New Capabilities

- `link-fields`: cell complexes of lattices and networks, edge and hyperedge groups on them, discrete exterior calculus,
  compact group domains, holonomies, link Monte Carlo, the Hodge decomposition of edge flows, clique complexes,
  frustration, Betti numbers, the link visual, and their validation declarations.

### Modified Capabilities

- `model-library`: the lattice Maxwell and gauge link field model in `physics/waves`, with its Wilson action and gauge
  observables, and the gauge network agents model in `social/opinion_dynamics`.

## Impact

- `skcomplex.core.base` gains `DomainKind.COMPACT_GROUP`; `skcomplex.core.components` gains the domain `CompactGroup`.
- `skcomplex.core.space` gains `CellComplex`, a network space kind.
- `skcomplex.core.laws` gains `curl`, `exterior_derivative`, `codifferential`, `hodge_star`, `holonomy`, and `staples`,
  and the parts of `Sample` gain the `overrelax` and `group` moves and the `'colored'` schedule for links.
- `skcomplex.core.compile` gains the lowering of edge and hyperedge states, exterior calculus, and holonomies, and
  `skcomplex.core.backends` gains their NumPy and Numba kernels, with declared support.
- `skcomplex.core.analysis` gains `hodge_decomposition`, `frustration`, and `betti_numbers`; `CellComplex` gains
  `clique_complex`.
- `skcomplex.core.viz` gains `LinksVisual`, registered under the kind `'links'`.
- `skcomplex.models.physics.waves` gains `lattice_gauge_fields`, the Wilson action, and `gauge_transform`,
  `wilson_loop`, `polyakov_loop`, `topological_charge`, and `gauss_residual`.
- No new runtime dependency. The feature builds on NumPy, SciPy sparse matrices, NetworkX, and Numba, which the
  project already depends on, after the survey in the design found no maintained Python library that fits.
- The inventory entry of `lattice_gauge_fields` moves from out of scope to pending this change.
