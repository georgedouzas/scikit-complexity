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

## What Changes

- Add cell complexes: the sites, oriented links, oriented plaquettes, and cubes of a `LatticeSpace`, and the nodes,
  oriented edges, and chosen faces of a `NetworkSpace`, with their incidence matrices and the cells around each cell.
- Add link fields: a group whose states are declared with a cell degree, 0 for sites, 1 for links, and 2 for
  plaquettes, so one element type holds a potential on links and a field strength on plaquettes.
- Add discrete exterior calculus on link fields: the exterior derivative, the Hodge star of a lattice with spacings, and
  the codifferential. The existing `gradient`, `divergence`, and `laplacian` lower to these on cell states, and a new
  `curl` maps links to plaquettes.
- Add compact group-valued states: U(1) angles, Z_N integers, and SU(2) unit quaternions, with the group product, the
  inverse, and the trace.
- Add holonomies: the ordered product of link variables around a plaquette or any closed path, and the staple sum of
  a link, as symbolic functions that the compiler lowers to index tables.
- Extend Monte Carlo to links: a local energy over the plaquettes that contain a link, with Metropolis, heat-bath, and
  overrelaxation updates and a colored schedule of links that share no plaquette. The Wilson action is a ready-made
  energy.
- Add gauge transformations and gauge-invariant observables: the mean plaquette, Wilson loops, Polyakov loops, the
  topological charge of two-dimensional U(1), and the Gauss law residual.
- Add the Hodge decomposition of an edge flow into its gradient, curl, and harmonic parts, as an analysis function.
- Add a link visual that streams link states to Rerun as colored segments or arrows and plaquette states as images or
  colored faces.
- Add simplicial complexes of networks, after Mihaylov and Cacciatori (2025, arXiv 2509.01581): the clique complex of a
  network up to a chosen dimension, so triangles and higher simplices are cells of agent interaction networks, with the
  same incidence matrices and cell states as lattices.
- Add frustration and topology measures: the frustration of a link field, the share and the energy of plaquettes whose
  holonomy is not the identity, the Betti numbers of a complex from the ranks of its boundary matrices, and the
  topological charge of a U(1) link field over any closed oriented surface of a complex, so complexity can be measured as
  a local and a topological obstruction to consistent order.
- Add the library model `gauge_network_agents` in `social/opinion_dynamics`, agents with internal U(1) phases on a
  network whose edges carry connections, after Mihaylov and Cacciatori, with a gallery example.
- Add the library model `lattice_gauge_fields` in `physics/waves`, with Monte Carlo sampling of the Wilson action for
  U(1), Z_N, and SU(2), and the real-time Hamiltonian evolution of lattice Maxwell fields, with a gallery example.

## Capabilities

### New Capabilities

- `link-fields`: cell complexes of lattices and networks, states on links and plaquettes, discrete exterior calculus,
  compact group-valued states, holonomies, link Monte Carlo, gauge transformations and gauge-invariant observables,
  the Hodge decomposition of edge flows, simplicial complexes of networks, frustration, Betti numbers, and the
  link visual.

### Modified Capabilities

- `model-library`: the lattice Maxwell and gauge link field model in `physics/waves`, and the gauge network agents model
  in `social/opinion_dynamics`.

## Impact

- `skcomplex.core.base` gains the domain `GaugeGroup`, and `state` gains a `degree` keyword.
- `skcomplex.core.model` gains `CellComplex`, `LinkField`, `WilsonAction`, `curl`, `exterior_derivative`,
  `codifferential`, `hodge_star`, `holonomy`, and `staples`. `Move` gains the `overrelax` and `group` kinds, and
  `Sampler` gains the `'colored'` schedule for links.
- `skcomplex.core.compile` gains the lowering of cell states, exterior calculus, and holonomies to NumPy and Numba.
- `skcomplex.core.analysis` gains `gauge_transform`, `wilson_loop`, `polyakov_loop`, `topological_charge`,
  `gauss_residual`, `hodge_decomposition`, `frustration`, and `betti_numbers`; `CellComplex` gains `clique_complex`.
- `skcomplex.core.viz` gains `LinksVisual`, registered under the kind `'links'`.
- No new runtime dependency. The feature builds on NumPy, SciPy sparse matrices, NetworkX, and Numba, which the
  project already depends on, after the survey in the design found no maintained Python library that fits.
- The inventory entry of `lattice_gauge_fields` moves from out of scope to pending this change.
