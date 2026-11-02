# Proposal

## Why

Lattice laws run only on boxes of the square lattice `Z^d`, and they update every site or a sweep of sites. Four
models in the inventory need more than that, and they are recorded as blocked on this change:
- Ising on a graph (Argun et al. 2021 §2.1) samples spins whose neighbours are the edges of an arbitrary network;
- the forest fire on hexagonal lattices and graphs (Argun et al. 2021 ch. 3) spreads fire to six neighbours, or to
  the neighbours of a node;
- the FHP lattice gas (Dilão 2023 §5.16) moves particles along the six directions of a triangular lattice and
  collides them at the sites;
- Bak-Sneppen (Gros 2015 §6.6, Thurner et al. 2018 §5.4.4, Jensen 2023 §4.2) replaces the least fit species and its
  neighbours, one extremal site at a time.

The framework already holds networks in `NetworkSpace`, and fields already read graph neighbours. Local rules, local
energies, and samplers cannot.

## What Changes

- Add a triangular geometry to lattice spaces. Sites keep integer coordinates on a skewed grid, so every existing
  boundary and the vectorised neighbour tables carry over, and sites have positions in the plane for drawing.
- Add the six-neighbour triangular neighbourhood, ordered counterclockwise so that direction `k + 3` is opposite to
  direction `k`. The cells of a triangular lattice of sites are the hexagons of a hexagonal tiling.
- Let lattice groups live on a `NetworkSpace`, with one site per node, so local rules, drives, relaxation, and Monte
  Carlo laws run on arbitrary graphs, including the honeycomb, triangular, and grid graphs that NetworkX generates.
- Add a graph neighbourhood, the adjacency of a site, padded to the largest degree, with the number of real
  neighbours passed to rules that ask for it.
- Add a neighbour sum for local energies, which sums an expression of a site and one neighbour over the real
  neighbours only, so energies are exact on sites of any degree.
- Add a colored schedule for local rules and samplers, which updates the classes of a proper coloring of the
  neighbour graph in turn. It generalises the checkerboard schedule to triangular lattices and graphs.
- Add extremal updates: a law that finds the site with the smallest or largest value of a key state, applies a rule
  to that site and its neighbours, and records the site and the value as an event.
- Add a site visual that streams triangular lattices as hexagonal cells and graph lattices as colored nodes and edges
  to Rerun.
- Add the Ising model on a graph, the forest fire on triangular lattices and graphs, the FHP lattice gas, and
  Bak-Sneppen to the model library.

## Capabilities

### New Capabilities

- `lattice-geometries`: triangular lattices, lattice laws on graphs, graph neighbourhoods and neighbour sums, the
  colored schedule, extremal updates, and drawing non-square lattices.

### Modified Capabilities

- `model-library`: the models this change unblocks.

## Impact

- `skcomplex.core.model` gains `Adjacency`, `Extremal`, `adjacent`, and `neighbor_sum`, the `geometry` parameter and
  the `positions` property of `LatticeSpace`, `Neighborhood.triangular`, the `index_table` method of `NetworkSpace`,
  and the `'colored'` schedule.
- `skcomplex.core.compile` lowers the new laws to NumPy and Numba, and moves the padded graph neighbour table of
  fields into `NetworkSpace.index_table`, so fields and lattice laws share one table.
- `skcomplex.core.viz` gains `SiteVisual`, and the trajectories gain extremal records next to relaxation records.
- `skcomplex.models` gains models under `physics/statistical_mechanics`, `physics/criticality`, `physics/fluids`, and
  `biology/evolution`.
- No new dependency. NetworkX, already a runtime dependency, provides the graph generators and the graph coloring.
