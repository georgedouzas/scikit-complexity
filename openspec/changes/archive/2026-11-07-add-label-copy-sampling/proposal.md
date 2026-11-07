# Proposal

## Why

The cellular Potts model of Graner and Glazier, run on regular and disordered lattices by Nemati and de Graaf (arXiv
2404.09055, §II.A and §II.B), moves the boundaries between regions of sites that share a label. Its Monte Carlo move
copies the label of a neighbour into a site, and its energy reads totals per label, the volume and the surface of every
region, together with contact energies between labels. `SamplingInteraction` samples local objectives of one site,
whole rows, and holonomies, but none of these reads a reduction grouped by label, and no move copies a neighbour's
label.

## What Changes

- A region score spec in the core, `RegionScoreSpec(contact, volume, surface, hits)`, names the objective of a state of
  integer labels: weighted contact energies across the boundaries between regions and elastic constraints on the
  volume and the surface of every region.
- `SamplingInteraction` takes the move `'copy'` with a region score, on a `LatticeSpace` or a `NetworkSpace`. Every
  sweep proposes label copies from random neighbours, keeps the volume and surface of every label as totals, and
  updates the two changed totals on every accepted copy, in a Numba kernel. A network weighs contacts by the edge
  attribute `'weight'` and site volumes by the node attribute `'volume'`.
- The physiology package of biology gains `TissueSiteElement`, `CellAdhesionInteraction`, and the catalog system
  `cellular_potts`, which sorts cells of two kinds by differential adhesion on square, triangular, and disordered
  (periodic Voronoi) lattices, with the share of homotypic contacts and the deviation of the areas as observables, an
  inventory entry, and a gallery example.

## Capabilities

### New Capabilities

### Modified Capabilities
- `lattice-dynamics`: Monte Carlo label copies scored by contacts between regions and constraints on their totals.
- `model-library`: the cellular Potts model of cell sorting.

## Impact

- Core: `skcomplex.core.specs` (new `RegionScoreSpec`) and `SamplingInteraction` (move `'copy'`, the region sampling
  path and its Numba kernel). No new package or dependency.
- Systems: `skcomplex.systems.biology.physiology`.
- Docs: the lattice systems guide, the inventory with the `label-copies` primitive, and a gallery example.
