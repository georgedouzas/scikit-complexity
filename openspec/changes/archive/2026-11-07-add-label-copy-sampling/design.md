# Design

## Context

`SamplingInteraction` has three objective paths: a symbolic local objective compiled into the Monte Carlo kernel, a
`RowScoreSpec` and a `HolonomyScoreSpec`, both of which make it return a `DriveLaw`, a NumPy function of the whole state
array that receives the random generator, the time, the parameters, and the space. The cellular Potts energy cannot be
a local objective of one site, since the area term of a cell depends on all its sites.

## Goals / Non-Goals

**Goals:**
- A label copy move and a score of per-label totals, named after the mathematics (labels, regions, contacts, volume
  and surface constraints), in existing core packages.
- Totals kept per label and updated per accepted move, not recomputed.
- Regular lattices and networks with weighted contacts and site volumes, so a disordered lattice is a graph.
- The cellular Potts model of cell sorting as a biology realization.

**Non-Goals:**
- Connectivity constraints of cells, chemotaxis or other site fields, cell growth and division, and a medium with its
  own boundary conditions.
- General nonlinear functions of the totals: the constraints are elastic, `a (V - A)^2`, the form of the cellular Potts
  literature, which lets the whole sweep run in Numba.

## Decisions

- **`RegionScoreSpec` in `core/specs`.** `contact(first, second, **parameters)` is evaluated once per sweep on the grid
  of all labels into a table. `volume` and `surface` return the strength and the target of the constraint of every
  label, also evaluated once per sweep. Alternative: arbitrary Python functions of the totals per move, which would
  call Python once per attempt and run two orders of magnitude slower.
- **Sampling through a drive.** As for the other scores, a region score makes `SamplingInteraction` return a
  `DriveLaw`. The drive reads the neighbour table of the space (the given neighbourhood, or the default of the space),
  compacted with the other sites first and padded with the site itself, the contact weights and the site volumes,
  cached per space and neighbourhood. At the start of a sweep it computes the volume and surface of every label with
  `bincount`, draws the sites, the neighbour picks, and the acceptance numbers from the generator of the simulation,
  and runs a Numba kernel that proposes the copies in sequence. A copy from label `a` to `b` at site `i` changes only
  the totals of `a` and `b`, by the volume of the site and by `±w` over the contacts of `i`, so the change of the
  energy and the update of the totals cost the size of the neighbourhood.
- **Weights from the graph.** On a `NetworkSpace` the edge attribute `'weight'` is the contact weight `w_ij` and the
  node attribute `'volume'` the site volume, both 1 by default, the NetworkX convention for edges. A lattice has unit
  weights and volumes.
- **Move `'copy'`.** It is added to the moves of `SamplingInteraction` and accepted only with a region score, which in
  turn accepts only `'copy'` and the Metropolis or Glauber rule. The schedule does not apply.
- **Libraries.** CompuCell3D and Morpheus are full simulation environments with their own runtimes, not libraries that
  sample a NumPy state, and no maintained Python package offers the move alone, so NumPy and Numba (already runtime
  dependencies) suffice. The disordered lattice of the system uses SciPy's `Voronoi` on a 3 by 3 tiling of jittered
  points for periodicity.
- **Biology realization in `biology/physiology`.** No morphogenesis subfield exists, and physiology already holds
  tissue models. `TissueSiteElement` holds the label `cell`, `CellAdhesionInteraction` builds the region score from an
  adhesion matrix between kinds, the kind of every cell, and area and perimeter constraints, with the temperature as
  one over the intensity, and `cellular_potts` builds rectangles of `cell_shape` sites with kinds at random.

## Risks / Trade-offs

- [The contact table is quadratic in the number of labels] → a thousand cells take a million entries, evaluated once
  per sweep, small next to a sweep of the corresponding lattice.
- [Sorting is slow in the cellular Potts model] → a 40 by 40 tissue runs 3000 sweeps in about a second, enough for the
  gallery and for tests with 1500 sweeps.
- [Cells can fragment or vanish] → the area constraint keeps them in practice, and connectivity checks are a non-goal.
