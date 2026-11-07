# Design

## Context

Link fields today are groups on the edges of a `CellComplexSpace` with scalar states. `SamplingInteraction` samples
them through a symbolic objective over `CofaceNeighborhood`, where `exterior_derivative` is the oriented sum of angles
around a face, compiled into the Monte Carlo kernel. A sum cannot express an ordered product of matrices, and the
symbolic compiler handles scalar expressions. A second path already exists for objectives that are not symbolic: a
`RowScoreSpec` objective makes `SamplingInteraction` return a `DriveLaw`, a NumPy function of the whole state array.

## Goals / Non-Goals

**Goals:**
- Matrix group states, holonomies, and path-ordered products named after the mathematics, in existing core packages.
- Sampling that keeps links in the group, vectorised over the links that share no face.
- SU(2) and SU(3) lattice gauge theory as a physics realization in `systems/physics/waves`.

**Non-Goals:**
- Heat-bath and overrelaxation updates (Creutz, Kennedy-Pendleton, Cabibbo-Marinari), fermions, and gauge fixing.
- Compiled (Numba) matrix kernels, or matrix states in symbolic laws.

## Decisions

- **Matrix group spec in `core/specs`.** `MatrixGroupSpec(name, degree)` is a frozen spec like `TensorSpec`, holding
  the conversions between matrices and real components, Haar draws (SciPy's `unitary_group`, an actively maintained
  dependency already in use, divided by a root of the determinant for `SU(n)`), moves `exp(i width H)` with random
  Hermitian `H`, the polar projection, and `holonomy_of`. No mature library samples Lie groups with an active release
  cadence that adds more than these few NumPy lines, so NumPy and SciPy suffice.
- **`StateSpec(group=...)`** stores `2 n^2` real components per member as the member shape, the way a tensor state
  stores its independent components. Laws read it as a vector, and the layout pass sizes it from the group.
  Alternative: one scalar state per component, which would need 18 states for SU(3) and a law changing many states.
- **Face boundaries on `CellComplexSpace`.** `boundaries_` and `boundary_signs_` give the ordered, signed edges of every
  face. A lattice takes them from its plaquette construction, since on a periodic axis of two sites two links join the
  same nodes, and a network from its node cycles. Any other closed path is a row of edges and signs for `holonomy_of`.
- **Sampling through a drive.** A `HolonomyScoreSpec(function, group, hits)` objective makes `SamplingInteraction`
  return a `DriveLaw`, as a `RowScoreSpec` does. The drive receives the components with a last axis and the space by
  the keyword `space`, a small extension of `DriveLaw` and the drive kernel, which also fixes the shape of drives on the
  edges of a network. Only drives may target states of several components. The local objective of an edge sums a class
  function over the holonomies of its faces started at the edge, `U S_p` or `U^-1 S_p` with the staple `S_p` of the
  other edges, so the staples of a class of edges that share no face are computed once per class. The classes are a
  greedy coloring of the graph of edges that share a face, cached per complex.
- **Physics realization.** `IsospinLinkElement` and `ColorLinkElement` hold SU(2) and SU(3) matrices,
  `WilsonSamplingInteraction` takes a `group` and `hits` for the action `-(beta / N) Re tr U_p`, `GaugeAnalysis` takes a
  `group` for Wilson loops and the Polyakov loop, and `lattice_yang_mills` builds the system.

## Risks / Trade-offs

- [Python loop over color classes and hits] → vectorised NumPy per class keeps a sweep of a 6^4 SU(2) lattice near
  0.1 s, enough for the catalog and the gallery.
- [Metropolis decorrelates slowly near the continuum] → `hits` and `width` are tunable, and heat bath can be added later
  as another move of the same objective.
- [Observables re-prepare the complex for the Polyakov loop each frame] → a cell complex of a 6^4 lattice prepares in
  about 10 ms, small next to a sweep.
