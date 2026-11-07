# Proposal

## Why

The catalog samples lattice gauge fields only with angle-valued link states, U(1) and Z_n, whose products around a
plaquette commute and reduce to sums. The non-abelian gauge theories of the weak and strong forces, SU(2) and SU(3) as
in Satz 2022 chapter 7 (lattice QCD) and Schwichtenberg 4.4, need link states valued in a compact matrix group, ordered
products around plaquettes and other closed paths, and Monte Carlo moves that keep the links in the group.

## What Changes

- A matrix group spec in the core names `SU(n)` or `U(n)` and converts matrices to and from their real components,
  draws Haar matrices and matrices near the identity, projects rounded matrices back onto the group, and gives the
  path-ordered product of link matrices along paths of oriented edges, the holonomy.
- A state may be declared with a matrix group, which stores the real components of every matrix of a group member.
- Cell complexes list the oriented edges around every face in cyclic order, the path of its holonomy.
- Drives change a state of several components per member as a whole, with the components on a last axis, and receive
  the space of the group when they name it. A drive of a group on the edges of a network reads one row per member.
- `SamplingInteraction` takes a holonomy score, a class function of the holonomy of every face, and samples a matrix
  group state on the edges of a cell complex with Metropolis or Glauber moves by matrices near the identity or Haar
  draws, a colored schedule, and a projection onto the group after every sweep.
- The physics package of waves gains SU(2) and SU(3) link elements, the Wilson action of matrix links in
  `WilsonSamplingInteraction`, Wilson and Polyakov loops of matrix links in `GaugeAnalysis`, and the catalog system
  `lattice_yang_mills` with the mean plaquette and the Polyakov loop as observables, an inventory entry, and a gallery
  example.

## Capabilities

### New Capabilities

### Modified Capabilities
- `link-fields`: link states valued in a compact matrix group, holonomies along face boundaries and other paths, and
  Monte Carlo sampling of matrix links.
- `model-library`: the lattice Yang-Mills system for SU(2) and SU(3) and its gauge observables.

## Impact

- Core: `skcomplex.core.specs` (new `MatrixGroupSpec` and `HolonomyScoreSpec`, `StateSpec.group`, `VariableSpec.group`),
  `CellComplexSpace` (`boundaries_`, `boundary_signs_`), `SamplingInteraction`, `DriveLaw`, the drive kernel, the
  lattice state spec, and the layout and lattice compile passes. No new package or dependency, SciPy draws the Haar
  matrices.
- Systems: `skcomplex.systems.physics.waves`.
- Docs: the cell complexes and link fields guide, the inventory, and a gallery example.
