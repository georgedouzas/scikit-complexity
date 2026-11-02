# Tasks

## 1. Triangular lattices

- [x] 1.1 Add the `geometry` parameter and the `positions` property to `LatticeSpace`, and `Neighborhood.triangular`,
  with docstring examples, and verify the six neighbours on a torus and the opposite directions scenarios
- [x] 1.2 Add tests of a triangular local rule on the NumPy and Numba backends, and verify the hexagonal spread
  scenario on both

## 2. Lattice laws on graphs

- [x] 2.1 Add `Adjacency` and `NetworkSpace.index_table`, built from the NetworkX adjacency in CSR form, move the padded
  table of `compile/_fields.py` onto it, and verify the existing graph field tests still pass
- [x] 2.2 Accept lattice groups on a `NetworkSpace` in the compile tree, with initial values in node order or by node,
  and reject a capacity, and verify the grid graph and the sites of different degree scenarios
- [x] 2.3 Pass `degree` to function rules that ask for it, and verify a rule's degrees on a star graph
- [x] 2.4 Add `adjacent` and `neighbor_sum` and lower them with the padding mask on NumPy and Numba, and verify the
  neighbour sum and the Ising on a graph scenarios

## 3. Colored schedule

- [x] 3.1 Add the `'colored'` schedule to `LocalRule` and `Sampler`, with the color classes from `nx.greedy_color`
  cached per table, and verify the coloring scenario
- [x] 3.2 Point the checkerboard error at the colored schedule for triangular lattices and graphs, and verify the
  rejected checkerboard scenario

## 4. Extremal updates

- [x] 4.1 Add `Extremal` and `ExtremalRecord`, compile them to the lattice steps, and add the records to trajectories
  and frames, and verify the only-the-neighbourhood scenario
- [x] 4.2 Run repeated extremal applications in one Numba loop on the shared generator, and verify the reproducible
  extremal runs scenario on both backends

## 5. Visualization

- [x] 5.1 Add `SiteVisual`, register it as `'sites'`, and make it the default of non-square lattice groups, and verify
  the triangular forest fire scenario on a recording saved to a file

## 6. Library

- [x] 6.1 Add the Ising model on a graph under `physics/statistical_mechanics`, and verify its ordering scenario and
  the conformance suite
- [x] 6.2 Add the forest fire on triangular lattices and graphs under `physics/criticality`, and verify the forest fire
  on a graph scenario
- [x] 6.3 Add the FHP lattice gas under `physics/fluids`, and verify the conservation scenario
- [x] 6.4 Add Bak-Sneppen with the extremal update under `biology/evolution`, and verify the self-organisation scenario
  in the benchmark suite
- [x] 6.5 Mark the four models as implemented in the inventory, and verify the inventory check passes

## 7. Documentation

- [x] 7.1 Document triangular lattices, graph lattices, neighbour sums, the colored schedule, and extremal updates in
  the lattice section of the user guide, with runnable examples, add a changelog entry, and verify the documentation
  build passes

## 8. Gate

- [x] 8.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
