# Tasks

## 1. Networks

- [ ] 1.1 Add array-based network spaces with layers and the generators, and verify the same-seed scenario
- [ ] 1.2 Add `Edges` with edge laws, and verify the Hebbian scenario and the two-relations scenario

## 2. Reductions

- [ ] 2.1 Add `neighbors` with network, radius, nearest, and cone neighbourhoods on NumPy and Numba, and verify the
  Kuramoto and flocking scenarios

## 3. Topology

- [ ] 3.1 Add capacity layouts with active masks for variable groups, and verify existing group tests pass unchanged
- [ ] 3.2 Add the topology laws, and verify the preferential attachment and adaptive epidemic scenarios

## 4. Recording and library

- [ ] 4.1 Record network snapshots and log them to Rerun as graphs, and verify the snapshot scenario
- [ ] 4.2 Add the network models as registered factories, and verify the Boolean network scenario and the conformance
  suite

## 5. Documentation and gate

- [ ] 5.1 Document networks, edges, layers, reductions, and topology laws in the user guide, and verify the
  documentation test passes
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
