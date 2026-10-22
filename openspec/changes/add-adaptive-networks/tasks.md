# Tasks

## 1. Networks

- [ ] 1.1 Add array-based network spaces with layers and the generators to `skcomplex.core.space`, and verify the
  same-seed scenario
- [ ] 1.2 Let interactions with the binding `'neighbours'` bind an edge group of the same network and layer, and verify
  the Hebbian scenario and the two-relations scenario

## 2. Reductions

- [x] 2.1 Generalise `neighbors` to network, radius, nearest, and cone neighbourhoods on NumPy and Numba, keeping the
  lattice case, and verify the Kuramoto and flocking scenarios and the existing lattice tests

## 3. Topology

- [ ] 3.1 Add capacity layouts with active masks for variable groups on the NumPy and Numba backends, and verify existing
  group tests pass unchanged
- [ ] 3.2 Add the topology requests, and verify the preferential attachment and adaptive epidemic scenarios

## 4. Declarations and validation

- [ ] 4.1 Declare the requirements of layers, the edge binding, the neighbourhoods, and the topology requests, and the
  support of the solvers and backends for variable and edge groups, and verify the declaration test passes
- [ ] 4.2 Add the validation issue `group-fixed` and the layer and neighbourhood checks, and verify the request on a
  fixed group, missing layer, and metric neighbourhood on a network scenarios

## 5. Recording and library

- [ ] 5.1 Record network snapshots and log them to Rerun as graphs, and verify the snapshot scenario
- [ ] 5.2 Add the network models as registered factories, and verify the Boolean network scenario and the conformance
  suite

## 6. Documentation and gate

- [ ] 6.1 Document generators, layers, interactions over edges, reductions, and topology requests in the user guide,
  and verify the documentation test passes
- [ ] 6.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
