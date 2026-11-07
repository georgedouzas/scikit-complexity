# Tasks

## 1. Topology requests

- [x] 1.1 Add the kinds `'close-triangle'`, `'contract-edge'`, `'duplicate-node'`, and `'replace-node'` and the fields
  `candidates`, `score`, `one`, `inherit`, and `parent` to `TopologyRequestSpec` with their validation, verified by
  tests of `TopologyRequestSpec`.
- [x] 1.2 Compile the score like the weights and check the rows of the new kinds, `'add-edge'` without a target, and
  jumps with `one`, verified by tests of the passes through `Simulation.prepare`.
- [x] 1.3 Apply the new kinds, the choice among candidates, and `one` in `TopologyKernel`, verified by tests of the
  kernel through simulations.
- [x] 1.4 Add the actions and options to `TopologyInteraction` and fire `'replace-node'` as one request, verified by
  its tests and the tests of the systems that replace nodes.

## 2. Triadic regulation

- [x] 2.1 Read several signed regulators per link in the interaction of `spatial_triadic_percolation`, verified by a
  test of the regulation on both backends.

## 3. Probe systems

- [x] 3.1 Add `explosive_percolation` with tests of the abrupt jump and of the Erdős-Rényi case.
- [x] 3.2 Add `davidsen_ebel_bornholdt` with tests of its mean degree and clustering.
- [x] 3.3 Add `node_merging` with a test of its hubs.
- [x] 3.4 Add `duplication_deletion` with a test of the connectance of Eq. 8.84.
- [x] 3.5 Add `spatial_triadic_percolation` with tests of its blinking and its spatial patterns.
- [x] 3.6 Add `network_superposition` with a test of its degrees against Eq. 4.34.

- [x] 3.7 Keep every new system self-contained in its own module, its tests mirroring it.

## 4. Documentation and checks

- [x] 4.1 Record every new system in `docs/overview/inventory.toml` and add its gallery example, verified by the
  inventory check and `gallery_tools coverage --strict`.
- [x] 4.2 Describe the new requests in the networks guide, verified by its doctests.
- [x] 4.3 Run ruff, black, docformatter, mypy, and pydoclint on the changed files, the tests of the touched modules,
  the catalog tests, and the doctests of changed modules.
