# Tasks

## 1. Hyperedges during a run

- [x] 1.1 Confirm the XGI 1.0 dependency and lock, and run the tests of hypergraphs and higher-order interactions
- [x] 1.2 Add `rank` to `GroupElement` and pad the members of ranked hyperedge groups in the layout pass, verified by
  the rank tests in `tests/core/components/elements/test_group.py`
- [x] 1.3 Add `'add-hyperedge'`, `'remove-hyperedge'`, `independent`, and `direction` to `TopologyRequestSpec`,
  `TopologyInteraction`, the topology kernel, and the topology function pass, verified by the new tests in
  `tests/core/components/interactions/test_topology.py`
- [x] 1.4 Refuse laws per hyperedge on a ranked group, verified by the rank test in `test_topology.py`

## 2. Shortest paths

- [x] 2.1 Add `shortest_paths` to the law utilities and lower it as a neighbourhood reduction of the kind `'path-'`
- [x] 2.2 Compute it with NetworkX in `NeighborhoodKernel`, cached on the active links, and fall back to Python on Numba,
  verified by `tests/core/components/laws/utils/test_paths.py`

## 3. Member sets and autocatalytic sets

- [x] 3.1 Add `group` to `AgentView.neighbors`, verified by the member set test in `tests/core/engine/solvers/test_mesa.py`
- [x] 3.2 Add `AutocatalyticSetAnalysis`, verified by `tests/core/analysis/test_autocatalytic.py` and its doctests

## 4. Systems

- [x] 4.1 Add `UrnElement`, `BallElement`, and `interacting_urns_network`, verified by
  `tests/systems/economics/innovation/test_discovery.py`
- [x] 4.2 Add `tap_catalysis`, verified by `tests/systems/economics/innovation/test_catalysis.py`
- [x] 4.3 Add `LoadedNodeElement` and `motter_lai_overload`, verified by
  `tests/systems/mathematics/networks/test_overload.py`
- [x] 4.4 Add the inventory entries and primitives, verified by `inventory_problems()` returning no problem
- [x] 4.5 Add the gallery examples, verified by `gallery_tools coverage --strict` and a run of every example

## 5. Documentation and checks

- [x] 5.1 Document the capabilities in the networks and agent-based guides, verified by running their code blocks
- [x] 5.2 Run ruff, black, docformatter, mypy, and pydoclint on the changed files, the catalog tests, and the tests of
  the touched modules
