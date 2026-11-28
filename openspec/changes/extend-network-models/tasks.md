# Tasks

## 1. Start

- [ ] 1.1 Check that `fix-network-model-bugs` is applied and merged into `development`, so its `_ensembles.py` and
  `_growth.py` are the base, and verify with `git log development -- src/skcomplex/systems/mathematics/networks`

## 2. Agent initial states (#72, #73)

- [ ] 2.1 Let `PassCompiler._copy_with_initial_state` set the states and position states of agent groups by dotted
  name in `_CompiledAgentsSpec.values`, with the length check and an error that lists the agent state names, and
  verify with tests in `tests/core/engine/compile/test_pass_compiler.py`
- [ ] 2.2 Check that the Mesa solver creates and places the agents from the replaced values, with a test of a group on
  a network and one on a grid in `tests/core/engine/solvers/test_mesa.py`
- [ ] 2.3 Document agent initial states in the docstring of `BaseSystem.simulate` and in
  `docs/overview/user_guide/simulate.md`, and verify that the examples run under the doctest and markdown test sessions

## 3. Random graph ensembles (#66, #70)

- [ ] 3.1 Add `directed=True` to the Erdős-Rényi ensemble of `RandomGraphEnsemblesSystem`, with the step check
  `steps >= ceil(mean_degree)` and the error for other ensembles, tested in
  `tests/systems/mathematics/networks/test_ensembles.py`
- [ ] 3.2 Add `ensemble='newman_watts'`, with the `added` mark on the lattice links, and test that the ring stays and
  the shortcut count matches `rewiring` in `tests/systems/mathematics/networks/test_ensembles.py`
- [ ] 3.3 Add `directed=True` to `ConfigurationModelSystem`, with given and drawn in and out degrees and the directed
  rewiring, tested for kept degrees, simplicity and the sum error in
  `tests/systems/mathematics/networks/test_ensembles.py`
- [ ] 3.4 Update the docstrings and the gallery example `docs/examples/mathematics/networks/plot_random_graph_ensembles.py`
  to show the new ensemble and the directed option, and verify that the example runs

## 4. Growth variants (#70)

- [ ] 4.1 Add `offset` to `BarabasiAlbertSystem`, with and without `fitness`, and its check, and test the exponent
  `3 + C / m` and the error in `tests/systems/mathematics/networks/test_growth.py`
- [ ] 4.2 Add `attachment='uniform'` and its check, and test the exponential degree tail in
  `tests/systems/mathematics/networks/test_growth.py`

## 5. Small fixes (#70)

- [ ] 5.1 Return the summary `components` as one integer array per frame, and update the docstring of
  `NetworkMeasuresResult`, tested in `tests/core/analysis/test_networks.py`
- [ ] 5.2 Say in the docstring of `NetworkSuperpositionSystem` that the subgraphs are disjoint, and verify the doctest

## 6. Individual epidemics on a given graph (#71)

- [ ] 6.1 Add `ContactHostElement` and `ContactNetworkEpidemicSystem` with `compartments='sir'` and `'sis'` to
  `systems/biology/epidemiology/_adaptive.py`, re-export them, and register the system with its field, topics and
  references
- [ ] 6.2 Test the start from chosen nodes, the SIR final size and the endemic SIS state in
  `tests/systems/biology/epidemiology/test_adaptive.py`
- [ ] 6.3 Add a gallery example `docs/examples/biology/epidemiology/plot_contact_network_epidemic.py` and verify that it
  runs and that the inventory check of the documentation build passes

## 7. Walks on networks (#72)

- [ ] 7.1 Add `start` and `random_state` to `NetworkRandomWalkSystem` and record `walkers.node`, tested in
  `tests/systems/mathematics/networks/test_processes.py`
- [ ] 7.2 Add `mode='master'` on `NodeRankElement` and `PagerankInteraction(damping=1.0)`, with its errors, and test
  the stationary law of the karate club in `tests/systems/mathematics/networks/test_processes.py`
- [ ] 7.3 Add `kind='mean-first-passage-matrix'` and `kind='mean-return-times'` to `passage`, with graphs as sources,
  and test the ring of five nodes, the karate club and the reducible error in `tests/core/analysis/test_passage.py`
- [ ] 7.4 Show the start node, the master equation and the return times in
  `docs/examples/mathematics/networks/plot_network_random_walk.py`, and verify that it runs

## 8. Voters (#73)

- [ ] 8.1 Add `schedule='single'` to `VoterImitationInteraction` and `VoterSystem`, with its check, and document the
  initial opinions in the docstring of `VoterSystem`
- [ ] 8.2 Test the initial opinions and the star martingale in `tests/systems/social/opinion_dynamics/test_voters.py`

## 9. Cascades and outputs (#74)

- [ ] 9.1 Add `update='synchronous'` to `ThresholdActivationInteraction` and `WattsCascadesSystem` through
  `_update_agents`, document both orders, and test the counts 1, 3, 5, 7 on a cycle in
  `tests/systems/social/spreading/test_cascades.py`
- [ ] 9.2 Set `liabilities_` and `capital_` in `DebtrankSystem._prepare`, and test that they reproduce the cascade in
  `tests/systems/economics/finance/test_contagion.py`
- [ ] 9.3 Document the frames of `NetworkDiffusionSystem` and `nodes.giant` of `MotterLaiOverloadSystem`, and verify the
  doctests of both modules

## 10. Checks

- [ ] 10.1 Run `black` and `ruff check` on the changed files and fix every finding
- [ ] 10.2 Run `mypy`, `pydoclint` and `interrogate` on the changed packages and fix every finding
- [ ] 10.3 Run `lint-imports` and `python openspec/scripts/check_conventions.py` and fix every finding
- [ ] 10.4 Run the tests of the touched modules with the thread variables set to 1: `tests/core/engine/compile`,
  `tests/core/engine/solvers/test_mesa.py`, `tests/core/analysis/test_networks.py`,
  `tests/core/analysis/test_passage.py`, `tests/systems/mathematics/networks`,
  `tests/systems/biology/epidemiology/test_adaptive.py`, `tests/systems/social/opinion_dynamics/test_voters.py`,
  `tests/systems/social/spreading/test_cascades.py` and `tests/systems/economics/finance/test_contagion.py`
