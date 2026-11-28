# Proposal

## Why

The network chapters of the Complexity Science tutorials filed issues #66, #70, #71, #72, #73 and #74. They rebuild
with NetworkX and NumPy what the library should hold: directed random graphs, two classic growth variants, spreading
on a given graph with one state per node, walkers and voters that start where the user puts them, and the outputs of
the cascade systems. The bugs of the same models are planned apart in `fix-network-model-bugs`, so this change only
adds what is missing.

## What Changes

- **Directed random graphs (#66):** `RandomGraphEnsemblesSystem(ensemble='erdos_renyi', directed=True)` grows a
  directed Erdős-Rényi graph, and `ConfigurationModelSystem(directed=True)` pairs out-stubs with in-stubs from a pair
  of in and out degree sequences or from drawn ones. The degree measures of directed graphs asked in #66 are not part
  of this change.
- **Newman-Watts shortcuts (#70):** `RandomGraphEnsemblesSystem(ensemble='newman_watts')` keeps the ring and adds a
  shortcut from a lattice edge with total probability `rewiring`, as `nx.newman_watts_strogatz_graph` does.
- **Attachment variants (#70):** `BarabasiAlbertSystem(offset=C)` attaches with probability proportional to
  `k + C`, and `BarabasiAlbertSystem(attachment='uniform')` attaches to nodes drawn uniformly.
- **Small fixes (#70):** the `components` of a `network_measures` summary become one integer array per frame, like
  `degrees`, and the docstring of `NetworkSuperpositionSystem` says that the subgraphs are disjoint.
- **Spreading on a given graph (#71):** a new `ContactNetworkEpidemicSystem(graph, compartments='sir')` runs the
  individual SIR or SIS epidemic on any graph, with the state `hosts.state` of every node, so
  `initial_state={'hosts.state': x0}` starts it from chosen infected nodes.
- **Agent initial states (#72, #73):** `simulate(initial_state=...)` accepts the states of agent groups, such as
  `voters.opinion` and the node `walkers.node` of a walker, one value per agent.
- **Walks on networks (#72):** `NetworkRandomWalkSystem(start=...)` starts the walkers at chosen nodes, and
  `NetworkRandomWalkSystem(mode='master')` evolves the probability `nodes.rank` of a walker at every node by `p(t + 1) = p(t) P`.
  `passage(graph_or_matrix, kind='mean-first-passage-matrix')` returns the mean first-passage time between every pair
  of nodes, and `kind='mean-return-times'` the mean return time of every node, `2 L / k_i` on a connected graph.
  Katz centrality from #72 is not part of this change.
- **Voters (#73):** `VoterSystem(schedule='single')` updates one voter drawn uniformly per step, beside the default
  random sequential sweep, and the opinions are states that `initial_state` sets.
- **Cascades and outputs (#74):** the docstrings of `NetworkDiffusionSystem`, `WattsCascadesSystem` and
  `MotterLaiOverloadSystem` say what a frame, an update and `nodes.giant` are. `WattsCascadesSystem(update='synchronous')`
  activates every node from the activations of the previous step. `DebtrankSystem` exposes the liabilities and
  capitals it uses as `liabilities_` and `capital_` after `prepare`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-catalog`: the new options of the network ensembles, growth, walks, voters, cascades and DebtRank, and the
  individual epidemic on a given graph.
- `network-analysis`: the summary components as arrays.
- `simulation`: initial states of agent groups.
- `stochastic-analysis`: mean first-passage matrices and mean return times of a transition matrix or a graph.

## Impact

- Systems: `systems/mathematics/networks/_ensembles.py`, `_growth.py`, `_processes.py`, `_superposition.py` and
  `_overload.py`, `systems/biology/epidemiology/_adaptive.py`, `systems/social/opinion_dynamics/_voters.py`,
  `systems/social/spreading/_cascades.py` and `systems/economics/finance/_contagion.py`, with the `__init__` modules
  that re-export their new names.
- Engine: `core/engine/compile/_pass_compiler.py`, which applies initial states, and the agent records of
  `core/engine/problems/_compiled_agents_spec.py` that `core/engine/solvers/_mesa.py` reads.
- Analysis: `core/analysis/_networks.py` and `core/analysis/_passage.py`.
- Tests under the mirrored paths of `tests/`, and the gallery examples of the network walk and of the systems whose
  docstrings change.
- Issues #70, #71, #73 and #74 close with the change. #66 and #72 stay open for the degree measures and Katz
  centrality.
