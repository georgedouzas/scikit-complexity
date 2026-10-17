# Proposal

## Why

Agent-based models are the last major class of complex systems the framework cannot describe: agents follow rules
written as code, live on grids, networks, or continuous spaces, and are born and die during a run. Mesa 3.5 is the
standard Python library for them, with agent sets, discrete and continuous spaces, an event scheduler, and a large
catalogue of published models. The framework should run agent rules on Mesa, run existing Mesa models unchanged, and
give both its real-time runtime, live parameters, trajectories, and Rerun visualization.

## What Changes

- Add a `Rule` law kind, a Python function that updates one agent with access to its neighbours and the model, and an
  agent-based form that the compiler infers from it.
- Add agent types and agent groups whose members can be created and removed during a run.
- Add grid, network, and continuous spaces for agents, backed by Mesa's discrete and continuous spaces.
- Add a Mesa backend that builds a Mesa model from a system with rules and advances it frame by frame.
- Add `MesaSystem`, which wraps any existing `mesa.Model` with declared reporters and parameters, so it runs under the
  framework's runtime and visualization.
- Add Rerun visuals for agents on grids, networks, and continuous spaces.
- Add a `social` model module with Schelling segregation, Boltzmann wealth, and the voter model, and a wolf-sheep
  predation model in the ecology module.
- **BREAKING** Make `mesa` and `networkx` runtime dependencies.

## Capabilities

### New Capabilities

- `agent-based-modeling`: agent rules, agent groups of changing size, agent spaces, the Mesa backend, and wrapped Mesa
  models.

### Modified Capabilities

- `rerun-visualization`: visuals for agents on grids, networks, and continuous spaces.
- `model-library`: social models and an agent-based predator and prey model.

## Impact

- `skcomplex.model` gains `Rule`, agent spaces, and `AgentType`. `skcomplex.backends` gains `mesa`. `Frame` and
  `Trajectory` gain agent snapshots, whose size can change between frames.
- `pyproject.toml` gains `mesa` and `networkx`. Mesa 3.5 imports `networkx` at top level without declaring it, so both
  are declared.
- Depends on `add-element-groups` and on the backend contract of `add-lammps-backend`.
