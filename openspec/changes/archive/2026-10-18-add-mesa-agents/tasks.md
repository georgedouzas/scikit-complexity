# Tasks

## 1. Setup

- [x] 1.1 Add `mesa>=3.5` and `networkx` as dependencies, relock, and verify `import mesa` succeeds in a fresh
  environment

## 2. Agent description

- [x] 2.1 Add `Rule`, `AgentType`, `Form.AGENT`, and form inference for rules, and verify the inference and mixing tests
  pass
- [x] 2.2 Add `GridSpace`, `NetworkSpace`, and `ContinuousSpace`, and verify the grid neighbour test passes

## 3. Mesa backend

- [x] 3.1 Generate Mesa agent classes from agent types and build the Mesa model with spaces and seed, and verify the
  reproducibility test passes
- [x] 3.2 Add agent snapshots to `Frame` and `Trajectory`, with creation and removal through rules, and verify the
  population growth and per-agent trajectory tests pass
- [x] 3.3 Add `MesaSystem` with reporters and live parameters, and verify the wrapped Boltzmann wealth tests pass

## 4. Visualization

- [x] 4.1 Add agent visuals for grids, continuous spaces, and networks with agent ids, and verify the Schelling,
  voter, and removed-agent tests pass

## 5. Model library and documentation

- [x] 5.1 Add the `social` module and the wolf and sheep model, and verify their spec scenario tests pass
- [x] 5.2 Document agent rules, spaces, and wrapped Mesa models in the user guide, update the Project Profile, and
  verify the documentation test passes
- [x] 5.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass

## Workflow follow-up

- Archive the change after review.
