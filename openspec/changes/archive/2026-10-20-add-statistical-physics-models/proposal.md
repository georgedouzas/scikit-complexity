# Proposal

## Why

Bertin 2021, *Statistical Physics of Complex Systems: A Concise Introduction*, was added to the survey. It poses models
across statistical physics, stochastic processes, active matter, traffic, economics, social dynamics, evolution,
epidemics, neural networks, and nonlinear dynamics. Its survey found:
- 31 models the core expresses today;
- 5 models pending existing changes;
- 2 models that need the general capability of `add-static-models`;
- 30 models the library already has, which gain Bertin as a source.

The survey checked each model against the current core, including the group reductions, variadic roles, and member
offsets that landed recently.

## What Changes

- **Inventory:**
  - add the source key `Bt`;
  - record every Bertin model with its field path, sources, and status;
  - add Bertin sources to the existing models.
- **Pending changes:** list the newly pending models in their specs:
  - `levy_flight` and `trap_model` in `add-anomalous-processes`;
  - `random_walk_with_memory` and `diluted_hopfield` in `add-stochastic-maps`;
  - `random_graph_ensembles` in `add-adaptive-networks`.
- **Models:** implement the 31 models in six groups, each with a checkable scenario, a behavioural check per model, and
  a gallery example. The groups are:
  - mean-field and disordered spin systems;
  - driven and noisy particles;
  - birth-death and interacting particle processes;
  - active matter and traffic;
  - agents, evolution, and populations;
  - maps and coupled oscillators.

## Capabilities

### New Capabilities

None.

### Modified Capabilities
- `model-library`: the Bertin models and their inventory record.

## Impact

- **Code:** new modules in existing subfield packages of physics, mathematics, engineering, economics, social, and
  biology. The core does not change.
- **Docs:** 31 new gallery examples, and the inventory gains 38 entries.
- **Dependencies:** none.
