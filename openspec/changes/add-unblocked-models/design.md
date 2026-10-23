# Design

## Context

The inventory marked these four models blocked on `add-adaptive-networks` and `add-data-driven-dynamics`. Both changes
are archived: neighbourhood reductions inside any law, network generators, and edge groups exist, as do surrogate
systems fitted by mode decomposition, sparse regression, and reservoirs, and optimal-control objectives and solvers.

## Goals / Non-Goals

**Goals:**
- Build the four models from existing primitives, with tests of their known behaviour.

**Non-Goals:**
- New core primitives. A model that turns out to need one stops, and its need becomes a change of its own.

## Decisions

### Systems

- `szabo_cells`: self-propelled cells in a periodic continuous space whose velocity adds the sum of pair forces over
  neighbours within a radius and aligns its polarity with its velocity. The inventory tags it with a network and an
  edge group; the tags follow the system the model builds.
- `active_passive_flocking`: Vicsek movers of which a fraction aligns and the rest only follows the flow, with
  neighbourhood reductions over a radius; the tags likewise follow the built system.
- `koopman_model_predictive_control`: a surrogate fitted by extended dynamic mode decomposition with control to
  trajectories of the Duffing oscillator, then used as the model of a receding-horizon controller.
- `next_generation_reservoir`: a surrogate fitted by ridge regression on polynomial features of delay-embedded series
  of a chaotic flow, forecasting it beyond the training window.

## Risks / Trade-offs

- A surrogate of a controlled system may need inputs that the fitting functions lack. If so, the model stops and the
  need becomes a change of its own.
