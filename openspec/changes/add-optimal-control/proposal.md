# Proposal

## Why

Several references ask how to steer a system, not only how it evolves:
- **Optimal control:** Dilão (2023 ch. 9) derives optimal controls with Pontryagin's maximum principle, including for a
  macroeconomic model.
- **Feedback control:** Ryzhov et al. (2021) close loops with PID controllers.
- **Controllability:** Forrest (2018 ch. 6) and Mobus (2022 ch. 12) treat controllability and governance as properties
  of input-output systems.
- **Policy design:** Bala et al. (2017 §10.6) design policies for system-dynamics models, and Meadows (2008 ch. 6) ranks
  leverage points.

The framework has no notion of a control input, a cost, or a controller.

## What Changes

- Add control inputs: named time-dependent inputs with bounds that laws use like parameters.
- Add optimal control problems: a cost made of a running cost and a terminal cost, with bounds on controls and states.
  Solvers are Pontryagin's principle with symbolic costates and a forward-backward sweep, and direct transcription by
  collocation with a nonlinear programming solver.
- Add feedback controllers: proportional-integral-derivative controllers and state-feedback gains as components that
  read observables and drive inputs during a run.
- Add linear analysis: linearisation to state-space matrices `A, B, C, D` at an operating point, controllability and
  observability ranks, and linear-quadratic regulator gains.
- Add library models:
  - optimal harvesting of a logistic resource;
  - vaccination in an SIR epidemic;
  - Ramsey optimal growth;
  - a PID-stabilised inverted pendulum on a cart;
  - the linear-quadratic regulated oscillator.

## Capabilities

### New Capabilities

- `control`: control inputs, optimal control, feedback controllers, and linear analysis.

### Modified Capabilities

- `model-library`: control models.

## Impact

- `skcomplex.core.model` gains `Control` and `Controller` with `PID` and `StateFeedback`, and a new package
  `skcomplex.control` gains `OptimalControlProblem`, `pontryagin`, `collocation`, `linearize`, `controllability`,
  `observability`, and `lqr`.
- Depends on `add-general-model-interface` for observables, `add-hybrid-dynamics` for time-dependent inputs, and
  `add-analysis-toolkit` for Jacobians.
