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

The framework has no notion of a bounded control input, a cost over a horizon, or a controller.

**Rebased on `redesign-system-model`.** The redesign already provides two parts this change first planned to add:
- adaptive parameters, `parameter(adaptive=True)`, which laws may change and which trajectories record, so a control
  input is an adaptive parameter rather than a new `Control` class;
- the `Objective` law kind of the constraint class, valid under continuous time for control over a horizon, so the
  running and terminal costs are `Objective` laws rather than arguments of a separate problem object.

This change still adds bounds on adaptive parameters, the continuous-time lowering of objectives, the Pontryagin and
collocation solvers, linear analysis, and the control models. PID and state-feedback controllers are control
realizations, so they move to `skcomplex.models.engineering.realizations` as elements built on rate and equation laws.

## What Changes

- Add bounds to parameters, `parameter(adaptive=True, bounds=(lower, upper))` in `core.components`, so an adaptive
  parameter is a control input that a schedule, an optimal solution, or a controller supplies.
- Add optimal control: the `Objective` law in `core.laws` gains its continuous-time form, a running cost and a terminal
  cost over the horizon of a run, with bounds on controls and states. Two solvers in `core.solvers` minimise it:
  `Pontryagin`, with symbolic costates and a forward-backward sweep, and `Collocation`, a direct transcription solved by
  a nonlinear programming solver. Both declare their `Supports`, and `validate()` reports a system they cannot solve.
- Add feedback controllers to the library: `PID` and `StateFeedback` in `skcomplex.models.engineering.realizations`,
  interactions whose rate and boundary laws read states and set adaptive control parameters during a run, with the
  integral of the PID error in an `Integrator` element.
- Add linear analysis in `core.analysis`: linearisation to state-space matrices `A, B, C, D` at an operating point,
  controllability and observability ranks, and linear-quadratic regulator gains.
- Add library models:
  - optimal harvesting of a logistic resource;
  - vaccination in an SIR epidemic;
  - Ramsey optimal growth;
  - a PID-stabilised inverted pendulum on a cart;
  - the linear-quadratic regulated oscillator.

## Capabilities

### New Capabilities

- `control`: bounded control inputs, optimal control by objectives and their solvers, and linear analysis.

### Modified Capabilities

- `model-library`: control models and the controller realizations.

## Impact

- `skcomplex.core.components`: `parameter` gains `bounds=`.
- `skcomplex.core.laws`: `Objective` gains its continuous-time form with `terminal=`.
- `skcomplex.core.solvers`: `Pontryagin` and `Collocation`, with `Supports` declarations.
- `skcomplex.core.analysis`: `linearize`, `controllability`, `observability`, and `lqr`.
- `skcomplex.models.engineering.realizations`: `PID` and `StateFeedback`.
- Builds on the archived `add-hybrid-dynamics` for time-dependent inputs and `add-analysis-toolkit` for Jacobians, and
  on `redesign-system-model` for adaptive parameters, the `Objective` law kind, simulations that take observables, and
  validation. The `Objective` law is shared with `add-static-models` and `add-constraint-based-models`.
