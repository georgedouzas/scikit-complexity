# Tasks

## 1. Inputs and controllers

- [ ] 1.1 Add `bounds=` to `parameter` for adaptive parameters, with the validation issue for an initial value outside
  its bounds, and verify the scheduled control and control outside its bounds scenarios
- [ ] 1.2 Add `PID` and `StateFeedback` to `skcomplex.models.engineering.realizations` as elements with rate and
  equation laws on adaptive control parameters, and verify the inverted pendulum and no controller in the core
  scenarios

## 2. Optimal control

- [ ] 2.1 Add the continuous-time form of the `Objective` law with its running and terminal costs and its `Requires`,
  and verify the objective under discrete time scenario
- [ ] 2.2 Add the `Pontryagin` solver with symbolic costates and its `Supports` declaration, and verify the
  linear-quadratic and unsupported system scenarios
- [ ] 2.3 Add the `Collocation` solver with its `Supports` declaration, and verify the linear-quadratic and harvesting
  scenarios, and that the declaration test passes

## 3. Linear analysis

- [ ] 3.1 Add `linearize`, `controllability`, `observability`, and `lqr` to `core.analysis`, and verify the
  uncontrollable mode scenario

## 4. Library

- [ ] 4.1 Add the control models as registered factories whose catalog entries declare their observables, and verify
  the vaccination scenario and the conformance suite

## 5. Documentation and gate

- [ ] 5.1 Document bounded adaptive parameters, objectives under continuous time, the optimal control solvers and their
  declarations, the controller realizations, and linear analysis in the user guide, and verify the documentation test
  passes
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
