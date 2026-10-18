# Tasks

## 1. Inputs and controllers

- [ ] 1.1 Add `Control` with schedules, and verify the scheduled control scenario
- [ ] 1.2 Add `PID` and `StateFeedback` as compiled components, and verify the inverted pendulum scenario

## 2. Optimal control

- [ ] 2.1 Add `OptimalControlProblem` and `pontryagin` with symbolic costates, and verify the linear-quadratic scenario
- [ ] 2.2 Add `collocation`, and verify the linear-quadratic and harvesting scenarios

## 3. Linear analysis

- [ ] 3.1 Add `linearize`, `controllability`, `observability`, and `lqr`, and verify the uncontrollable mode scenario

## 4. Library

- [ ] 4.1 Add the control models as registered factories, and verify the vaccination scenario and the conformance suite

## 5. Documentation and gate

- [ ] 5.1 Add the `control` layer to the Project Profile, document controls, optimal control, controllers, and linear
  analysis in the user guide, and verify the documentation test passes
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
