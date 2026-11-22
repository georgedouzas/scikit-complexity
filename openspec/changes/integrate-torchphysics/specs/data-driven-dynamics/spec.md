## ADDED Requirements

### Requirement: Physics-informed surrogates cover a range of parameters

`parametric_surrogate(system, params=...)` SHALL train one physics-informed network on the laws of a field system over
its domain, its time span, and the ranges of the named parameters, without runs of the system. Its result SHALL hold
`system`, `training_error`, and `validation_error`, and the `system` SHALL give the field at any parameter value inside
the ranges without a new training or solve.

#### Scenario: Diffusion sweep

- **WHEN** the heat equation on the unit interval with fixed zero ends and the initial profile `sin(πx)` is trained
  for the diffusion coefficient `k` in `(0.5, 2.0)` up to time 0.2 with a fixed random seed
- **THEN** at `k = 0.7`, `1.3`, and `1.9` the surrogate agrees with the exact solution `exp(-k π² t) sin(πx)` within
  2e-2 in the relative L2 norm over space and time

### Requirement: Parametric surrogates validate against solves

The validation error of a parametric surrogate SHALL compare it with solves of the system itself at parameter values
held out of the training.

#### Scenario: Validation against solves

- **WHEN** the surrogate of the diffusion sweep is built with three held out values of `k`
- **THEN** its validation error is the relative L2 error against finite difference solves of the system at those
  values, and is below 2e-2

### Requirement: Parametric surrogates stay inside their ranges

The system of a parametric surrogate SHALL reject a parameter value outside its trained range, with an error that
names the parameter and its range.

#### Scenario: Value outside the range

- **WHEN** the surrogate of the diffusion sweep is simulated at `k = 3.0`
- **THEN** it raises an error that names `k` and the range `(0.5, 2.0)`
