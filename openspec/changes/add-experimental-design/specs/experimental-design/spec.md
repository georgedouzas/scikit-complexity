# Spec Delta

## Purpose

Lets a user choose the experiment that best determines the parameters of a system, before any data or round by round,
and search the parameters of a system for the best outcome, with one or several objectives.

## ADDED Requirements

### Requirement: Designs maximise the information of the parameters
`design_experiment(system, candidates, ...)` SHALL choose `size` points among candidate observation times, inputs, or
conditions, or within their bounds, by the D-optimal or A-optimal criterion of the Fisher information or by the
expected information gain over a prior, and SHALL return a frozen `DesignResult` with the design, its value, the value
of every candidate, and the system set to the chosen conditions.

#### Scenario: Decay sampling time
- **WHEN** the D-optimal single observation time of `y = exp(-k t)` with `k = 0.5` and Gaussian noise is chosen among
  401 times on `[0, 10]`
- **THEN** the chosen time is 2.0 within the spacing of the candidates

#### Scenario: Michaelis-Menten rates
- **WHEN** the locally D-optimal two point design of the initial rate `V S / (K + S)` with `V = 1` and `K = 1` is
  chosen for substrates within `[0, 10]`
- **THEN** the design holds 10 and `10 / 12` within 1%, the values of Duggleby (1979) once checked against that paper

#### Scenario: Linear Gaussian information gain
- **WHEN** the expected information gain of `y = theta xi + epsilon` with standard normal `theta` and `epsilon` is
  estimated at `xi = 1` with 2000 outer and 200 inner samples
- **THEN** it is within 3% of `log(2) / 2`

#### Scenario: Same seed
- **WHEN** an `'eig'` design runs twice with the same `random_state`
- **THEN** both results hold equal designs and values

#### Scenario: Unknown variable
- **WHEN** a candidate variable is neither `'times'` nor a state, parameter, or input of the system
- **THEN** it raises `ValidationError` that names the variable and lists the allowed kinds

### Requirement: Designs adapt round by round
`design_experiment` with `rounds` and `policy` SHALL train a `PolicyInteraction` whose action is the next design and
whose observation is the history of designs and outcomes, on the sequential information gain over the prior, and SHALL
return the system holding the trained policy, so `simulate` runs an adaptive experiment.

#### Scenario: Location finding
- **WHEN** a design policy is trained for 10 rounds of the one-dimensional location finding of Foster et al. (2021)
  with `random_state=0`
- **THEN** its estimated information gain exceeds that of uniformly random designs and of the best fixed design of 10
  points

#### Scenario: Adaptive experiment runs
- **WHEN** the system of the result is simulated
- **THEN** it records 10 designs and 10 outcomes, each design within its bounds

### Requirement: Optimisation finds the best parameters of a system
`optimize(system, parameters, objective, method=...)` SHALL search bounded parameters for the best value of one
objective with `'bayesian'` or `'cma'`, or the Pareto front of several with `'nsga2'`, and SHALL return a frozen
`OptimizationResult` with the best parameters and value, the history of every evaluation, the front, and the system set
to the best parameters.

#### Scenario: Switch time of a bang-bang control
- **WHEN** the switch of the open-loop `MinimumTimeControlSystem` from `(3, 2)` is searched within `[0, 6.47]` to
  minimise the distance from the origin at `2 + 2 sqrt(5)`, with a budget of 60 evaluations
- **THEN** both `'bayesian'` and `'cma'` find `2 + sqrt(5)` within 0.01

#### Scenario: Two objectives
- **WHEN** `'nsga2'` searches the switch and the final time of the same system for the smallest final time and the
  smallest final distance
- **THEN** the front holds a point whose final time is within 2% of `2 + 2 sqrt(5)` and whose distance is below 0.05

#### Scenario: Same seed
- **WHEN** an optimisation runs twice with the same `random_state` on the serial executor
- **THEN** both results hold equal histories

#### Scenario: Several objectives with one objective method
- **WHEN** two objectives are given with `'bayesian'`
- **THEN** it raises `ValidationError` that names `'nsga2'`
