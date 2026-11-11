## Why

A model is useful for deciding what to do next: where and when to measure, which input to apply, which parameters to
choose for a design. The library simulates, calibrates, and identifies systems, but it cannot say which experiment would
pin the parameters down best, nor search the parameters of a system for the best outcome. Optimal experimental design
chooses observation times, inputs, or conditions by the Fisher information of the parameters or by the expected
information gain of Bayesian design (Rainforth et al. 2302.14545, Schoukens and Ljung 1902.00683), and adaptive design
chooses each experiment from the outcomes of the last ones, which a trained policy can do (Rainforth et al., sect. 6).
Design optimisation searches a system's parameters for the best value of an outcome, by Bayesian optimisation,
evolution strategies, or multi-objective search (Frazier 1807.02811). Before any design, structural identifiability says
whether the parameters can be determined from the observed outputs at all, even from perfect data (Raman 6.1.2).

Optuna and pymoo are already runtime dependencies, `add-learning-foundation` gives the observation model and policies as
components, so these become experiments with no new concept.

## What Changes

- New experiment `design_experiment(system, candidates, *, parameters, observation, criterion, size, ...)` in
  `skcomplex.core.experiments`, which chooses observation times, inputs, or conditions among candidates or within
  bounds by the D-optimal or A-optimal criterion of the Fisher information at given parameter values, or by the expected
  information gain over a prior, and returns a frozen `DesignResult` with the design, its value, and the value of every
  candidate.
- `design_experiment` gains `rounds` and `policy`: a design policy, a `PolicyInteraction` whose action is the next
  design and whose observation is the history of designs and outcomes, trained by `train_policy` on the sequential
  information gain, after the policy-based adaptive design of Rainforth et al. (sect. 6). Its result holds the trained
  system, so `simulate` runs adaptive experiments.
- New experiment `optimize(system, parameters, objective, *, method='bayesian', ...)`, which searches bounded
  parameters for the best value of an observable or a function of the run, with `'bayesian'` (Optuna's tree-structured
  Parzen estimator), `'cma'` (CMA-ES of pymoo), and `'nsga2'` for several objectives (pymoo), and returns a frozen
  `OptimizationResult` with the best parameters, the history, the Pareto front of several objectives, and the system
  set to the best values.
- `identifiability` gains `method='structural'`, the local structural identifiability of every parameter from the rank
  of the Jacobian of the Taylor coefficients of the observed outputs, computed on the symbolic laws with Sage.
- `MinimumTimeControlSystem` gains `control='open-loop'` with the switch time of the bang-bang control as a model
  parameter, whose optimum is known in closed form, as the probe of `optimize`.

## Capabilities

### New Capabilities

- `experimental-design`: optimal and adaptive experimental design, and design optimisation of systems.

### Modified Capabilities

- `experiments`: structural identifiability.
- `model-library`: the open-loop form of minimum time control.

## Impact

- Core: new modules `core/experiments/_experimental_design.py` and `_optimization.py`, and `_identifiability.py`.
- Systems: `engineering/control/_minimum_time_control.py`.
- No new dependency. Optuna 5.0.0 and pymoo 0.6.2 are runtime dependencies, and pymoo brings `cma` 4.5.0.
- Docs: a design section in the experiments page of the user guide and gallery examples of D-optimal sampling times
  and of the open-loop switch time found by every optimiser.
- Depends on `add-learning-foundation` for `ObservationSpec`, `PolicyInteraction`, and `train_policy`. The static
  designs, `optimize`, and structural identifiability need only the observation model.
