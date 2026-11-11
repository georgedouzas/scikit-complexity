## Context

- `add-learning-foundation` adds `ObservationSpec`, which names the observed paths and their noise, `PolicyInteraction`,
  a policy as a component that sets actions from a learned function of its observation, and `train_policy`.
- `calibrate` and `identifiability` (`core/experiments/_calibration.py`, `_identifiability.py`) fit and test the
  parameters of a system. `identifiability` has the methods `'profile'` and `'refits'`, both practical: they ask
  whether the data at hand determine the parameters. `_CalibrationObjective.get_traced` gives the differentiable map
  from parameters to simulated observations on JAX.
- `sweep` takes a `design` among `'grid'`, `'random'`, `'latin_hypercube'`, and `'sobol'`, which are sampling plans of
  a sweep and not optimal designs.
- Optuna 5.0.0 and pymoo 0.6.2 are runtime dependencies. Optuna's `GPSampler` needs torch and its `CmaEsSampler` needs
  the package `cmaes`, neither of which is installed. pymoo depends on `cma` 4.5.0 (2026-09-13) and wraps it as
  `pymoo.algorithms.soo.nonconvex.cmaes.CMAES`.
- `MinimumTimeControlSystem` (`engineering/control/_minimum_time_control.py`) steers `x'' = a x + u`, `|u| <= 1`, by a
  sampled feedback on the switching curve. For the free particle from `(3, 2)` it reaches the origin at
  `v + 2 sqrt(x + v^2 / 2) = 2 + 2 sqrt(5)`, about 6.47.
- The gap report of the learning drafts (2026-11-10) found optimal design, adaptive design, design optimisation, and
  structural identifiability missing, and the user approved this change the same day.

## Goals / Non-Goals

**Goals:**
- Local and Bayesian optimal designs of observation times, inputs, and conditions for any system with an observation
  model.
- Adaptive design by a trained policy, with the parts of the foundation.
- Search of the parameters of any system for the best outcome, with one or several objectives.
- Structural identifiability before any data.

**Non-Goals:**
- Global structural identifiability by differential algebra. The local test by the rank of Taylor coefficients answers
  the common question, and a global test can follow.
- Gaussian process Bayesian optimisation, which Optuna runs on torch. TPE is the Bayesian optimiser of the core, and a
  GP sampler can join when torch is a dependency.
- Multi-fidelity design and optimisation.
- Archives of diverse solutions, which `add-diversity-search` holds.

## Decisions

### Placement

| Component | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| `design_experiment`, `DesignResult` | 2, an experiment over any system with an observation model | `skcomplex.core.experiments` | pharmacology (sampling times), epidemiology (allocation of tests), ecology (survey times), engineering (excitation signals), systems biology (perturbations), physics (measurement settings) |
| The adaptive design system | 2, a private system that the experiment builds from a `PolicyInteraction` and the system | `core/experiments/_experimental_design.py` | as above |
| `optimize`, `OptimizationResult` | 2, an experiment over any system | `skcomplex.core.experiments` | engineering (controller tuning), epidemiology (timing of interventions), ecology (harvest quotas), chemistry and materials (inverse design), economics (policy parameters) |
| `identifiability(method='structural')` | 2, an option of an experiment over any system with symbolic laws | `core/experiments/_identifiability.py` | systems biology, pharmacology, epidemiology, ecology |
| `control='open-loop'` of `MinimumTimeControlSystem` | 3, a model parameter of one system | `engineering/control/_minimum_time_control.py` | engineering |

The name `design_experiment` is a verb first, and keeps the word design of `sweep(design=...)` apart: a sweep samples a
plan, a designed experiment is chosen by a criterion.

### Libraries

No new dependency. `'bayesian'` uses Optuna's `TPESampler`, a sequential model-based optimiser, since `GPSampler`
needs torch. `'cma'` uses pymoo's `CMAES` on `cma` 4.5.0, since Optuna's `CmaEsSampler` needs `cmaes`, which is not
installed, and pymoo already brings `cma`. `'nsga2'` uses pymoo's `NSGA2`. pyro-ppl 1.9.2 (2026-10-03) has
`contrib.oed` for Bayesian design, but needs torch, so the Fisher information and the nested Monte Carlo estimator are
written on JAX and numpyro, a few lines each.

### Optimal designs

- `design_experiment(system, candidates, *, parameters, observation, criterion='d-optimal', size=1, prior=None,
  samples=(1000, 100), rounds=None, policy=None, until=None, executor='serial', random_state=None)` in
  `core/experiments/_experimental_design.py`.
- `candidates` maps a design variable to a sequence of candidate values or to bounds. A variable is `'times'`, the
  observation times of the run, or the path of an initial state, a model parameter, or an input held over the run.
  The experiment chooses `size` points, with repeats allowed, among the candidates, by the exchange algorithm of
  Fedorov over a finite set, or by CMA-ES within bounds.
- `parameters` maps the parameters of interest to their values, the point of a local design. `observation` is the
  `ObservationSpec` of the measurements, whose Gaussian scale gives the noise of the Fisher information.
- **Local criteria.** The Fisher information of a design is `J^T R^-1 J`, with `J` the Jacobian of the observed values
  in the parameters, by `jax.jacfwd` of the traced run for differentiable systems and by central differences otherwise.
  `'d-optimal'` maximises its log determinant and `'a-optimal'` minimises the trace of its inverse.
- **Bayesian criterion.** `'eig'` maximises the expected information gain over `prior`, a mapping of parameters to
  scipy distributions as `calibrate` takes them, by the nested Monte Carlo estimator of Rainforth et al. (sect. 3),
  with `samples` outer and inner draws, and with the noise of the observation model as the likelihood.
- `DesignResult` holds `design` (variable to chosen values), `value` (the criterion of the design), `values` (the
  criterion of every single candidate), `information` (the Fisher information of the design for local criteria), and
  `system`, the system set to the chosen conditions and inputs.

### Adaptive design

- With `rounds=k` and `policy`, an Equinox module, the experiment builds a private system: the parameters drawn from
  the prior at the start of every episode, a `PolicyInteraction` that observes the designs and outcomes so far and sets
  the next design as its action, and the system run at that design and observed through the observation model. The
  reward of the last round is the sequential prior contrastive estimate of the information gain of Foster et al.
  (2021), the bound of policy-based adaptive design that Rainforth et al. (sect. 6) review. `train_policy` trains the
  policy on it with `'reinforce'` or `'pathwise'`.
- The result's `system` is that system with the trained policy, so `simulate` runs an adaptive experiment of `k`
  rounds, and `value` is the estimated information gain of the policy.

### Design optimisation

- `optimize(system, parameters, objective, *, method='bayesian', direction='minimize', budget=100, until=None,
  replicates=1, constraints=(), executor='serial', random_state=None)` in `core/experiments/_optimization.py`.
- `parameters` maps parameter paths to bounds. `objective` is the path of an observable, read at the end of the run, or
  a callable of the trajectory, or a sequence of them for `'nsga2'`, with `direction` one value or one per objective.
  With `replicates` above 1, stochastic systems are scored by the mean over seeded runs. `constraints` are callables of
  the trajectory that must be at most zero.
- Every evaluation is a run of `simulate` with the parameters set through `set_params`, and evaluations of one batch run
  on the executor.
- `OptimizationResult` holds `best_params`, `best_value`, `history` (every evaluated point and value), `front` (the
  Pareto set and its values for several objectives, `None` otherwise), and `system`, set to the best parameters, or to
  the point of the front closest to the ideal point for several objectives.

### Structural identifiability

- `identifiability(system, data, bounds, method='structural', order=None)` reads only the paths of the observation
  model of `data` and the names of `bounds`. It differentiates the observed outputs along the rates of the system with
  Sage, `order` times (the number of states plus one by default), evaluates the Jacobian of these Taylor coefficients
  in the parameters at a random rational point within the bounds, and reports a parameter as not identifiable when
  removing its column does not lower the rank. The verdicts are `'identifiable'` and `'not identifiable'`, the names
  the profile method uses, and the result gains `rank`.
- The test is local and generic. The docstring says so, and that a parameter found identifiable can still be poorly
  determined by finite data, which the practical methods measure.

### Probes

- **Decay sampling time.** For `y = exp(-k t)` with Gaussian noise, the D-optimal single observation time is `1 / k`,
  since `t^2 exp(-2 k t)` is largest there. Analytic, so no source is needed.
- **Michaelis-Menten rates.** For the initial rate `V S / (K + S)` over substrates in `[0, S_max]`, the locally
  D-optimal two point design takes `S_max` and `K S_max / (2 K + S_max)`, after Duggleby (1979). That paper is not in
  `references/`, so the value must be checked against it, or replaced by an example of 2302.14545, before the test is
  written. The test builds the rate with `MichaelisMentenInteraction` of `biology/common/kinetics`.
- **Linear Gaussian information gain.** For `y = theta xi + epsilon` with `theta ~ N(0, s^2)` and `epsilon ~ N(0,
  sigma^2)`, the expected information gain is `log(1 + s^2 xi^2 / sigma^2) / 2`, which checks the nested Monte Carlo
  estimator.
- **Location finding.** The adaptive design probe is the one-dimensional location finding of Foster et al. (2021), a
  hidden source sensed with an intensity that falls with distance. The check is relative: the trained policy gains more
  information over 10 rounds than random and fixed designs.
- **Switch time.** `MinimumTimeControlSystem(control='open-loop', switch=...)` applies `u = -1` until `switch` and
  `u = 1` after it. For the free particle from `(3, 2)`, the state reaches the origin at `2 + 2 sqrt(5)` exactly when
  `switch = 2 + sqrt(5)`, about 4.236, the switch of the Pontryagin solution. `optimize` minimises the distance from the
  origin at that time over the switch.
- **Structural cases.** `x' = -(k1 + k2) x` observed in `x` leaves `k1` and `k2` not identifiable, and the
  Lotka-Volterra system observed in both populations has all four rates identifiable.

## Risks / Trade-offs

- [Nested Monte Carlo is costly and biased for few inner samples] → `samples` sets both sizes, the docstring states the
  bias, and the linear Gaussian scenario fixes the sizes that pass.
- [Local designs depend on the guessed parameters] → The docstring says so and points to `'eig'` with a prior.
- [TPE is weaker than a Gaussian process for smooth objectives of few dimensions] → The scenario sets a budget that
  TPE meets, and a GP sampler can join with torch.
- [The rank test evaluates at one random point] → The point is rational and the rank is exact over the rationals, so a
  wrong verdict needs the point to fall on a measure-zero set, and a second point is drawn when the rank differs from
  that of a first one.

## Later

These items of the gap report fit the theme of deciding with a model but are systems and recipes, not experiments, so
they are listed here rather than bloating this change:

- A reusable receding horizon controller, hoisted from `KoopmanModelPredictiveControlSystem`, as a policy whose action
  is the first control of an optimal control solve on an identified system (Brunton and Kutz 10.1, Moerland et al.
  2006.16712). It is used by engineering, epidemiology, and ecology, so it belongs in `systems.general`, and a new
  subpackage for it needs the maintainer's approval first.
- Adaptive control: extremum seeking (Brunton and Kutz 10.3) and model reference adaptive control (Padmanabhan et al.
  ch. 10) as systems of `engineering/control`, step 3.
