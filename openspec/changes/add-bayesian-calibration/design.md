## Context

`calibrate` in `src/skcomplex/core/experiments/_calibration.py` builds candidate systems with `_build`, runs each with
`simulate`, interpolates the series at the data times, and minimises squares with `least_squares` or a Gaussian
likelihood with `minimize`. The objective lives in closures inside the function. Bounds are pairs only, while `sweep`
already reads a pair or a SciPy distribution through `_check_bounds` of `_designs.py`.

Draws in laws are symbolic calls (`bernoulli`, `normal`, `uniform`, `categorical`, `draw`) whose `DrawSpec` names the
distribution. Agent rules, such as `EncountersInteraction` of the Deffuant system, draw from `system.rng` of the view
and leave no record. `GillespieSolver` fires jump laws but keeps no list of firings. Gradients through runs exist only
inside `GradientDescentSolver`, whose `_unroll` lowers a nested system with the JAX backend and unrolls it with
`jax.lax.scan` for maps and `diffrax.diffeqsolve` for ordinary differential equations. The learning compiler rejects
noise, jump, rule, and callable laws as not differentiable.

## Goals / Non-Goals

**Goals:**
- One `calibrate` with methods for point fits, ABC-SMC, MCMC, and variational inference, one result type, and priors
  given as the bounds.
- An exact likelihood for any stochastic model through its own recorded draws, with no model-specific code.
- Diagnostics that say whether a posterior can be trusted: R-hat, effective sample size, ABC tolerances, posterior
  predictive bands, and coverage over prior draws.

**Non-Goals:**
- Neural posterior, likelihood, or ratio estimation (sbi, 2508.12939, 2601.07413). It needs torch and goes in the later
  torch change.
- Latent initial states inferred jointly with the parameters (the partly observed scenarios of 2310.02766).
- Normalizing-flow variational families. `add-generative-samplers` brings flowjax, and a flow guide can follow it.
- Amortised inference across data sets.

## Decisions

- **One function, two axes.** `method` says how to search or sample, `loss` says what scores a candidate. The allowed
  pairs are: `'local'` and `'global'` with `'squares'`, `'likelihood'`, or `'events'`, `'abc_smc'` with `'squares'`
  (the loss is then the distance of summaries), `'mcmc'` with `'likelihood'` or `'events'`, and `'variational'` with
  `'likelihood'` or `'squares'`. A table of the pairs is the one place they are checked. Alternative: a function per
  method, rejected because the result, the priors, and the objective are shared, and scikit-learn uses `solver=` the
  same way.
- **The objective is a private class.** The closures of `calibrate` become the private class `_CalibrationObjective`
  in `_calibration.py`, with methods that build a candidate, simulate it at the data times, and return residuals, the
  Gaussian log-likelihood, the event log-likelihood, the log prior, and summaries. Every method calls it, and
  `add-identifiability-analysis` reuses it from the same package.
- **Priors from the bounds.** `_check_bounds` of `_designs.py` already accepts pairs and SciPy distributions. A pair is
  `uniform(low, high - low)`. Point fits search the pair or the `ppf` of 0.0005 and 0.9995. The numpyro methods need
  traceable priors, so a private method maps the SciPy families `uniform`, `norm`, `lognorm`, `truncnorm`, `beta`,
  `gamma`, and `expon` to their numpyro distributions and raises `ValidationError` for any other family when the method
  is `'mcmc'` or `'variational'`.
- **`'local'`.** Today's code path, unchanged in its estimates.
- **`'global'`.** `optuna.create_study` with `TPESampler(seed=random_state)` and `budget` trials over the search box,
  then the covariance from the Jacobian at the best trial, as for `'local'`. Alternatives: `scipy.optimize`'s
  `differential_evolution`, kept out because TPE copes better with the noisy losses of stochastic runs and Optuna is
  the mature default for black-box search.
- **`'abc_smc'`.** `pyabc.ABCSMC` with the priors as `pyabc.Distribution` of `RV` objects built from the same SciPy
  distributions, an `AdaptivePNormDistance` over the summaries, `population_size` from `n_samples`, and a
  `SingleCoreSampler`, run against a SQLite file in a `tempfile.TemporaryDirectory`. It stops at `minimum_epsilon` or
  `max_total_nr_simulations=budget`. The last population gives `samples` and `weights`, and the history gives the
  tolerance and acceptance rate of every generation. Summaries default to the concatenated series at the data times.
  Alternatives: elfi, excluded by the user because it pins `matplotlib<3.9`. Synthetic likelihood is left out, since
  ABC-SMC covers the probes of 2409.19675.
- **`'mcmc'`.** numpyro `MCMC` with `num_chains` chains of `n_samples`. On a system with the tag `differentiable` and a
  Gaussian likelihood the kernel is `NUTS` on the JAX run. Otherwise it is the gradient-free ensemble kernel `AIES`,
  with the log-likelihood evaluated on the host through `jax.pure_callback`, which a gradient-free kernel allows.
  `numpyro.diagnostics.split_gelman_rubin` and `effective_sample_size` fill `diagnostics`. Alternatives: emcee, stale
  since 2024-04, and pints, which overlaps numpyro without JAX.
- **`'variational'`.** numpyro `SVI` with an `AutoMultivariateNormal` guide and an optax Adam optimiser. The model
  samples the parameters from the priors, runs the system on JAX, and adds the negative loss through
  `numpyro.factor`, so the ELBO is the expected loss plus the divergence from the prior: the classical posterior for
  `'likelihood'` and the generalised posterior of Knoblauch et al. for `'squares'` (2305.15340, Eq. 2). `samples` are
  `n_samples` draws of the fitted guide, and `diagnostics` holds the loss history.
- **The differentiable run is shared.** The unroll inside `GradientDescentSolver._unroll` moves to a method of
  `JaxBackend` that returns a traced function from the parameter vector to the states at given times, with
  `lax.scan` for maps and `diffeqsolve` with `Tsit5` for ordinary differential equations. The solver and
  `_CalibrationObjective` both call it, so the unroll stays in one place.
- **Recording draws.** With `record_draws=True`, the NumPy backend evaluates draws through one private recorder of the
  runtime, which stores the value and the evaluated distribution arguments by name and step in the trajectory.
  `GillespieSolver` stores the time and the law index of every firing. The system view gains `draw(name,
  distribution, **arguments)`, which samples from `rng` and records when recording is on, and `EncountersInteraction`
  draws its outcome through it. Recording selects the NumPy backend, since compiled kernels cannot call back.
- **Replay is the event likelihood.** With `replay=trajectory`, the recorder returns the recorded value of each named
  draw instead of sampling, evaluates the log-probability of that value under the arguments computed now, and sums it
  into `log_probability`. A replayed Gillespie run fires the recorded laws at the recorded times and adds `log
  a_k(x)` per firing minus the integral of the total propensity, which is exact for piecewise constant states.
  `loss='events'` is the negative of `log_probability`. This is the trace replay of probabilistic programming, and it
  needs no model-specific likelihood. Alternatives: writing each model as a numpyro program (Lenti et al.), rejected
  because it duplicates every model, and surveyed libraries (pyabc, numpyro, pints) offer no replay of a foreign
  simulator, so the recorder is in house.
- **Posterior predictive checks.** After a posterior method, `_CalibrationObjective` simulates `min(n_samples, 200)`
  draws, chosen by the weights, and stores the 5%, 50%, and 95% quantiles at the data times by path, and the share of
  observed points inside the band. ArviZ is not added, since numpyro already gives the diagnostics needed.
- **Coverage.** `posterior_coverage` in a new `src/skcomplex/core/experiments/_coverage.py` draws `n` truths from the
  priors, simulates data at them with optional Gaussian noise, calls `calibrate` per truth, and returns
  `CoverageResult` with the coverage per level and parameter and the ranks of simulation-based calibration. It runs
  the truths through the executors of `_runs.py`.
- **`CalibrationResult`.** New fields `samples: dict[str, Array] | None`, `weights: Array | None`, `diagnostics:
  dict[str, object]`, and `predictive: dict[str, object] | None`. Fields keep keyword construction.
- **New arguments.** `method='local'`, `summary=None`, `n_samples=1000`, `budget=None` (a default per method), and
  `chains=4`. `replicates`, `dt`, `solver`, and `random_state` keep their meaning. `random_state` seeds Optuna, pyabc,
  and the JAX keys of numpyro.
- **Probes.** Deffuant with steepness probes the event likelihood of agent rules. Stochastic SIR probes Gillespie
  firings and ABC-SMC. Brock-Hommes probes global fits and ABC-SMC on rough losses (in place of the sbi of
  2601.07413). `OffLatticeTumourGrowthSystem` and `CellCycleInvasionSystem` probe ABC-SMC with summaries on spatial
  agents (2409.19675). `DifferentiableCovidAbmSystem` probes variational calibration through a differentiable agent
  run (2305.15340).
- **Differentiable COVID agents carry probabilities.** The agents hold real states, the probabilities of being
  susceptible, infected, and recovered, updated by the expected infection of Eq. 6, so the run is a smooth map that the
  learning compiler accepts. Places are membership matrices built by the module from a seeded synthetic population,
  read through weighted reductions. GradABM uses relaxed Bernoulli draws instead, which would need reparameterised
  draws in the JAX lowering, see the open questions.
- **Off-lattice tumour.** A variable group of cells in a continuous square space. The module computes the Delaunay
  neighbours with `scipy.spatial.Delaunay` each step and relaxes springs by the update of Eq. 2.2. Division and
  invasion draw through named draws, so runs record them.
- **Dependencies.** Checked on PyPI on 2026-11-08: optuna 5.0.0 (2026-09-07, Python 3.9 or later), pyabc 0.13.1
  (2026-10-28, Python 3.11 or later), numpyro 0.22.0 (2026-09-18, Python 3.11 or later, needs only jax, jaxlib,
  multipledispatch, numpy, and tqdm). All are within the 24-month limit of `dependency-freshness` and support the
  Python versions of the project.

## Risks / Trade-offs

- [pyabc draws from the global NumPy random state] → save and restore `np.random` state around the run, seed it from
  `random_state`, and use `SingleCoreSampler`, so a result is reproducible and the caller's state is unchanged.
- [pyabc pulls `distributed`, `redis`, `sqlalchemy`, and `ipykernel` into the install] → accepted as runtime
  dependencies, since the user listed pyabc. The dependency audit of the gate covers them.
- [Optuna logs every trial] → silence its logger for the study only, through `optuna.logging` and a restore after.
- [The ensemble sampler with a host callback is slow and mixes poorly in many dimensions] → document it, report R-hat,
  and keep the probes to few parameters.
- [Recording draws holds one value per draw and step in memory] → recording is off by default, and the trajectory
  stores draws as arrays per name.
- [Replay diverges when a candidate changes how many draws a step makes, such as a variable group that grows] → the
  recorder raises `ValidationError` naming the step and the draw when the recorded and the requested draws disagree,
  and the event loss treats it as an infinite loss in point fits.
- [Variational calibration needs a JAX lowering of the whole system] → the method checks the tag `differentiable` and
  the learning compiler's report before any work.
- [`CalibrationResult` gains fields] → breaking for positional construction only, recorded in the changelog.

## Migration Plan

Calls of `calibrate` without `method` keep their estimates. The changelog names the new fields of
`CalibrationResult`. No data migration.

## Open Questions

- `CellCycleInvasionSystem` is a cell biology model of melanoma cells. The brief names `biology/cell_biology`, which
  does not exist. The plan puts it in `biology/oncology`. A new subfield needs the user's approval and only changes the
  module path and the registered field.
- Relaxed Bernoulli draws (Gumbel-softmax) in the JAX lowering would let `DifferentiableCovidAbmSystem` keep discrete
  agents as GradABM does. It is a new core primitive that can follow without changing this change's specs.
