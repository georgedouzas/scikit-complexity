## Why

`calibrate` returns one point estimate, from a local SciPy fit of squares or of a Gaussian likelihood, with a covariance
from the Jacobian. Stochastic agent-based models need more than that: their losses are rough, their likelihoods are
intractable or known only through their own random draws, and a user needs posterior uncertainty with checks that it
is trustworthy (2508.12939, 2409.19675). Simulation-based inference, likelihood inference, and variational calibration
through differentiable runs are mature in libraries on the JAX and SciPy stack that the core already uses.

## What Changes

- **`calibrate` gains `method=`.** `'local'` is the current SciPy fit and stays the default. `'global'` searches the
  bounds with Optuna. `'abc_smc'` samples an approximate posterior by ABC-SMC with pyabc. `'mcmc'` samples the
  posterior of a likelihood with numpyro. `'variational'` fits a variational posterior with numpyro SVI through
  differentiable runs on JAX and diffrax (2305.15340).
- **Priors.** A bound is a pair `(low, high)`, read as a uniform prior, or a SciPy distribution, as in `sweep`.
- **Summaries.** `summary=` maps simulated or observed series to summary statistics, which ABC-SMC compares.
- **Event likelihood.** `loss='events'` scores the recorded draws of a run of a stochastic model under candidate
  parameters, by replaying the run with those draws fixed (2310.02766). It works for draws in laws, for draws of agent
  rules, and for the firings of jump laws under Gillespie.
- **Runs record and replay their draws.** `simulate(record_draws=True)` stores every named draw and every jump firing
  in the trajectory. `simulate(replay=trajectory)` forces those draws and records their log-probability. Agent rules
  draw through a new named `draw` of the system view, so their draws are recorded too.
- **BREAKING: `CalibrationResult` gains posterior fields.** It holds `samples`, `weights`, `diagnostics`, and
  `predictive` (posterior predictive quantiles and the share of observed points inside them). `estimates` is the
  posterior median for the posterior methods. `success` reads the convergence of every method.
- **Coverage.** The new experiment function `posterior_coverage` calibrates on data simulated from prior draws and
  reports the empirical coverage of the credible intervals at given levels.
- **Probes.** `DeffuantSystem` gains `steepness`, the stochastic outcome of Lenti et al. `SIRSystem`, `DeffuantSystem`,
  and `BrockHommesSystem` serve as probes. New systems `OffLatticeTumourGrowthSystem` and `CellCycleInvasionSystem`
  after 2409.19675, and `DifferentiableCovidAbmSystem` after 2305.15340.
- **Relaxed discrete draws.** `bernoulli` and `categorical` take `temperature` and `hard`. The JAX lowering replaces a
  draw with a temperature by its concrete (Gumbel-softmax) relaxation, with straight-through hard values as an option,
  so gradients flow through discrete draws. Other backends draw exactly and ignore the relaxation.
- **Delaunay neighbourhoods.** `DelaunayNeighborhood` joins the members of a continuous space along the edges of their
  Delaunay triangulation, across periodic boundaries, recomputed as members move.
- **Dependencies.** optuna, pyabc, and numpyro become runtime dependencies.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `experiments`: calibration methods, priors, summaries, the event likelihood, posterior results with diagnostics,
  and the coverage experiment.
- `simulation`: runs record their draws and jump firings and replay them with their log-probability.
- `model-library`: the stochastic Deffuant outcome and the three new probe systems, recorded in the inventory.
- `stochastic-maps`: Bernoulli and categorical draws take a continuous relaxation in the JAX lowering.
- `agent-spaces`: continuous spaces gain Delaunay neighbourhoods.

## Impact

- Code: `src/skcomplex/core/experiments/_calibration.py` (methods, priors, posterior result), a new
  `_coverage.py` beside it, `src/skcomplex/core/engine/runtime/_simulate.py` and `_trajectories.py` (recording and
  replay), the draws of the NumPy backend and `GillespieSolver`, the system view protocol in
  `src/skcomplex/core/components/types.py`, and `JaxBackend`, which takes the differentiable unroll that
  `GradientDescentSolver` uses today.
- Draws and neighbourhoods: `bernoulli` and `categorical` in `src/skcomplex/core/components/laws/utils`, `DrawSpec`, the
  draw lowering and `JaxBackend`, a new module `_delaunay.py` in `src/skcomplex/core/components/spaces`, and the metric
  search of `src/skcomplex/core/engine/problems/_metric_search.py`.
- Systems: `social/opinion_dynamics/_bounded_confidence.py`, and new modules in `biology/oncology` and
  `biology/epidemiology`, with inventory entries and gallery examples.
- API: `CalibrationResult` gains fields, `calibrate` gains `method`, `summary`, `n_samples`, and `budget`, and
  `simulate` gains `record_draws` and `replay`. Code that builds `CalibrationResult` positionally breaks.
- Dependencies: optuna 5.0.0 (2026-09-07), pyabc 0.13.1 (2026-10-28), and numpyro 0.22.0 (2026-09-18) are added to
  `pyproject.toml` and to the Project Profile of the constitution.
- Later changes: `add-identifiability-analysis` builds on the objective and the priors of `calibrate`.
