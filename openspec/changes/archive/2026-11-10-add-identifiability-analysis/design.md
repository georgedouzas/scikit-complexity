## Context

`add-bayesian-calibration` has landed. `src/skcomplex/core/experiments/_calibration.py` holds `calibrate`,
`CalibrationResult`, and the private class `_CalibrationObjective`, which builds candidate systems, simulates them at
the data times, and returns residuals, the search box, and the Gaussian log-likelihood with the variance profiled out.
`calibrate` reads bounds as pairs or SciPy priors. `sweep` in `_sweep.py` already draws rows from bounds with
`design='random'` or `'latin_hypercube'`, runs them on an executor, and keeps outputs that are paths or functions of a
trajectory. `DosingInteraction` in `biology/common/pharmacokinetics` gives doses on a schedule, `CompartmentElement` in
`biology/common/kinetics` holds one amount, and `AntiPD1CheckpointSystem` in `biology/oncology/_immunotherapy.py` is a
smaller checkpoint model with a continuous infusion. Results of experiments are frozen dataclasses, and methods begin
with a verb, so `ExperimentResult` has `convert_to_frame`.

## Goals / Non-Goals

**Goals:**
- Practical identifiability of any calibratable system, by profile likelihood and by refits, with plain verdicts.
- Virtual populations and per-individual calibration as experiment functions built on `sweep` and `calibrate`.
- A QSP probe with dosing, plausibility constraints, and observable immune ratios.

**Non-Goals:**
- Structural identifiability. It needs differential algebra (StructuralIdentifiability.jl in Julia in 2503.17135), and
  a symbolic test on the Sage layer can follow in its own change.
- Mixed effects and population parameters learned jointly with individuals. `add-hybrid-neural-laws` brings mixed
  effects.
- The full QSP-IO platform with its many species and multi-omics calibration.

## Decisions

- **Placement.** The placement rule of `openspec/constitution.md` places parts, and experiment functions are the
  public experiment API of `skcomplex.core.experiments`, where `basin_stability` and `posterior_coverage` already live
  although they are expressible with `ensemble` and `calibrate`. The three new functions apply to any calibratable
  system of any field. Identifiability asks what logistic growth data in ecology, case counts of an epidemic, or the
  concentrations of a drug determine, virtual populations sample plausible patients in pharmacology, plausible
  economies in macroeconomics, or plausible climates, and per-individual calibration fits each patient, each region of
  an epidemic, or each firm. So they stay in `skcomplex.core.experiments`, next to `calibrate` and `sweep`. Placing
  them in `skcomplex.systems.general` would need a new package there, and the core never imports the systems, so the
  public experiment API would split in two. `ImmunoOncologyQspSystem` is one system, so step 3 of the rule places it in
  its own module, `biology/oncology/_immuno_oncology_qsp.py`, built from `CompartmentElement` and `DosingInteraction`
  of the biology `common` package and `ReactionInteraction` of `skcomplex.systems.general.reactions`. It needs no new
  core part.
- **One function per question.** `identifiability` asks what the data determine, `virtual_population` asks which
  parameters are plausible, and `calibrate_individuals` fits each individual. Each returns a frozen result, like the
  other experiment functions. Alternative: methods of `calibrate`, rejected because their results differ in kind.
- **Profiles with pyPESTO.** A `pypesto.Objective` wraps the negative Gaussian log-likelihood of
  `_CalibrationObjective`, with finite-difference gradients through `pypesto.objective.FD`, in a `pypesto.Problem`
  with the bounds of the search box. `pypesto.optimize.minimize` with a SciPy optimiser and `n_starts` starts gives the
  best fit, and `pypesto.profile.parameter_profile` the profiles. The interval at `level` comes from the chi-square
  threshold of one degree of freedom on the profile ratio. A side that reaches its bound without crossing the
  threshold is open, and the verdict is `'not identifiable'`. Alternatives: an in-house profile loop over `minimize`,
  rejected by the constitution since pyPESTO does it, and pints, which has no profiles.
- **Refits reuse `calibrate`.** `'refits'` calls `calibrate` with `method='local'` once on the data for the best fit,
  then on `n` noisy copies per level, with `noise` relative to the simulated values as in 2503.17135. The average
  relative error and the thresholds of its Definition 4.1 give the verdicts, and a parameter takes its worst verdict
  over the levels. The noise of every copy comes from one generator seeded by `random_state` before any fit runs, so
  the result does not depend on the executor. The fits run on an executor through `_map_calls` of `_runs.py`, the one
  helper that maps a function over arguments on the executors, which `posterior_coverage` now uses too in place of its
  own pool.
- **Profiles on the unit box.** pyPESTO works on the bounds scaled to the unit box, so its profile steps and finite
  differences are shares of the range of every parameter. On the raw scale a step of `1e-6` is below the tolerance of
  the solver for a capacity of hundreds, and the profiles came out ragged.
- **`IdentifiabilityResult`.** Fields `estimates`, `verdicts`, `profiles` (by parameter, the grid and the ratio, or
  `None` for refits), `intervals` (by parameter, `(low, high)` with `nan` for an open side, or `None` for refits), and
  `errors` (by parameter and level, or `None` for profiles).
- **Virtual populations on `sweep`.** `virtual_population` calls `sweep` with `design='random'` or
  `'latin_hypercube'` over the priors and with `outputs`, paths whose last values or functions of a trajectory whose
  values are the outputs, then keeps the rows inside every constraint. `constraints` maps an output name to its
  `(low, high)` range, and `target` maps one output name to its observed values.
  With `target`, the probability of inclusion is the ratio of a `scipy.stats.gaussian_kde` of the target to one of the
  plausible outputs, scaled so its largest value is one, and a seeded draw selects rows. This is the method of Allen et
  al. (2016) that 2403.03335 describes. No mature library implements it, and the surveyed SBI and QSP tools (pyabc,
  pypesto, numpyro) do not, so it stays in house as a few lines over `sweep` and SciPy.
- **`VirtualPopulationResult`.** Fields `plausible`, a `pandas.DataFrame` of the plausible rows with one column per
  parameter and output, `acceptance`, `selected`, the data frame of the selected rows or `None`, and `inclusion`, the
  probability of inclusion of every plausible row or `None`. The fields are data frames already, so the result needs
  no conversion method.
- **Per-individual calibration.** `calibrate_individuals` takes data by individual and `known` by individual, sets the
  known parameters with `set_params` on a copy of the system or passes them to the factory, and calls `calibrate`
  with any of its methods. `IndividualCalibrationResult` holds `calibrations`, the `CalibrationResult` of every
  individual, and `estimates`, a `pandas.DataFrame` with one row per individual. The fits run on an executor through
  `_map_calls`, as the refits do.
- **QSP system.** `ImmunoOncologyQspSystem` in `biology/oncology/_immuno_oncology_qsp.py` holds five compartments,
  cancer cells with logistic growth, effector T cells recruited in proportion to the tumour and killing it at a rate
  that free PD-1 lowers, regulatory T cells recruited by the tumour that suppress effector proliferation, and the
  central and peripheral amounts of a two-compartment antibody with doses through `DosingInteraction`. PD-1 occupancy
  is at quasi steady state, `c / (c + Kd)` of the central concentration `c`. It declares the observables `diameter`,
  the diameter of a sphere of the tumour volume in centimetres, and `effector_treg_ratio`. Parameters default to the
  orders of the QSP-IO papers that 2403.03335 cites, with an antibody dose of 240 mg every two weeks.
- **Dependency.** pypesto 0.8.0, released 2026-11-04 (checked on PyPI on 2026-11-10), needs Python 3.12 or later,
  which matches the project, and pulls matplotlib, seaborn, h5py, and pandas. The core experiments use it, so it is a
  runtime dependency, as `pyabc` and `optuna` are.

## Risks / Trade-offs

- [pyPESTO draws start points from the global NumPy random state] → seed it from `random_state` and restore the caller's
  state after the run.
- [Profiles cost many fits, each of many runs] → the defaults keep `n_starts` and the profile steps small, and the
  docs state the cost.
- [Verdicts depend on the bounds, since an interval is open when it reaches a bound] → the result holds the intervals
  and the profiles, so a user can widen the bounds and rerun.
- [pypesto adds seaborn and h5py to the install] → accepted as runtime dependencies, covered by the dependency audit.
- [Selection by inclusion needs enough plausible rows for the densities] → `virtual_population` raises
  `ExperimentError` when fewer than ten rows are plausible with a target.

## Migration Plan

Additive. No migration.

## Open Questions

None. The brief named `biology/pharmacology` for `ImmunoOncologyQspSystem`, a subfield that does not exist. Creating
it needs the user's approval, so the system lives in `biology/oncology`, as the plan says.
