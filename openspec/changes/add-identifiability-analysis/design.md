## Context

After `add-bayesian-calibration`, `src/skcomplex/core/experiments/_calibration.py` holds the private class
`_CalibrationObjective`, which builds candidate systems, simulates them at the data times, and returns residuals and
the Gaussian log-likelihood, and `calibrate` reads bounds as pairs or SciPy priors. `sweep` in `_sweep.py` already
draws rows from bounds with `design='random'` or `'latin_hypercube'`, runs them on an executor, and applies analyses to
each trajectory. `DosingInteraction` in `biology/common/pharmacokinetics` gives doses on a schedule, and
`AntiPD1CheckpointSystem` in `biology/oncology/_immunotherapy.py` is a smaller checkpoint model with a continuous
infusion.

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
  relative error and the thresholds of its Definition 4.1 give the verdicts. The copies run on the executors of
  `_runs.py`, with streams spawned from `random_state`.
- **`IdentifiabilityResult`.** Fields `estimates`, `verdicts`, `profiles` (by parameter, the grid and the ratio, or
  `None` for refits), `intervals` (by parameter, `(low, high)` with `nan` for an open side, or `None` for refits), and
  `errors` (by parameter and level, or `None` for profiles).
- **Virtual populations on `sweep`.** `virtual_population` calls `sweep` with `design='random'` or
  `'latin_hypercube'` over the priors and with the outputs as analyses, then keeps the rows inside every constraint.
  With `target`, the probability of inclusion is the ratio of a `scipy.stats.gaussian_kde` of the target to one of the
  plausible outputs, scaled so its largest value is one, and a seeded draw selects rows. This is the method of Allen et
  al. (2016) that 2403.03335 describes. No mature library implements it, and the surveyed SBI and QSP tools (pyabc,
  pypesto, numpyro) do not, so it stays in house as a few lines over `sweep` and SciPy.
- **`VirtualPopulationResult`.** Fields `plausible` (the rows and their outputs), `acceptance`, `selected` (rows or
  `None`), and `inclusion` (probabilities or `None`), with `to_frame` as on `ExperimentResult`.
- **Per-individual calibration.** `calibrate_individuals` takes data by individual and `known` by individual, sets the
  known parameters with `set_params` on a copy of the system or passes them to the factory, and calls `calibrate`
  with any of its methods. The table is a `pandas.DataFrame` with one row per individual. The fits run on the
  executors of `_runs.py`.
- **QSP system.** `ImmunoOncologyQspSystem` in its own module of `biology/oncology` holds cancer cells with logistic
  growth, effector T cells recruited in proportion to the tumour and killing it at a rate lowered by free PD-1, Tregs
  that suppress effector proliferation, PD-1 occupancy at quasi steady state from the central antibody, and a
  two-compartment antibody with doses through `DosingInteraction`. The tumour diameter comes from the volume of a
  sphere. Parameters default to the ranges of the QSP-IO papers that 2403.03335 cites.
- **Dependency.** pypesto 0.8.0, released 2026-11-04 (checked on PyPI on 2026-11-08), needs Python 3.12 or later,
  which matches the project, and pulls matplotlib, seaborn, h5py, and pandas.

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

- The brief names `biology/pharmacology` for `ImmunoOncologyQspSystem`, a subfield that does not exist yet.
  `add-hybrid-neural-laws` also needs it. The plan puts the system in `biology/oncology`. Creating the subfield needs
  the user's approval and only changes the module path and the registered field.
