## Why

A calibration returns estimates even when the data cannot pin them down, and the Jacobian covariance of `calibrate`
says little when the loss is flat or curved. Practical identifiability by profile likelihood and by refits under noise
tells a user which parameters the data determine (2503.17135). Quantitative systems pharmacology goes one step further:
it samples virtual patients, keeps those whose outputs are plausible, matches the population to observed patients,
and then calibrates digital twins one patient at a time (2403.03335). None of these exist in the experiments layer.

## What Changes

- **Identifiability is an experiment.** `identifiability(system, data, bounds, method=...)` returns a frozen
  `IdentifiabilityResult`. With `method='profile'` it computes the profile likelihood of every parameter with pyPESTO,
  the likelihood-ratio confidence interval at a level, and a verdict per parameter. With `method='refits'` it refits
  the model to data simulated at the best fit with Gaussian noise at given levels, and reports the average relative
  error of every parameter with the verdicts of Liyanage et al. (strongly, weakly, or not practically identifiable).
- **Virtual populations.** `virtual_population(system, bounds, outputs, constraints, ...)` samples parameters from the
  priors, runs the system, keeps the plausible ones whose outputs satisfy the constraints, and, given observed values
  of an output, selects among them by the probability of inclusion, the ratio of the observed density to the density
  of the plausible population. It returns a frozen `VirtualPopulationResult`.
- **Per-individual calibration.** `calibrate_individuals(system, data, bounds, ...)` calibrates every individual's data
  with `calibrate`, with known per-individual parameters such as a dose, and returns a frozen
  `IndividualCalibrationResult` with the fit of each individual and a table of the estimates.
- **New system.** `ImmunoOncologyQspSystem`, a reduced quantitative systems pharmacology model of a tumour, effector
  and regulatory T cells, and an anti-PD-1 antibody with two-compartment kinetics and dosing, after the QSP-IO platform
  that 2403.03335 reviews.
- **Dependency.** pypesto becomes a runtime dependency.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `experiments`: identifiability by profile likelihood and by refits, virtual populations with plausibility
  constraints and selection by inclusion, and per-individual calibration.
- `model-library`: the immuno-oncology QSP system, recorded in the inventory.

## Impact

- Code: new modules `_identifiability.py`, `_virtual_population.py`, and `_individuals.py` in
  `src/skcomplex/core/experiments/`, exported from its `__init__`. They reuse `_CalibrationObjective` and the priors
  of `add-bayesian-calibration`, and `sweep` for the runs of virtual patients.
- Systems: a new module in `biology/oncology`, using `DosingInteraction` of `biology/common/pharmacokinetics`, with an
  inventory entry and a gallery example.
- Dependencies: pypesto 0.8.0 (2026-11-04) is added to `pyproject.toml` and to the Project Profile.
- Depends on `add-bayesian-calibration`, which must land first.
