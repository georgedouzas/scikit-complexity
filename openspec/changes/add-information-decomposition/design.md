# Design

## Context

Entropies and transfer entropy go through `antropy` and the project's estimators, and `add-emergence-analysis` adds partition coarse-graining and effective information. The new measures build on both.

This design is rebased on `redesign-system-model`. The measures live in `skcomplex.core.analysis` and read trajectories
through the existing adapters, whose observables come from simulations and catalog entries after the redesign. They add
no primitive, solver, or backend, so they need no requirement or support declaration.

## Goals / Non-Goals

**Goals:**
- Measure redundancy and synergy among many variables of any model.
- Test whether a macro feature of a model is emergent by the criteria the literature uses.

**Non-Goals:**
- Integrated information Φ of IIT, which PyPhi computes and which needs exhaustive bipartitions.

## Decisions

### Libraries

`dit` is the maintained library of discrete information theory with partial information decomposition and several redundancy measures, and the toolkit wraps it for discrete data. O-information and its gradients on continuous data use Gaussian estimators, which are closed-form covariance determinants, and the k-nearest-neighbour estimators already in the toolkit.

Survey result:

- `dit` 2.3 is released on PyPI, is pure Python on NumPy, SciPy, NetworkX, pandas, and xarray, and runs on Python 3.12
  and 3.13. It computes total correlation, dual total correlation, and O-information of a joint distribution, and the
  partial information decomposition with 28 redundancy measures, among them the minimum mutual information, Williams
  and Beer's `I_min`, BROJA, `I_ccs`, `I_pm`, and `I_dep`. It qualifies, so it is a runtime dependency, imported on first
  use through the analysis package's lazy loader, and the plug-in measures of discrete data build a `dit` distribution
  from the empirical frequencies.
- The Gaussian estimators are closed forms over the sample covariance: total correlation is `-log det R / 2` of the
  correlation matrix `R`, and dual total correlation is `(log det Σ + Σ_i log (Σ⁻¹)_ii) / 2`. No library is needed.
- The k-nearest-neighbour estimator is `ennemi`, already a dependency, which estimates the mutual information of one
  variable with another, conditioned on any set. Total correlation is the chain sum `Σ_i I(X_i ; X_<i)`, and each term is
  the chain sum `Σ_{j<i} I(X_i ; X_j | X_<j)`. The sum of `I(X_i ; X_-i)` over the variables, which equals total plus dual
  total correlation, is built the same way, so the three measures need only `ennemi` calls.
- Local measures are the logarithms of the plug-in frequencies or of the fitted Gaussian densities at every sample, so
  their means equal the global measures exactly. No library computes them on arrays, and JIDT, which does, is a Java
  library.
- ε-machine reconstruction: `cmpy` of the Crutchfield group is not on PyPI and targets Python 2, `transCSSR` is a script
  repository with no release, and neither `dit` nor any maintained package implements causal-state splitting, so the
  core implements it.
- Surrogate tests shuffle every variable independently, so they need no library.

### Emergence criteria

Ψ, Δ, and Γ follow Rosas et al. (2020) with the minimum mutual information redundancy, so they need only mutual informations, which keeps them cheap; the redundancy is a keyword.

### Closure and ε-machines

A coarse-graining is closed when the macro series predicts its own future as well as the micro series predicts it, compared by conditional entropies. ε-machines are reconstructed from discrete series by the causal-state splitting algorithm.

## Risks / Trade-offs

- [Estimators are biased for short series] → Every measure reports its estimator and accepts a surrogate test.

## Decisions made during implementation

- The proposal names one `local_information` function. The local measures are three functions instead,
  `local_entropy`, `local_mutual_information`, and `local_transfer_entropy`, since each takes different arguments and
  mirrors a global measure of the same name. Each takes the `'plugin'` or `'gaussian'` estimator, and its mean equals
  the global measure of the same estimator exactly.
- The multivariate measures take `'plugin'` through `dit`, `'gaussian'`, and `'knn'` through `ennemi`. The partial
  information decomposition takes the redundancy measures `'mmi'`, `'williams_beer'`, `'broja'`, `'ccs'`, `'pm'`, and
  `'dep'` of `dit`.
- The surrogate test of the risks section is one function, `surrogate_test`, which takes any measure of many variables
  and shuffles every variable independently, so every measure accepts it without a keyword of its own.
- `emergence_criteria` computes the practical criteria Ψ, Δ, and Γ of Rosas et al. (2020), which are defined with the
  minimum mutual information redundancy. Another redundancy would need the full integrated information decomposition
  of the four variables of two times, which no maintained library computes, so the function takes no redundancy
  keyword, and `partial_information` keeps the keyword for two sources.
- `add-emergence-analysis` is not merged, so `closure` takes a partition in the convention its design fixes: the macro
  label of every integer micro state, or a function from the micro series to the macro series. When
  `partition_coarse_grain` lands, `closure` can accept its partitions unchanged. The result holds the two conditional
  entropies, `H(V_t | V_past)` and `H(V_t | X_past)`, and their gap, and leaves the threshold of closure to the caller.
