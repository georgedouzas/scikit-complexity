## Context

`stability` in `src/skcomplex/core/analysis/_stability.py` has the kinds `'fixed-points'` and `'dispersion'`. The
first builds `_DynamicsSpec` through `_dynamics_of`, which expands groups symbolically and raises above
`MEMBER_LIMIT = 64` members or on callable laws. `_find_points` asks Sage `solve` for closed forms and falls back to
`_numeric_points`, which runs `scipy.optimize.root` from 50 uniform starts in `[-10, 10]` and keeps the distinct
successes. `_eigenvalues_at` substitutes a point into the Sage Jacobian, and `_classify` names the class.

The compiled `ProblemSpec` already carries the numeric drift `drift(t, x, p)` and, for symbolic laws, a
`JacobianKernel` whose `run` gives the exact Jacobian (`ScipyOdeSolver._jacobian_options` uses it). Disordered
systems draw their quenched parameters in `_build_system` from `random_state`, as `HigherOrderRandomLotkaVolterraSystem`
does, so a new realization is the same system with another `random_state`.

## Goals / Non-Goals

**Goals:**
- One numeric root search, with deflation, shared by the census and by the fallback of `'fixed-points'`.
- A census that scales to groups of a few hundred members and to callable laws.
- Quenched and annealed counts by index from one call.

**Non-Goals:**
- Certified enumeration of all roots, such as homotopy continuation. The census counts what deflated Newton finds.
- Analytic complexities of the paper. They stay in the gallery example as reference curves.
- Censuses of PDE fields, delays, or stochastic systems.

## Decisions

- **A kind of `stability`, not a new analysis.** The census answers the question of `stability`, which fixed points
  exist and how stable each is, for systems that Sage cannot solve. A new function would split one concept. The kind
  reads the settings `parameters`, `starts`, `bounds`, `realizations`, and `random_state`, checked with
  `_check_settings` like the other kinds. `starts` defaults to 100, and `bounds` to `(-10, 10)`, the box of the former
  fallback, or one pair of bounds per state component.
- **Deflation of Farrell, Birkisson, and Funke (2015).** After roots `r_1 … r_m` are found, the search solves
  `M(x) f(x) = 0` with `M(x) = prod_j (1 / |x - r_j|^2 + 1)`, by `scipy.optimize.root` from every start. Deflation
  turns Newton away from known roots, so starts in the basin of a found root reach new ones. A root is accepted when
  the undeflated residual is below a tolerance and it is farther than `1e-6` from every known root. The power 2 and
  the shift 1 are the paper's defaults and stay module constants, not settings.
- **Numeric drift and Jacobian.** A private `_census_function` reads `problem.drift`, or the next value minus the state
  for a map, at the parameter values. The Jacobian comes from `problem.jacobian.run` when the problem has an exact
  `JacobianKernel`, from `jax.jacfwd` of the traced function of `JaxBackend` when the system holds learned functions of
  `add-learning-foundation` and validates as differentiable, and from `scipy.differentiate.jacobian` otherwise, for
  callable laws and scikit-learn regressors. Systems identified with `identify` are the cases that need the second
  path, and the exact Jacobian keeps their indices free of finite-difference errors. Indices come from
  `np.linalg.eigvals`, with the tolerance `ZERO` that `_classify` already uses for marginal eigenvalues.
- **`_numeric_points` is replaced** by the deflated search, so `'fixed-points'` gains it. Its symbolic path is
  unchanged.
- **Realizations by `random_state`.** The census clones the system with `set_params(random_state=seed)` for seeds
  drawn from `np.random.default_rng(random_state)`, compiles each, and searches it. A system without the parameter
  raises `AnalysisError` when `realizations > 1`.
- **Result fields.** `StabilityResult` gains `points` (per realization, an array of points by state component),
  `indices` (per realization), `counts` (realizations by index from 0 to the number of components), `mean_count`,
  `typical_count`, `annealed_complexity`, and `quenched_complexity`. `names` holds the component names, as for
  `'fixed-points'`. The complexities are `log(mean n) / N` and `mean log(1 + n) / N`, as in Eq. 3 of the paper.
- **Random force field by random Fourier features.** `RandomForceFieldRelaxationSystem` draws `M` wave vectors `k_m`
  from a Gaussian of variance `1 / N` per component, phases `phi_m`, and amplitudes `a_m`. The field is
  `F(x) = sum_m a_m cos(k_m . x + phi_m)`. The gradient part takes `a_m` along `k_m`, so its Jacobian is symmetric,
  and the divergence-free part takes `a_m` orthogonal to `k_m`, so `a_m . k_m = 0`. They are weighted by `sqrt(tau)`
  and `sqrt(1 - tau)`, and the overall scale sets `gamma` of Eq. 7. As `M` grows the field tends to an isotropic
  Gaussian field with a Gaussian covariance, one of the families of the paper. The field is a numeric function of the
  state read by a rate law of the group, the `numeric-functions` primitive.
- **Libraries surveyed.** SciPy has `optimize.root` and `differentiate.jacobian` but no deflation. `defcon` 0.12.2
  (2025-07-16) deflates over FEniCS and PETSc problems, not NumPy drifts. `PyDSTool` 0.91.0 (2020-01-31) is stale.
  Homotopy continuation (`HomotopyContinuation.jl`, Bertini) needs Julia or polynomial systems. So deflation is a short
  private wrapper over SciPy in the core.

- **Placement.** The kind `'census'` is step 2, a kind of a core analysis that reads compiled problems, in
  `core/analysis/_stability.py`. Domains: statistical physics of disordered systems, ecology of large communities,
  neural networks, economics of many goods, and every system identified from data. `RandomForceFieldRelaxationSystem`
  is step 3, one system, in `physics/statistical_mechanics`.

## Risks / Trade-offs

- [Exponentially many roots] → The census counts the roots that `starts` deflated searches find. The result states
  the number of starts, and the guide says the counts are lower bounds that grow with `starts`.
- [Cost of deflation] → Each search evaluates the product over found roots. For thousands of roots this dominates. It
  is vectorised over roots with NumPy, and the example keeps `N` small.
- [Finite-size counts differ from the asymptotic complexities] → The example compares the trend in `gamma` and `tau`,
  not exact values.

## Migration Plan

None. The kind is new, and the fallback of `'fixed-points'` returns a superset of the roots it returned before.

## Open Questions

- The number of random features `M` of the default system. A few times `N` should suffice. The gallery example will
  settle it by checking the empirical covariance.
