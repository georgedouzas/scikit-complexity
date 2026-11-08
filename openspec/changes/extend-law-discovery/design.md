# Design

## Context

- `sparse_identification` in `src/skcomplex/core/analysis/_sparse_identification.py` builds one `pysindy.SINDy` with
  `STLSQ` and `PolynomialLibrary`, reads the runs through `_read_samples`, and turns the coefficients and the
  `powers_` of the library into rate laws with `_sparse_laws` and `_surrogate_system`. It already pools a list of runs
  in one fit.
- `mode_decomposition` in `_mode_decomposition.py` fits `pydmd.DMD` or `pydmd.DMDc` on pairs of samples of the lifted
  runs. A dictionary is any scikit-learn transformer, and with one the states of the system are the observables. It
  reuses the private helpers of `_sparse_identification.py`.
- Particle trajectories are read by `_read_particle_snapshots` in `_base.py`. `PairInteraction` runs laws for every
  pair within a cutoff. k-body roles over moving neighbours are added by `add-many-body-neighbour-roles`.
- `ensemble` and `sweep` return an `ExperimentResult` whose runs hold the trajectories of every design point and
  replicate.

## Goals / Non-Goals

**Goals:**
- Every new option of the two functions is an argument of the function, and every result stays a frozen dataclass with a
  runnable `system`.
- `cluster_expansion` follows the method of 2611.08314 for the interaction terms, in two and three dimensions.

**Non-Goals:**
- Implicit SINDy, SINDy for PDEs, and constrained or stability-promoting optimizers. They can be added as more options
  later.
- Neural lifts of the Koopman operator. They need torch through pykoopman, which is not allowed.
- The self term of dACE for the agent's own past motion, the vertical frame under gravity, and the staged fit. The
  interaction terms are fitted alone. A later change can add the self term on the residual.
- Free-form symbolic regression with PySR, which needs Julia.

## Decisions

### Sparse identification

- **Weak form.** `weak=True` replaces the polynomial library with `pysindy.WeakPDELibrary` over the same
  `PolynomialLibrary`, on the time grid of every run, with `K` subdomains and test functions of degree `p` as
  arguments `subdomains` and `test_degree`. The coefficients are still those of the polynomial terms, so
  `_sparse_laws` builds the same rate laws. The errors are computed in the strong form, on finite differences, so they
  compare across options.
- **Ensemble.** `ensemble=n` wraps `STLSQ` in `pysindy.EnsembleOptimizer` with `bagging=True`, or with
  `library_ensemble=True` when `drop_terms` is set, seeded from `random_state`. `inclusion` is the share of
  `coef_list` that is nonzero per term. The median coefficients are the ones the laws use.
- **Parametric library.** `parameters` turns into `pysindy.ParameterizedLibrary` with the polynomial library of the
  states and a polynomial library of the parameters of degree `parameter_degree`. PySINDy takes the parameters as a
  control input `u` held constant over each run. The system declares every parameter with `ParameterSpec`, and the
  laws read it as a symbol. Feature names become products such as `rho x`.
- **Control inputs.** `inputs` splits the input columns off as `mode_decomposition` does with `_split_inputs`, and
  passes them as `u` to `SINDy.fit`. The laws read the inputs as adaptive parameters declared with the bounds of the
  observed inputs, through the same `_input_declarations` that mode decomposition uses. That helper moves to
  `_sparse_identification.py` so both import it downward.
- **Experiment results.** `_read_samples` gains a branch for `ExperimentResult`: it groups runs by design point and
  either averages the replicates frame by frame or keeps each. The branch is a private helper of `_base.py` because
  other analyses that take runs can reuse it.
- **Breaking result fields.** `SparseIdentificationResult` gains `inclusion` (NaN without an ensemble), `parameters`,
  and `inputs`. Results stay frozen and positional order changes, which is allowed in 0.x.

### Mode decomposition

- **Eigenfunctions.** The left eigenvectors come from `scipy.linalg.eig(operator, left=True)`, normalised against the
  modes. The result gains `eigenfunctions` (one column per eigenfunction, one row per observable),
  `eigenfunction_errors` on the held-out pairs, and `valid`, true where the error is below `eigenfunction_tolerance`.
  Evaluating an eigenfunction on new states is `dictionary.transform(states) @ eigenfunctions`, so no callable is
  stored.
- **Readout.** With a dictionary, the readout `C` is fitted by least squares from the lifted training samples to the
  variables. `_surrogate_system` takes observables, and the element declares one observable per variable as `C g`.
- **Delays.** `delays=d` stacks `d` past samples with every sample before lifting, as `pydmd.HankelDMD` does, and fits
  the same exact DMD, so control inputs and dictionaries keep working. The delayed copies are named `x_lag1` and so on.
  Stacking by hand over HankelDMD keeps one fitting path for every option.
- **Dictionary choice.** A list of dictionaries runs the fit once per candidate with `sklearn.base.clone` and keeps the
  smallest validation error. The result gains `dictionary_errors`. A search over hyperparameters, such as the bandwidth
  of `Nystroem`, is a list of candidates the user builds, so no search class is added.

### Cluster expansion

- **Library survey.** dACE.jl and ACEpotentials.jl are Julia packages, which the project does not call. No Python
  package expands vector laws over pairs of vectors per neighbour. e3nn is torch. So the basis is written in the core
  with NumPy, and the fit uses `sklearn.linear_model.BayesianRidge`, which maximises the evidence as the paper does.
- **Basis in 2D.** Positions and velocities are complex numbers. One-particle functions are `f_c(rho) P_n(y(rho))
  P_l(2|u|^2 - 1) rhat^m_r u^m_u` with the cutoff `f_c(rho) = (1 - rho^2 / R^2)^2`, the transformed distance `y` of the
  paper, and Legendre polynomials from `numpy.polynomial.legendre`. Centre functions are `P_p(2|v|^2 - 1) v^m`. A
  product is kept only if its charges add to 1, which makes it equivariant. Real coefficients make the law also
  equivariant under reflections, which is the default, and `chiral=True` keeps complex coefficients by fitting real and
  imaginary parts as separate columns.
- **Basis in 3D.** The heading frame of the paper: the velocity of the member gives `e3`, perpendicular parts are
  complex, and the same charge rule applies, with the perpendicular and axial accelerations fitted as two outputs. An
  agent at rest keeps the frame of its last nonzero velocity.
- **Pure body orders.** The sum over distinct neighbour tuples is expanded by inclusion and exclusion into products of
  pooled features, as in the paper, so each environment costs one pass over its neighbours per feature. `order` is at
  most 4, which the method needs and which keeps the set partitions small.
- **Degrees.** `degrees` maps each body order to its total polynomial degree, defaulting to those of the paper. The
  smoothness prior `1 + deg^4` scales the columns before the fit, which is the same as a weighted penalty.
- **Data.** Accelerations are finite differences of the recorded velocities, or the recorded rate of the velocity when
  the trajectory holds it. `validation` holds out whole runs to report the relative force error.
- **System.** The group gets one interaction per body order. Order 1 is a rate law of the member. Order 2 is a
  `PairInteraction` subclass whose law is the fitted 2-body sum. Orders 3 and 4 are the k-body neighbour interaction
  of `add-many-body-neighbour-roles`. The laws are written as symbolic polynomials of the one-particle features of the
  tuple, so the compiler shares the features between terms. These interactions are private classes of
  `_cluster_expansion.py`, as `_SurrogateElement` is of sparse identification.
- **Result.** `ClusterExpansionResult` holds `system`, `coefficients`, `basis` (a description of every basis function
  with its body order and degree), `alpha` and `lambda_` of the ridge, `contributions` (one array per body order),
  `training_error`, `validation_error`, and `predictive_std`.

### New system

- `online_offline_activity_network` lives in a new module `_online_offline.py` of `social/spreading`. It defines one
  `BaseSystem` with `form='stochastic'` or `form='mean_field'`. The stochastic form is a site group of individuals with
  a categorical state of the five compartments `S1` to `S5`, jump laws read from neighbour counts on a `NetworkSpace`,
  and a complete graph when `network=None`. The mean-field form is one element with the rates of equation (1) of the
  paper. Primitives: `jumps`, `discrete-states`, `networks`, `groups`, `interactions`, `rates`.

## Risks / Trade-offs

- [The weak library of PySINDy changes its arguments between releases] → The wrapper passes only `function_library`,
  `spatiotemporal_grid`, `K`, and `p`, which are stable since 1.7, and a test pins the Lorenz result.
- [Symbolic laws of a 4-body expansion with many terms compile slowly] → Laws are polynomials of shared features, the
  default degrees are moderate, and the inventory probe uses `order=3` for the free run.
- [The cluster expansion is new code in the core] → It is checked against exact equivariance, permutation invariance,
  and pure body orders on random environments before any fit.
- [Bird swarm has no cutoff] → The probe takes a cutoff larger than the swarm, so every pair is a neighbour.

## Open Questions

- Should `_input_declarations` and the other surrogate helpers shared by sparse identification, mode decomposition, and
  cluster expansion move to a private `_surrogate_systems.py` module of `core/analysis`, beside `_surrogates.py` of
  `surrogate_test`? This is a module split inside an existing package, not a new package, and can wait.
