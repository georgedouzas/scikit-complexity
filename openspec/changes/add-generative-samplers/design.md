# Design

## Context

Metropolis sampling is stated by an interaction whose `_prepare` sets an `ObjectiveLaw` of the local energy of a member
over a temperature, such as `SpinSamplingInteraction` of `skcomplex.systems.physics.common.spins`, and run by the
`MetropolisSolver` setting of the system (`core/engine/solvers/_metropolis.py`). Its `moves` map the dotted path of a
state to a move, `'flip'`, `'random-value'`, `'rotate'`, `'gaussian'`, `'cluster'`, or a function `move(values,
random)`, accepted by its `rule`. The compiler lowers the objective to `MonteCarloKernel`
(`core/engine/problems/_monte_carlo_kernel.py`), which runs one sweep per unit of time. Every move is local: the kernel
scores it by the change of the local objective of a member, read through the neighbour table of the lattice. The costs
of several objective laws add up. `ExpressionPrinter` prints expressions for NumPy and for JAX (`'jax'` target). jax,
optax, and diffrax are runtime dependencies. flowjax is not yet.

`add-learning-foundation` is a prerequisite. It adds Equinox as a runtime dependency and the private `_OptaxLoop` of
`GradientDescentSolver`, the one training loop of the library, which takes a function from parameters and a batch to a
loss and its gradient, with validation and early stopping. There is no `physics/field_theory` subfield. Lattice Yang-Mills
lives in `physics/waves` and the spin systems in `physics/statistical_mechanics`.

## Goals / Non-Goals

**Goals:**

- A general independence proposal in the Metropolis solver, of which a trained flow is one case.
- One function that trains a sampler on a Metropolis system and returns weights, diagnostics, and a runnable system.
- Exact energies for both scoring proposals and training, from one law.

**Non-Goals:**

- Discrete states, which need dequantization or discrete flows, and angles, which need flows on the circle.
- Transferable samplers across systems of different sizes, and molecular Boltzmann generators of particle systems,
  which have no Metropolis sweeps of members in the core.
- Gauge-equivariant flows for `lattice_yang_mills`.
- Mixing global and local moves on one state in one run. Two states or two runs can do it later.

## Decisions

- **Total energy per member.** `ObjectiveLaw` gains the keyword field `total`, an expression of the namespace of one
  member, written like the local objective with `neighbors(...)`, whose sum over the members is the total objective.
  The local objective counts every pair term fully for the change of one member, the total counts it half. Local moves
  keep reading the local objective, and only moves of whole configurations read `total`. Deriving the total from the
  local objective was rejected, since halving pair terms is ambiguous for terms of one member. A second objective law
  was rejected, since the costs of several objective laws add up. The compiler prints `total` with `ExpressionPrinter`
  over the neighbour table, and the kernel records the total as the observable `<interaction label>.energy`.
- **Proposal spec.** `ProposalSpec(sample, log_density, hits=1)` in `core/engine/solvers/_proposal_spec.py`, beside
  `MetropolisSolver`, which reads it: `sample(rng, n)` returns `n` configurations shaped like the state and their log
  densities, and `log_density(x)` returns the log densities of given configurations. Its `_check` checks that both are
  callable and `hits` is a positive integer. A `MetropolisSolver` whose `moves` give a `ProposalSpec` for a state lowers
  it to the new private path `_propose_configuration` of `MonteCarloKernel`, which keeps the log density of the current
  configuration and accepts by the independence Metropolis rule. The kernel records the acceptance rate as
  `<interaction label>.acceptance`. A new kernel class was rejected because the targets, temperature, and arguments of
  the existing kernel are the same.
- **Function.** `core/analysis/_generative_sampler.py` holds `GenerativeSamplerResult` and
  `generative_sampler(system, *, kind='coupling', training='reverse', samples=None, n=4096, layers=8, width=64,
  epochs=500, batch=256, learning_rate=1e-3, validation=0.0, patience=None, observables=None, hits=1,
  random_state=None)`. It clones and prepares the system, finds its one `ObjectiveLaw` with a `total` and the
  `MetropolisSolver` of its settings, reads the target shape and the intensity, the inverse of the temperature of the
  solver, and refuses a state whose allowed values are not real, naming the state. With `training='forward'` and no
  `samples` it draws them from the system's own chain with `simulate`. It returns `system`, `sampler`, `samples`,
  `log_weights`, `weights`, `effective_sample_size`, `relative_effective_sample_size`, `free_energy`, `estimates`
  (mean and standard error per observable), and `losses`. It lives in `core/analysis` beside `sparse_identification`,
  since both learn a runnable system. `core/experiments` was considered, since the function runs simulations, but it
  holds no learning function.
- **Flows.** `kind='coupling'` is `flowjax.flows.coupling_flow` with an `Affine` transformer and `kind='spline'` with a
  `RationalQuadraticSpline` transformer, both on a standard normal base over the flattened lattice. flowjax flows are
  Equinox modules, so they are trained on the shared `_OptaxLoop` of `add-learning-foundation` and not by
  `flowjax.train`. Reverse training minimises the reverse divergence `E_q[log q + intensity * E]` on batches of draws
  from fresh keys, whose energy is the `total` printed for JAX and summed over the neighbour table. Forward training
  minimises the negative log density of the samples, with the share `validation` of them held out for early stopping
  after `patience` epochs.
- **Flow matching.** `kind='flow_matching'` is an Equinox MLP velocity field `v(x, t)` trained on the shared loop on
  the conditional optimal-transport loss between base draws and samples, as in 2406.14426. Samples and log densities
  come from integrating `x' = v` and `log q' = -div v` with `diffrax.diffeqsolve` (`Tsit5`, adaptive steps), with the
  divergence as the trace of `jax.jacfwd`, exact for the lattice sizes of the probes. The private class
  `_FlowMatchingSampler` exposes `sample` and `log_density` like a flowjax distribution, so the result and the proposal
  do not branch on the kind.
- **Weights and free energy.** `log w = -intensity * E - log q`. The normalised weights use `logsumexp`. The effective
  sample size is Kish's `(Σw)² / Σw²`. The free energy is `-(logsumexp(log w) - log n) / intensity`. A weighted mean
  has the standard error of the self-normalised estimator.
- **Learned system.** `system` is a clone of the system whose solver setting is set through `set_params` to
  `MetropolisSolver(moves={path: ProposalSpec(sampler.sample, sampler.log_density, hits)}, rule='metropolis')`, with the
  temperature of the original solver.
- **System.** `LatticePhi4System` goes in `systems/physics/statistical_mechanics/_phi4.py`, a `BaseSystem` with `size`,
  `dims`, `m2`, `lam`, `width` of the local Gaussian move, `initial`, and `random_state`, whose `_prepare` sets
  `system_`. It holds one `Element` with the real value `phi`, placed with `space=` on a periodic `LatticeSpace`, and
  one interaction of the module whose `_prepare` sets `ObjectiveLaw(local, phi, total=site)` with the local objective
  `Σ_y (φ_y - φ_x)² / 2 + m² φ_x² / 2 + λ φ_x⁴ / 24` over the von Neumann neighbourhood and the site energy
  `Σ_y (φ_y - φ_x)² / 4 + m² φ_x² / 2 + λ φ_x⁴ / 24`, with `m2` and `lam` parameters of that interaction. Its tuned
  solver is `MetropolisSolver(moves={'field.phi': 'gaussian'}, width=width)` at temperature 1.
- **Dependencies.** `flowjax>=19.2.1` (19.2.1 released 2026-11-07) joins the runtime dependencies. Equinox comes with
  `add-learning-foundation`. flowjax needs equinox, jax, jaxtyping, lineax, optax, paramax, and tqdm, none of which
  conflicts with the current pins. flowjax warns that its API may change. Only its flows and transformers are used,
  since training runs on the shared loop, so the pin has a lower bound and the tests cover the two flow constructors
  used.
- **Placement.** `ProposalSpec`, the field `total`, and the proposal path of `MonteCarloKernel` are step 2, a value of
  a solver, a field of a core law kind, and a kernel the engine handles by type, in `skcomplex.core.engine.solvers`,
  `skcomplex.core.components.laws`, and `core/engine/problems`. Domains: statistical physics and lattice field theory,
  Bayesian inference, molecular sampling in chemistry, and combinatorial models in computer science.
  `generative_sampler` is step 2, an analysis of a Metropolis system in `skcomplex.core.analysis`, for the same
  domains. `LatticePhi4System` is step 3, one system, in `physics/statistical_mechanics/_phi4.py`.

### Fit with the redesigned core

Written after `redesign-core-api` landed on 2026-11-22. The earlier draft put the proposal on `SamplingInteraction`,
the energy on `SampleLaw`, and the spec in `skcomplex.core.specs`, all of which are gone. Sampling is now an
`ObjectiveLaw` stated by an interaction and run by the `MetropolisSolver` setting of a system, whose `moves` already
hold the local moves, so the proposal joins them and the total energy joins the objective law.

## Risks / Trade-offs

- [Mode collapse of reverse training in a broken phase] → The probes stay near the symmetric phase, the result reports
  the effective sample size so collapse is visible, and forward training on chain samples is offered.
- [flowjax API changes] → Only `coupling_flow` and the two transformers are used, behind one private module, so a
  change touches one place.
- [Exact divergence cost of flow matching grows with the square of the sites] → The user guide states the cost, and
  the function raises `AnalysisError` above 1,024 sites for flow matching.
- [Training time in tests] → Tests use 4 × 4 lattices and few epochs, and mark the 8 × 8 probes as slow.

## Open Questions

- The total energy of a member is a new field `total` of `ObjectiveLaw`, a change to one of the seven core law kinds.
  The alternatives are an `energy` argument of `MetropolisSolver` given as an observable, as the temperature is already
  a setting of the solver, or a `ProposalSpec` that carries the energy itself. To be decided before the proposal path is built.
