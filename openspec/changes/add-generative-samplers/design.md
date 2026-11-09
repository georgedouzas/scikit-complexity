# Design

## Context

Monte Carlo sampling is declared by `SamplingInteraction` (`core/components/interactions/_sampling.py`), which returns a
`SampleLaw` (`core/components/laws/_sample.py`) with an `ObjectiveSpec`, a `MoveSpec`, and a `SamplerSpec`. The lattice
pass (`core/engine/compile/passes/_lattices.py`) lowers it to `MonteCarloKernel`
(`core/engine/problems/_monte_carlo_kernel.py`), which `MonteCarloSolver` runs one sweep per unit of time. Every move is
local: the kernel scores it by the change of the local objective of a site, read through the neighbour table of the
lattice. `ExpressionPrinter` prints expressions for NumPy and for JAX (`'jax'` target). jax, optax, and diffrax are
runtime dependencies. flowjax and equinox are not yet. There is no `physics/field_theory` subfield. Lattice Yang-Mills
lives in `physics/waves` and the spin systems in `physics/statistical_mechanics`.

## Goals / Non-Goals

**Goals:**

- A general independence proposal in the Monte Carlo kernel, of which a trained flow is one case.
- One function that trains a sampler on a Monte Carlo system and returns weights, diagnostics, and a runnable system.
- Exact energies for both scoring proposals and training, from one declaration.

**Non-Goals:**

- Discrete states, which need dequantization or discrete flows, and angles, which need flows on the circle.
- Transferable samplers across systems of different sizes, and molecular Boltzmann generators of particle systems,
  which have no Monte Carlo solver in the core.
- Gauge-equivariant flows for `lattice_yang_mills`.
- Mixing global and local moves inside one interaction. Two interactions on one state can do it later.

## Decisions

- **Total energy per site.** `SamplingInteraction` gains `energy`, a function of the namespace of one site, written
  like the local objective with `neighbors(...)`, whose sum over the sites is the total objective. The local objective
  counts every pair term fully for the change of one site, the energy counts it half. Deriving the total from the local
  objective was rejected, since halving pair terms is ambiguous for terms of one site. `SampleLaw` carries the energy,
  the lattice pass prints it with `ExpressionPrinter` over the neighbour table, and the kernel records the total as the
  observable `<label>.energy`.
- **Proposal spec.** `ProposalSpec(sample, log_density, hits=1)` in `core/specs/_proposal_spec.py`: `sample(rng, n)`
  returns `n` configurations shaped like the state and their log densities, and `log_density(x)` returns the log
  densities of given configurations. It checks that both are callable and `hits` is a positive integer. A
  `SamplingInteraction` whose `move` is a `ProposalSpec` lowers to the new private path `_propose_configuration` of
  `MonteCarloKernel`, which keeps the log density of the current configuration and accepts by the independence
  Metropolis rule. The kernel records the acceptance rate as `<label>.acceptance`. A new kernel class was rejected
  because the targets, intensity, and arguments of the existing kernel are the same.
- **Function.** `core/analysis/_generative_sampler.py` holds `GenerativeSamplerResult` and
  `generative_sampler(system, *, kind='coupling', training='reverse', samples=None, n=4096, layers=8, width=64,
  epochs=500, batch=256, learning_rate=1e-3, observables=None, hits=1, random_state=None)`. It prepares the system,
  finds its one sampling interaction with an energy, reads its target shape and the value of its intensity, and refuses
  a state whose domain tag is not real, naming the tag. With `training='forward'` and no `samples` it draws them from
  the system's own chain with `simulate`. It returns `system`, `sampler`, `samples`, `log_weights`, `weights`,
  `effective_sample_size`, `relative_effective_sample_size`, `free_energy`, `estimates` (mean and standard error per
  observable), and `losses`. It lives in `core/analysis` beside `sparse_identification`, since both learn a runnable
  system. `core/experiments` was considered, since the function runs simulations, but it holds no learning function.
- **Flows.** `kind='coupling'` is `flowjax.flows.coupling_flow` with an `Affine` transformer and `kind='spline'` with a
  `RationalQuadraticSpline` transformer, both on a standard normal base over the flattened lattice. Reverse training is
  `flowjax.train.fit_to_key_based_loss` with the reverse divergence `E_q[log q + intensity * E]`, whose energy is the
  site energy printed for JAX and summed over the neighbour table. Forward training is `flowjax.train.fit_to_data`.
- **Flow matching.** `kind='flow_matching'` is an equinox MLP velocity field `v(x, t)` trained with optax on the
  conditional optimal-transport loss between base draws and samples, as in 2406.14426. Samples and log densities come
  from integrating `x' = v` and `log q' = -div v` with `diffrax.diffeqsolve` (`Tsit5`, adaptive steps), with the
  divergence as the trace of `jax.jacfwd`, exact for the lattice sizes of the probes. The private class
  `_FlowMatchingSampler` exposes `sample` and `log_density` like a flowjax distribution, so the result and the proposal
  do not branch on the kind.
- **Weights and free energy.** `log w = -intensity * E - log q`. The normalised weights use `logsumexp`. The effective
  sample size is Kish's `(Σw)² / Σw²`. The free energy is `-(logsumexp(log w) - log n) / intensity`. A weighted mean
  has the standard error of the self-normalised estimator.
- **Learned system.** `system` is a clone of the prepared composite whose sampling interaction takes, through
  `set_params`, `move=ProposalSpec(sampler.sample, sampler.log_density, hits)`, with the rule `'metropolis'`.
- **System.** `LatticePhi4System` goes in `systems/physics/statistical_mechanics/_phi4.py`, with `size`, `dims`, `m2`,
  `lam`, `width` of the local Gaussian move, `initial`, and `random_state`. It builds a `LatticeElement` of one real
  state `phi` on a periodic `LatticeSpace` and a `SamplingInteraction` with the local objective
  `Σ_y (φ_y - φ_x)² / 2 + m² φ_x² / 2 + λ φ_x⁴ / 24` over the von Neumann neighbourhood and the energy
  `Σ_y (φ_y - φ_x)² / 4 + m² φ_x² / 2 + λ φ_x⁴ / 24`, at intensity 1.
- **Dependencies.** `flowjax>=19.2.1` (19.2.1 released 2026-11-07) and `equinox>=0.13.8` (0.13.8 released
  2026-05-05) join the runtime dependencies. flowjax needs equinox, jax, jaxtyping, lineax, optax, paramax, and tqdm,
  none of which conflicts with the current pins. flowjax warns that its API may change, so the pin has a lower bound and
  the tests cover the two training calls used.

## Risks / Trade-offs

- [Mode collapse of reverse training in a broken phase] → The probes stay near the symmetric phase, the result reports
  the effective sample size so collapse is visible, and forward training on chain samples is offered.
- [flowjax API changes] → Only `coupling_flow`, the two transformers, `fit_to_data`, and `fit_to_key_based_loss` are
  used, behind one private module, so a change touches one place.
- [Exact divergence cost of flow matching grows with the square of the sites] → The user guide states the cost, and
  the function raises `AnalysisError` above 1,024 sites for flow matching.
- [Training time in tests] → Tests use 4 × 4 lattices and few epochs, and mark the 8 × 8 probes as slow.

