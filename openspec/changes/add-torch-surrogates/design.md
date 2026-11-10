## Context

See proposal.md for the motivation. This change sits on two drafted changes. `add-bayesian-calibration` gives
`calibrate` its `method` argument (`'local'`, `'global'`, `'abc_smc'`, `'mcmc'`, `'variational'`), priors read from the
bounds, the private `_CalibrationObjective` in `core/experiments/_calibration.py`, posterior fields of
`CalibrationResult`, and `posterior_coverage` in `core/experiments/_coverage.py`. `add-neural-particle-simulators`
gives `force_matching` in `core/analysis/_force_matching.py`, with `kind='linear'` and `kind='network'`, and particle
mappings in `coarse_grain`. Its learned systems run a fitted Python map through a `RateLaw` or `UpdateLaw` with a
vectorised `CallSpec`, as `reservoir` does with `_ReservoirStep` in `core/analysis/_reservoirs.py`.

Pair potentials are symbolic today: `PairInteraction` with a potential, into which `move-recipes-to-systems` merges the
former `PairCouplingInteraction`, and `GradientInteraction` take Sage expressions, and `LammpsSolver` in
`core/engine/solvers/_lammps.py` translates them to LAMMPS styles. Interactions it cannot translate that only change
velocities already run through `fix external`, evaluated by the framework. The installed LAMMPS wheel holds ML-IAP but
not the PYTHON package, so `pair_style mliap unified` and `pair_style mace` are not available, while `fix external` with
a callback is. The field systems `GrayScottSystem` (`chemistry/pattern_formation`), `KuramotoSivashinskySystem`
(`physics/fluids`), `HeatDiffusionSystem` (`physics/heat`), and `AllenCahnSystem` (`physics/pattern_formation`) run on
lattices with the spectral and stiff steppers. `lyapunov(kind='largest')` and `scaling(kind='correlation-dimension')`
measure series. `chain_shape` measures chains.

The general-mechanisms spec forbids field terms in core names, so a core class may say potential and kind but not
force, energy, atom, or species.

## Goals / Non-Goals

**Goals:**
- Wrap the torch libraries of the papers behind the existing main public methods: `calibrate`, analysis functions
  with frozen results, interactions, and solvers. No estimator classes.
- Every learned or pretrained object ends as a runnable `system` or as a field of a frozen result.
- Tests run offline, on CPU, in seconds, with small untrained or tiny networks.

**Non-Goals:**
- GPU tuning. Every function takes `device='cpu'` and accepts `'cuda'` or `'mps'`, but CI runs on CPU.
- Panda (2505.13755), which is not on PyPI. Chronos covers the zero-shot forecasts of 2409.15771.
- Neural operators on meshes (GINO). Only regular lattices.
- Training foundation potentials. `MaceWaterSystem` loads a published model, and only coarse-grained networks are
  trained here.
- Torch backends for the engine. Torch runs only inside the wrapped libraries and their callbacks.

## Decisions

### Dependencies

Checked on PyPI on 2026-11-08, with the requirements each release declares.

| Library | Version | Released | Status |
| --- | --- | --- | --- |
| torch | 2.15.0 | 2026-10-28 | Added |
| sbi | 0.27.0 | 2026-08-04 | Added |
| neuraloperator | 2.0.0 | 2025-10-22 | Added |
| chronos-forecasting | 2.3.2 | 2026-09-08 | Added |
| mace-torch | 0.3.17 | 2026-10-27 | Added |
| deepxde | 1.15.1 | 2026-11-02 | Added |
| torchmd-net | 3.1.0 | 2026-10-19 | Dropped |

- **All fit the current pins.** numpy: chronos-forecasting needs `numpy<3`, matscipy (from mace-torch) needs
  `numpy>=2.0`, and the rest are unbounded, so `numpy>=2.1.0` holds. scikit-learn: sbi, skorch, and scikit-optimize
  0.10.2 (from deepxde) accept `scikit-learn>=1.5.1`. matplotlib: the project has no runtime pin, and sbi,
  neuraloperator, deepxde, mace-torch, and ase need it unpinned (ase `>=3.5.2`), so it becomes a transitive runtime
  dependency. torch needs `sympy>=1.13.3` and `networkx>=2.5.1`, which the passagemath stack and `networkx>=3.3`
  allow. Every library supports Python 3.12 to 3.14, and the compiled transitive packages (matscipy, lmdb, orjson) ship
  cp314 wheels for macOS arm64 and Linux x86_64.
- **torchmd-net is dropped.** Its 3.1.0 release is an sdist only. Installing it compiles torch C++ and CUDA extensions,
  which needs a compiler and torch at build time and fails under the isolated builds of pip and pdm. It also pulls
  `warp-lang` (28 to 150 MB per platform), `torch_geometric`, and `lightning`. Its role, a learned force field fitted
  by force matching (2310.18278), is taken by mace-torch through `force_matching(kind='equivariant')`, so one torch
  potential library serves both the atomistic and the coarse-grained probes.
- **neuraloperator does not declare torch** and pulls `wandb`, `tensorly`, `tensorly-torch`, `configmypy`, `zencfg`,
  and `h5py`. torch is declared by this project, so the wrapper never relies on it implicitly.
- **Stale transitive packages.** nflows 0.14 (2020-12, from sbi), tensorly-torch 0.5.0 and configmypy 0.2.0
  (2024-06, from neuraloperator), e3nn 0.4.4 (2021-12, pinned exactly by mace-torch), torch-ema 0.3 (2021-11, from
  mace-torch), and scikit-optimize 0.10.2 (2024-06, from deepxde) are past the 24-month limit of
  `dependency-freshness`. They are not direct dependencies. The Project Profile of the constitution records them as
  exceptions with the next review at the next minor release, since dropping them means dropping the library the
  papers use.
- **e3nn is pinned to 0.4.4 by mace-torch.** No other dependency needs e3nn today, so nothing conflicts, but a future
  dependency on e3nn 0.5 or later cannot be added while mace-torch keeps the pin. Recorded as a risk.
- **Install size and CPU-only wheels.** The torch wheel is 139 MB on macOS arm64, which runs on CPU and MPS. On Linux
  the PyPI wheel is about 540 MB and pulls CUDA 13 libraries (`cuda-toolkit`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`,
  `nvidia-cusparselt-cu13`, `nvidia-nvshmem-cu13`) and `triton`, several gigabytes in all. There is no torch wheel for
  Intel macOS, so the classifiers and the install guide say macOS on Apple silicon. Package metadata cannot choose the
  CPU variant, so: `pyproject.toml` adds a `[[tool.pdm.source]]` named `pytorch-cpu` at
  `https://download.pytorch.org/whl/cpu` with `include_packages = ["torch"]`, so the lock file, nox, and CI install
  the CPU wheel (about 190 MB on Linux x86_64). The install guide tells users without a GPU to install torch from the
  CPU index before the package, and users with a GPU to keep the default wheel.

### Neural calibration

- **Methods of the same argument.** `'npe'`, `'nle'`, and `'nre'` join the table of method and loss pairs of
  `add-bayesian-calibration`, each accepting only `loss='squares'` there, read as the summaries being compared. They
  map to `sbi.inference.NPE`, `NLE`, and `NRE`. Alternative: a separate `neural_calibrate`, rejected because the result,
  the priors, and the objective are shared.
- **Simulator.** `_CalibrationObjective` gains a method that maps a batch of parameter vectors to summaries. It runs
  the batch through the executors of `core/experiments/_runs.py` and returns a float32 torch tensor. Runs that fail
  return NaN, and sbi drops them.
- **Priors.** The SciPy priors of the bounds map to `torch.distributions` (`Uniform`, `Normal`, `LogNormal`, `Beta`,
  `Gamma`, `Exponential`) joined by `sbi.utils.MultipleIndependent`. Any other family raises `ValidationError` naming
  the supported ones.
- **Sampling.** NPE samples the flow directly, with rejection of draws outside the prior support. NLE and NRE sample
  through sbi's MCMC with the slice sampler, seeded from `random_state` by `torch.manual_seed` inside a
  `torch.random.fork_rng` block, so the caller's torch state is unchanged.
- **Amortization and rounds.** `estimator` holds the trained sbi posterior together with the parameter names and the
  summary length. With `budget=0` the call only conditions it on the new data. With `rounds > 1` the proposal of every
  later round is the posterior at the observed data, as in sequential NPE (2601.07413, 2411.17337), and the result is
  marked as tied to that data. A pretrained estimator with a positive budget continues training on new runs from its
  posterior, which is the test-time adaptation of 2601.07413 in its simplest form.
- **Embeddings.** `embedding='mlp'` and `'cnn'` use `sbi.neural_nets.embedding_nets.FCEmbedding` and `CNNEmbedding`
  on the series at the data times. Learned summaries are what 2508.12939 recommends when no expert summary exists.
- **Diagnostics.** The training and validation losses come from the inference object. `posterior_coverage` with a
  neural method trains once on `budget` runs and conditions on every truth, and adds the TARP curve from
  `sbi.diagnostics.run_tarp` and the ranks from `run_sbc`. The shared `predictive` checks of change 1 apply unchanged.
- **Result.** `CalibrationResult` gains `estimator: object | None`. Pickling the result pickles the estimator, as sbi
  supports.

### Neural operators

- **Function.** `neural_operator(source, *, variables=None, kind='fno', lead=1, modes=16, width=64, layers=4,
  validation=0.2, epochs=200, batch_size=16, learning_rate=1e-3, horizon=0, device='cpu', random_state=None)` in
  `core/analysis/_neural_operator.py`. `kind` is `'fno'`, `'tfno'`, or `'uno'`, from `neuralop.models`. The source is
  a trajectory, a list of trajectories, or the result of a `sweep` or `ensemble` that holds trajectories.
- **Training.** Pairs of frames `lead` samples apart, stacked over runs, with fields as channels and swept parameters
  as constant extra channels. A plain torch loop with Adam and the relative L2 loss of `neuralop.losses.LpLoss`.
  Alternative: neuraloperator's `Trainer`, rejected because it is tied to its config files and to wandb logging.
- **System.** As in `reservoir`, a private `_OperatorStep` holds the network and its normalisation, and the result's
  `system` is one field element on a `LatticeSpace` of the source's geometry under `DiscreteTime`, with one `UpdateLaw`
  per field through a vectorised `CallSpec`. Swept parameters become model parameters of the element, read by the step.
  The system takes `shape` so a Fourier operator runs on another grid, which is the discretisation convergence of
  2412.10354.
- **Result.** `NeuralOperatorResult` holds `system`, `network`, `variables`, `parameters`, `lead`, `dt`,
  `training_error`, `validation_error`, and `forecast`.

### Pretrained forecasts

- **Function.** `pretrained_forecast(source, horizon, *, variables=None, context=512, model='amazon/chronos-bolt-small',
  quantiles=(0.1, 0.5, 0.9), truth=None, threshold=30.0, device='cpu')` in `core/analysis/_pretrained_forecast.py`,
  with `chronos.BaseChronosPipeline.from_pretrained`. Variables are forecast one by one, channel-independent, as in
  2409.15771. Chronos-Bolt small is the default because it is fast on CPU and permissively licensed. Any Chronos or
  Chronos-2 name or path is accepted.
- **Comparison.** With `truth`, the symmetric mean absolute percentage error per step and the valid prediction time,
  the first step where it exceeds `threshold` percent (2409.15771). The attractor statistics reuse
  `lyapunov(kind='largest')` and `scaling(kind='correlation-dimension')` on forecast and truth, so no measure is
  duplicated.
- **Weights and the network.** A named model downloads into the Hugging Face cache on first use. When the hub is
  offline and the model is missing, the hub error is caught and re-raised as `AnalysisError` naming the model and the
  cache. Tests never reach the network. A test fixture builds a tiny Chronos-Bolt model from a config with random
  weights, saves it to a temporary directory, and passes its path as `model`. The nox test sessions set
  `HF_HUB_OFFLINE=1`. One slow test uses the real default model and is skipped unless it is already in the cache. The
  gallery example downloads the default model once, which the docs build caches.
- **Result.** `PretrainedForecastResult` holds `forecast`, `quantiles`, `model`, `context`, and with `truth` also
  `error`, `valid_time`, and `attractor`, a mapping of the largest exponent and the correlation dimension of forecast
  and truth.

### Equivariant potentials

- **Core primitive.** Both new systems drive one new core interaction, `EquivariantPotentialInteraction`, in
  `core/components/interactions/_equivariant_potential.py`. It is the numeric counterpart of `GradientInteraction`:
  arguments `label`, `target`, `states`, `network` (a path to a saved model or a loaded torch module), `kinds` (one
  integer per member), and `divisor`. Its law is one `RateLaw` on the highest derivative of the states, with a
  vectorised `CallSpec` that builds the neighbour graph within the network's cutoff by the minimum image, evaluates the
  potential, and returns minus its autograd gradient over the divisor. The potential is recorded as an observable.
  Alternative: a symbolic `GradientInteraction` potential, rejected because a network is not an expression, as change 7
  found for its Hamiltonian network.
- **Field case.** `MaceInteraction` in `physics/common/particles/_mace.py` is a subclass that takes chemical symbols
  and a published model name (`'mace-mp-0-small'` by default) and loads it with `mace.calculators.mace_mp`, mapping
  symbols to kinds. Both new systems are physics, so the case lives in the field's `common`.
- **Evaluation.** The network runs in float64 on CPU by default, so energy drift and the LAMMPS comparison meet tight
  tolerances. MACE takes the positions, the cell, the periodic flags, and one-hot kinds in its `AtomicData` layout,
  built from NumPy arrays without ASE in the loop.
- **LAMMPS.** `LammpsSolver` treats the interaction as a `fix external` interaction. Its callback reads the positions,
  calls the same `CallSpec`, and writes the gradient back as per-atom values, while LAMMPS keeps the integrator, the
  thermostat, and the translated pair styles. Alternatives: `pair_style mace` and `pair_style mliap unified`, which
  need a LAMMPS built with libtorch or with the PYTHON package. The wheel has neither.
- **Coarse-grained fits.** `force_matching(kind='equivariant')` builds a small MACE model (`mace.modules.MACE`, two
  interaction layers, few channels, cutoff from the argument) with one kind per bead type and trains it on the mapped
  forces with a force-only loss in a plain torch loop, beside the repulsive prior `(σ / r)^12` of change 7's network
  kind. The result system uses `EquivariantPotentialInteraction` for the network and a `PairInteraction` with a
  potential for the prior. Alternative: mace-torch's training script, rejected because it reads files and a command
  line.

### Physics-informed solver

- **Solver.** `PinnSolver` in `core/engine/solvers/_pinn.py`, a solver of field systems with rate laws on rectangular
  lattices in one or two dimensions, with arguments `layers=(64, 64, 64, 64)`, `fourier_features=0`, `iterations=10000`,
  `refine=True` (L-BFGS after Adam), `collocation=4096`, `device='cpu'`, and `random_state=None`. It uses deepxde's
  `TimePDE`, `GeometryXTime`, `IC`, `DirichletBC`, and `PeriodicBC`, and returns frames at the requested times on the
  system lattice.
- **Backend.** deepxde reads its backend once at import from `DDE_BACKEND`. The solver module sets
  `os.environ['DDE_BACKEND'] = 'pytorch'` before it imports deepxde, and raises `SolverError` if deepxde was already
  imported with another backend. The constitution forbids hidden global state, so the solver docs say this
  explicitly.
- **Residuals from the declared laws.** A private `_PinnResidual` walks the compiled field laws of the problem and
  prints each rate expression to torch through a subclass of `ExpressionPrinter` of
  `core/engine/problems/_expression_printer.py`. `partial(u, k)` becomes `dde.grad.jacobian` and `laplacian(u)` the sum
  of `dde.grad.hessian` diagonals. Noise, jump, rule, callable, and algebraic laws are rejected in the solver's support
  check, as `LammpsSolver` rejects unsupported laws.
- **Fourier features** follow 2308.08468 as a fixed random feature layer applied to the space inputs through
  deepxde's `apply_feature_transform`. Causal weighting is left out, see the open questions.
- **Probes.** `HeatDiffusionSystem` and `AllenCahnSystem` in one dimension for the tests. A short window of
  `GrayScottSystem` and of `KuramotoSivashinskySystem` in the gallery, compared with the spectral run.

### Systems

- **`MaceWaterSystem`** in `physics/statistical_mechanics/_mace_water.py`, next to the Lennard-Jones gases. Arguments
  `n_molecules=32`, `density=0.997` (g/cm³), `temperature=300.0`, `friction=0.01` (1/fs), `model='mace-mp-0-small'`,
  `random_state`, and `label`. Molecules start on a cubic lattice with the experimental geometry. One group of atoms
  with masses per kind, a `MaceInteraction`, and the Langevin thermostat of `physics/common/particles`. MACE-MP-0 is the
  model of 2401.00096 and is MIT licensed, unlike MACE-OFF. The tag set includes `stochastic` through the thermostat.
- **`CgProteinFoldingSystem`** in `physics/soft_matter/_cg_proteins.py`. Arguments `resolution='coarse'`,
  `temperature`, `network=None`, `random_state`, and `label`. The fine resolution is a native-centric model of
  chignolin with a backbone and a side-chain bead per residue, harmonic bonds and angles, and Lennard-Jones 12-10
  native contacts, run with Langevin dynamics. It is the reference that the coarse model is fitted to, standing in for
  the all-atom data of 2310.18278. The coarse resolution keeps one bead per residue with bond, angle, and repulsion
  priors, and adds an `EquivariantPotentialInteraction` when `network` is given. The native coordinates of the ten
  residues are a small data file of the module, read with `importlib.resources.files`. No Biopython is needed, so the
  molecular-structures extras stay optional, and `chain_shape` measures both resolutions.

## Risks / Trade-offs

- [The install grows by several gigabytes on Linux with the default CUDA wheel] → the CPU source in the lock file, the
  install guide, and the size stated in the changelog.
- [Importing torch adds about a second to `import skcomplex.core.analysis`] → the torch modules import it at their top
  like every other library. A task measures the import time before and after and records it in the changelog.
- [torch and JAX both start thread pools] → the tests cap threads with `OMP_NUM_THREADS` and `torch.set_num_threads`
  in the nox sessions, as for NUMBA and BLAS.
- [sbi trains slowly for large budgets] → the tests use budgets of a few hundred runs and small flows, and mark the
  calibration scenarios of the spec as slow.
- [Pretrained forecasts depend on a remote hub] → offline tests with a tiny saved model, and a clear `AnalysisError`.
- [The e3nn pin of mace-torch blocks newer e3nn] → recorded. If another library needs e3nn 0.5 or later, the fallback
  is to keep mace-torch and wait for its release, since no other library offers MACE-MP-0.
- [Stale transitive packages] → recorded exceptions with a review date in the Project Profile.
- [PINN accuracy depends on training] → the tests use smooth short problems with loose tolerances. The solver docs say
  PINNs are not a replacement for the spectral steppers on chaotic fields.
- [The LAMMPS callback copies positions every step] → accepted. The network evaluation dominates the cost.
- [`CalibrationResult` gains a field] → breaking for positional construction only, recorded in the changelog.

## Migration Plan

No data migration. Calls without the new methods keep their results. The changelog names the new field of
`CalibrationResult`, the new runtime dependencies, the install size, and the CPU-only install steps.

## Open Questions

- Causal training weights of 2308.08468 are not part of deepxde. They can be added later as a solver option without
  changing the specs.
- Chronos-2 handles several variables jointly. If it proves better than channel-independent forecasts on the dysts
  systems, it can become the default model without changing the specs.
