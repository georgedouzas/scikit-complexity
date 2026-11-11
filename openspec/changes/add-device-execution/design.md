## Context

- `JaxBackend` in `core/engine/backends/_jax.py` writes the drift of ODEs and the update of maps as pure JAX functions,
  with member reads and reductions, for systems without a space. `_get_forward` wraps each in `jax.jit` and converts
  between NumPy and JAX at every call, inside `jax.enable_x64(True)`. It fills no `accelerated` loops, so a solver calls
  the forward function once per internal step from Python.
- `JaxBackend.unroll` already runs a whole map or ODE run as one `jax.lax.scan`, with JAX keys for the uniform and
  normal draws of maps. Calibration and the learning compiler use it. It is the seed of the loops of this change.
- `NumbaBackend` fills `AcceleratedSpec` with the loops of `_numba_loops.py`, written as source text and compiled with
  `njit`. The solvers `RK4Solver`, `VelocityVerletSolver`, `Yoshida4Solver`, `EulerMaruyamaSolver`, `GillespieSolver`,
  and the map solver call these loops once per frame.
- The automatic backend choice in `core/engine/compile/_base.py` picks Numba for groups of 1000 members or more and
  NumPy otherwise. The Numba loops run members in parallel with `prange` from that size on.
- `GradientDescentSolver` has `precision='float64'` or `'float32'`, the naming this change reuses.
- No backend has a device setting, no test runs on a GPU, and the `benchmark` marker exists for timing tests.

## Goals / Non-Goals

**Goals:**
- Run any problem the JAX backend lowers on a chosen device, in a chosen precision, with every internal step of a frame
  on the device.
- Lower noise and jump laws to JAX, so stochastic systems run on a GPU.
- A test marker and a CI strategy for GPU tests, and benchmarks against the current backends.
- A recorded decision on further compute backends.

**Non-Goals:**
- Batches of runs as one device program. `add-batched-runs` does it on top of these loops.
- Spaces, neighbourhoods, networks, and agents on the device. `add-large-system-kernels` does it.
- Several devices for one run or one batch. `add-multi-device-sharding` does it.
- Choosing the JAX backend automatically when a GPU is present. The automatic choice stays as it is, so a script gives
  the same numbers on every machine. A user asks for the GPU by name.
- Fractional noise on the device, whose Davies-Harte draws need the whole path in advance.
- Checkpoint and resume, which the user did not ask for.

## Decisions

### Placement

| Component | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| `device` and `precision` of `JaxBackend` | 2, settings of a backend the engine runs by type | `core/engine/backends/_jax.py` | every field with large runs: physics (oscillators, particles), epidemiology (stochastic outbreaks), neuroscience (spiking networks), ecology, finance (Monte Carlo paths) |
| JAX loops `iterate`, `rk4`, `velocity_verlet`, `yoshida4`, `euler_maruyama`, `gillespie` | 2, backend lowerings that the solvers call through `AcceleratedSpec` | new private module `core/engine/backends/_jax_loops.py` | the same fields, through every solver that reads `accelerated` |
| JAX lowering of noise and jump laws | 2, backend lowerings of law kinds | `core/engine/backends/_jax.py` | stochastic chemistry, epidemiology, finance, population genetics |
| `gpu` marker, `gpu` session, and GPU workflow | not a part of the package | `tests/conftest.py`, `noxfile.py`, `.github/workflows/gpu.yml`, `.github/workflows/release.yml` | none |

No new package and no new concept. A device is a setting of a backend, as `parallel` is a setting of the Numba
backend, and the loops fill a spec the solvers already read.

### Dependencies

Checked on PyPI on 2026-11-11.

| Library | Version | Released | Licence | Status |
| --- | --- | --- | --- | --- |
| jax, jaxlib | 0.12.0.1 | 2026-10-21 | Apache-2.0 | Kept as a runtime dependency |
| jax-cuda12-plugin, jax-cuda13-plugin | 0.12.0.1 | 2026-10-21 | Apache-2.0 | Not declared. The user installs `jax[cuda12]` or `jax[cuda13]`, which pins the plugin to the installed jaxlib, as the README shows |
| jax-rocm7-plugin | 0.12.0 | 2026-10-19 | Apache-2.0 | Not declared. The user installs it on Linux with an AMD GPU, as the README shows |
| jax-metal | 0.1.1 | 2024-10-08 | Apple proprietary | Rejected. Past the 24 month limit, not open source, and experimental. On Apple silicon the JAX backend runs on the CPU |

The GPU plugins are not imported by the package. JAX discovers them at start-up, as NumPy discovers a BLAS, so they are
hardware drivers chosen by the machine, not dependencies of the code. Declaring them would force a CUDA download on
every machine and would break installs on macOS. The rule that core libraries are runtime dependencies holds, since the
package imports only `jax`.

### Device and precision

- `JaxBackend(device=None, precision='float64')`. The constructor stores both. `prepare` resolves `device` to one
  `jax.Device` in `device_`: `None` gives `jax.devices()[0]`, a name gives the first device of that platform, where
  `'gpu'` takes CUDA or ROCm, and a `jax.Device` is kept. A name JAX does not know, or a platform with no device,
  raises `BackendError` that lists `jax.devices()` and the install command of the plugin.
- Every traced function and loop runs under `jax.default_device(device_)` and `jax.enable_x64(precision == 'float64')`.
  Parameters and states are put on the device once per run with `jax.device_put`.
- `float32` is opt-in. Consumer GPUs run double precision at a small fraction of single precision speed, but runs in
  single precision drift from NumPy runs after a few hundred steps of a chaotic system, so the default keeps the
  agreement of 1e-12 that the backends promise. The docstring states both facts.
- The problem records the device in its backend name, as `'jax:gpu'`, so a trajectory shows where it ran.

### Loops on the device

- `_jax_loops.py` holds one builder per loop. Each takes the traced drift, update, diffusion, or propensities and the
  boundary, and returns a `jax.jit` function with the signature of the Numba loop of the same name, so the solvers call
  both the same way. `fori_loop` runs the fixed number of internal steps of a frame, as `step_count` gives it, and
  `while_loop` runs the events of `gillespie` until the end of the frame.
- The loops take and return `jax.Array` values on the device. The solver keeps the returned array as its state. A frame
  becomes a NumPy array when the recorder reads it, through `numpy.asarray`, so a run copies one state per recorded
  frame and none per internal step.
- The forward functions in `functions_` stay for the solvers without loops, such as SciPy integrators, and keep their
  NumPy contract.
- Rejected: a whole run as one `scan`, as `unroll` does. It needs every frame time in advance and holds every frame on
  the device, which defeats observers, clocks, schedules, and real-time runs. Rejected: Diffrax for the fixed-step
  loops. Its `diffeqsolve` with a constant step controller works, but it does not cover jumps or the velocity Verlet
  split, and two loop families would hold one fact twice. Diffrax stays for the adaptive solve of `advance`.

### Noise and jumps

- Every loop that draws takes a JAX key, split once per internal step. The key of a run comes from its random state:
  an integer seed or a `SeedSequence` gives `jax.random.key` of its first 64 bits, so experiments that spawn sequences
  give each run its own key.
- Diagonal, shared, correlated, and conserved noise follow the Euler-Maruyama step of the NumPy backend, with the
  normal increments drawn by `jax.random.normal` and correlations by the Cholesky factor the problem holds. Fractional
  noise raises `BackendError`.
- Jumps follow the direct method: the total propensity, an exponential waiting time, and a categorical draw of the
  reaction by `jax.random.categorical` on the log-propensities. A frame ends at the first event past its end time, as
  in the Numba loop.
- Determinism: the threefry generator of JAX gives the same bits on every device, but sums on a GPU may change order
  between runs. The GPU tests therefore compare runs within 1e-10, and the docs name the XLA flag
  `--xla_gpu_deterministic_ops=true` for bit-equal repeats.

### Other compute backends

The suggested split asked whether another compute backend is justified. Decision: none.

| Option | Version and date | Licence | Why rejected |
| --- | --- | --- | --- |
| CuPy | cupy-cuda12x 14.2.0, 2026-08-20 | MIT | Array calls on NVIDIA and AMD GPUs, but every step of a frame is a Python call, with no fusion and no differentiation. A third printer would copy what JAX already lowers |
| Numba CUDA | numba-cuda 0.30.4, 2026-07-03 | BSD-2-Clause | Kernels close to the Numba printer, but NVIDIA only, no differentiation, and a second set of loops to keep in step with the CPU loops |
| PyTorch | 2.x | BSD-3-Clause | Not a dependency of the core. `add-torch-surrogates` keeps it to surrogates, where its models live |
| NVIDIA Warp, Taichi | active | Apache-2.0 | Kernel languages of their own, with no printer path from Sage expressions and a narrow user base |

JAX already compiles the traced functions with XLA for CPU, NVIDIA, and AMD devices, differentiates them, batches them
with `vmap`, and shards them over devices, which the later changes need.

### GPU tests and CI

- `tests/conftest.py` registers the marker `gpu: a test that needs a GPU that JAX sees` and skips such tests with the
  reason "no GPU device" when `jax.devices('gpu')` raises.
- Every device test that does not need a GPU uses `device='cpu'`, so CI covers the device code on its runners.
- A nox session `gpu` takes `all`, `tests`, or `benchmarks`, as the other sessions take their arguments. `tests` runs
  `pytest -m gpu`, `benchmarks` runs `pytest -m benchmark --benchmarks`, the benchmarks on the CPU and on the GPU, and
  `all` runs both. It fails when JAX sees no GPU, so a run on a GPU machine cannot pass by skipping.

### GPU runner and releases

Decided by the user on 2026-11-11: a GPU runner gates releases on the GPU tests and benchmarks.

- The runner is the maintainer's infrastructure, so the workflow does not name one. The repository variable
  `GPU_RUNNER` holds the label of the runner: the label of a self-hosted runner on a machine with an NVIDIA GPU and the
  CUDA driver, or the name of a GitHub GPU larger runner, which GitHub bills per minute and which an organisation
  creates in its settings. `CONTRIBUTING.md` documents both, with the driver, the JAX plugin, and the variable to set.
- `.github/workflows/gpu.yml` is a reusable workflow, called by `release.yml` and started by hand with
  `workflow_dispatch`. Its first job, on `ubuntu-latest`, reads `GPU_RUNNER`. When the variable is set, the job `gpu`
  runs on that label: it installs the package with `jax[cuda13]`, prints `jax.devices()`, and runs `nox -s gpu`. A
  last job `gate`, which always runs, fails when `gpu` failed or was cancelled.
- Until a runner exists, `GPU_RUNNER` is empty, the job `gpu` is skipped, and `gate` writes a warning annotation and a
  line in the job summary: "No GPU runner is configured (repository variable GPU_RUNNER), so the GPU tests and
  benchmarks did not run." The release then proceeds with that warning in its run, so a skip is visible and never
  reads as a pass.
- `release.yml` calls `gpu.yml` as the job `gpu`, and `build` and `docs-release` need both `ci` and `gpu`, so a failing
  GPU test or a missed GPU target stops the release before anything is published.
- Pushes and pull requests do not run the GPU job, which keeps the cost of a rented runner to releases and manual
  runs. CI keeps covering the device code on the CPU device.
- Several GPUs and several nodes: the multi-GPU benchmarks of `add-multi-device-sharding` run in the same job when the
  runner holds two GPUs or more, and are skipped with their reason otherwise. Multi-node targets stay on demand unless
  a multi-node runner exists.

### Probes

The targets are ratios to the Numba backend on the same machine, measured in task 1.2 before any code changes and
written in the benchmark docstrings.

- `KuramotoSystem` with 1000000 oscillators, coupled through the mean field, run with `RK4Solver` and step 0.01 for 100
  frames. On the CPU device the JAX loops take at most twice the Numba time. On a GPU they take at most a fifth of it.
- `SIRSystem` as a reaction network with a population of 10000000, run with `GillespieSolver` to time 50. On the CPU
  device the JAX loop takes at most three times the Numba time. A single Gillespie run is serial, so the GPU gives no
  speed-up here, and the probe checks only that it runs and agrees. Ensembles of it are the probe of
  `add-batched-runs`.
- Geometric Brownian motion by Euler-Maruyama: 2000 seeded runs on the JAX and NumPy backends give means and variances
  at time 1 within three standard errors of each other.

## Risks / Trade-offs

- [Single precision changes results] → `float64` is the default, and `float32` is documented with its drift.
- [GPU sums are not bit-reproducible] → GPU tests compare within 1e-10, and the docs name the XLA flag that makes them
  exact at a cost in speed.
- [JAX runs differ draw by draw from NumPy runs with the same seed] → The simulation spec states agreement in
  distribution for the JAX backend, and the NumPy and Numba promise of equal draws is unchanged.
- [Compile time of the loops] → One compile per problem and loop, reused across frames and runs of the same problem
  through the problem cache of experiments. It dominates short runs, which the docs say keep the default backend.
- [No GPU runner yet] → The CPU device runs the device code in CI, and the release job warns that the GPU tests did
  not run until `GPU_RUNNER` is set.
- [Cost of a rented GPU runner] → The job runs only for releases and by hand.

## Migration Plan

Nothing breaks. `JaxBackend()` keeps its behaviour, now with loops, so solvers that read `accelerated` run faster with
it. A problem that ran on the forward functions of the JAX backend runs on the loops with the same values within 1e-12.

## Open Questions

None. The user decided on 2026-11-11 that a GPU runner gates releases.
