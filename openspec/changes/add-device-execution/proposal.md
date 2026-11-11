## Why

The user asked on 2026-11-11 for massive simulations: many runs and huge single runs, on one GPU workstation, a
multi-GPU node, HPC clusters, and the cloud. The JAX backend is the only backend that can reach a GPU, but today it
cannot use one in practice:

- It has no device or precision setting. It runs on whatever device JAX picks and always in double precision.
- Its forward functions take and return NumPy arrays, so every internal step of a run copies the state from the host
  to the device and back. The NumPy and Numba backends fill `LoweredFunctionsSpec.accelerated`, the compiled loops
  that run every internal step of a frame, but the JAX backend fills none.
- It writes only deterministic ODEs and maps. Noise laws and jump laws, which every stochastic ensemble needs, have no
  JAX form outside the private unrolled runs of calibration.

This change makes the JAX backend a device backend: it runs a whole frame on a chosen device in a chosen precision,
with noise and jumps, and copies the state to the host only when a frame is recorded. It is the first of the planned
changes for massive simulations, and the batched, sharded, and large-system changes build on it.

## What Changes

- **Device and precision.** `JaxBackend(device=None, precision='float64')`. `device` is `'cpu'`, `'gpu'`, `'cuda'`,
  `'rocm'`, a `jax.Device`, or `None` for the default device of JAX. `precision` is `'float64'` or `'float32'`. An
  unknown or absent device raises `BackendError`, whose message lists the devices JAX sees and the install command of
  the plugin.
- **Runs that stay on the device.** The JAX backend fills `accelerated` with the loops `iterate`, `rk4`,
  `velocity_verlet`, `yoshida4`, `euler_maruyama`, and `gillespie`, each a `jax.jit` function of a whole frame built on
  `jax.lax.fori_loop` or `jax.lax.while_loop`. The state of a run lives on the device between frames, and a frame is
  copied to the host once, when the trajectory records it.
- **Noise and jumps in JAX.** Noise laws of every `NoiseKind` except fractional noise, and jump laws by the direct
  method of Gillespie, lower to JAX. Draws come from a JAX key derived from the random state of the run, so a seeded run
  repeats exactly on one device. JAX streams are not NumPy streams, so a JAX run agrees with a NumPy run in
  distribution, not draw by draw.
- **GPU tests.** A `gpu` test marker. Tests with it are skipped when JAX sees no GPU, and a new `gpu_tests` session of
  nox runs them and fails without a GPU. CI keeps running every device test on the CPU device of JAX.
- **Benchmarks.** Benchmarks of a large Kuramoto system and of a stochastic SIR ensemble member on the CPU and the GPU
  device of JAX against the Numba backend, with targets set as ratios to the Numba time on the same machine.
- **No new compute backend.** JAX covers NVIDIA and AMD GPUs. CuPy, Numba CUDA, PyTorch, Warp, and Taichi were
  surveyed and rejected in the design.

## Capabilities

### New Capabilities

- `device-execution`: running a compiled problem on a chosen device in a chosen precision, with every internal step
  on the device.

### Modified Capabilities

- `model-compilation`: the backend list names the JAX backend and its device, and the automatic choice stays on the
  CPU.
- `simulation`: seeded JAX runs are reproducible on one device and agree in distribution with NumPy runs.
- `test-suite`: the `gpu` marker and its session.

## Impact

- Core: `core/engine/backends/_jax.py` gains `device` and `precision`, and a new private module
  `core/engine/backends/_jax_loops.py` holds the JAX loops, in the way `_numba_loops.py` holds the Numba loops. The
  solvers that read `accelerated` keep the state the loop returns without converting it.
- Dependencies: none new. `jax` stays a runtime dependency. The GPU plugins `jax-cuda12-plugin`,
  `jax-cuda13-plugin`, and `jax-rocm7-plugin` are installed by the user through the extras of `jax`, as the README
  shows.
- Tests and CI: the `gpu` marker in `tests/conftest.py`, the `gpu_tests` session in `noxfile.py`, and a manual GitHub
  workflow for a GPU runner.
- Docs: a devices section in the simulate page of the user guide and the GPU install lines in the README.
- Order: first of the changes for massive simulations. `add-cluster-executors` runs in parallel with it.
  `add-batched-runs`, `add-large-system-kernels`, `add-multi-device-sharding`, and `add-domain-decomposition` follow.
