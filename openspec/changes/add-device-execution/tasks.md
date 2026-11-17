# Tasks

## 1. Baselines

- [ ] 1.1 Install `jax[cuda13]` on a GPU machine and record in the design the devices JAX lists, and verify
  `jax.devices('gpu')` returns at least one device, or record that no GPU machine is available and run only the CPU
  parts
- [ ] 1.2 Time the Kuramoto probe of 1000000 oscillators and the Gillespie probe of the SIR network of 10000000 people
  on the Numba and NumPy backends of the current code, write the times and the machine in the benchmark docstrings,
  and verify the numbers repeat within 10% over three runs

## 2. Device and precision

- [ ] 2.1 Add `device` and `precision` to `JaxBackend`, with `_check` of the precision and `_prepare` that resolves the
  device into `device_`, and verify the missing-GPU and unknown-precision scenarios in
  `tests/core/engine/backends/test_jax.py`
- [ ] 2.2 Run every traced function under `jax.default_device` and `jax.enable_x64` of the precision, put parameters
  and states on the device once per run, and record the backend name with its platform, and verify the CPU-device
  scenario
- [ ] 2.3 Verify the single-precision scenario, and that the existing JAX tests, calibration tests, and learning
  compiler tests pass unchanged with the default settings

## 3. Loops on the device

- [ ] 3.1 Add `core/engine/backends/_jax_loops.py` with the builder of `iterate` for deterministic and drawing maps on
  `fori_loop`, fill `accelerated.iterate`, and verify the logistic map part of the same-values scenario
- [ ] 3.2 Add the builders of `rk4`, `velocity_verlet`, and `yoshida4` with the boundary of the problem, and verify the
  Lorenz and harmonic oscillator parts of the same-values scenario
- [ ] 3.3 Keep the `jax.Array` a loop returns as the state of the solvers that read `accelerated`, convert it in the
  recorder only, and verify the one-copy-per-frame scenario with a counting observer
- [ ] 3.4 Verify that observers, clocks, schedules, and invariants work on a run with the JAX loops, through the
  existing runtime tests run once with `JaxBackend(device='cpu')`

## 4. Noise and jumps

- [ ] 4.1 Derive the JAX key of a run from an integer seed, a `SeedSequence`, or a NumPy generator, and verify the
  spawned-sequences scenario in `tests/core/engine/runtime/test_simulate.py`
- [ ] 4.2 Lower diagonal, shared, correlated, and conserved noise to the `euler_maruyama` loop, add the noise kinds to
  the support of the backend, and verify the geometric Brownian motion scenario
- [ ] 4.3 Lower jump laws to the `gillespie` loop by the direct method, add `JumpLaw` to the support of the backend,
  and verify the stochastic-SIR and same-seed-on-the-CPU-device scenarios
- [ ] 4.4 Raise `BackendError` for fractional noise, and verify the fractional-noise scenario

## 5. Compilation

- [ ] 5.1 Keep the automatic backend choice off the JAX backend, and verify the automatic-choice-ignores-a-GPU scenario
  in `tests/core/engine/compile/test_base.py` with a fake device list
- [ ] 5.2 Verify the backends-agree scenario with the JAX backend in double precision added beside the Numba and NumPy
  backends

## 6. GPU tests and benchmarks

- [ ] 6.1 Register the `gpu` marker and its skip in `tests/conftest.py`, and verify the machine-without-a-GPU scenario
- [ ] 6.2 Add the `gpu` session to `noxfile.py` with the arguments `all`, `tests`, and `benchmarks`, and verify the
  GPU-session-without-a-GPU scenario
- [ ] 6.3 Add the benchmarks of the large Kuramoto system on the CPU device and the GPU, and verify the CPU target
  locally and the GPU target on the GPU machine of task 1.1
- [ ] 6.4 Verify the GPU-device scenario on the GPU machine

## 7. GPU runner

- [ ] 7.1 Add the reusable workflow `.github/workflows/gpu.yml`, with `workflow_call` and `workflow_dispatch`, a job
  that reads the repository variable `GPU_RUNNER`, a job `gpu` that runs `nox -s gpu` on that label when it is set,
  and a job `gate` that always runs, fails when `gpu` failed or was cancelled, and writes the no-runner warning to the
  annotations and the job summary when `GPU_RUNNER` is empty, and verify it with `actionlint`
- [ ] 7.2 Call `gpu.yml` from `.github/workflows/release.yml` as the job `gpu` and make `build` and `docs-release` need
  `ci` and `gpu`, and verify with `actionlint` and by reading the job graph that no publishing job can start before
  `gpu` ends
- [ ] 7.3 Start `gpu.yml` by hand with `GPU_RUNNER` empty, and verify the run shows the job `gpu` as skipped and the
  warning that names `GPU_RUNNER`, which is the no-GPU-runner scenario
- [ ] 7.4 Document in the release section of `CONTRIBUTING.md` the two kinds of runner, a self-hosted runner with an
  NVIDIA GPU, its driver, and its label, and a GitHub GPU larger runner created in the organisation settings, and the
  variable `GPU_RUNNER` that names either, and verify the variable and job names match `gpu.yml`
- [ ] 7.5 When a runner is configured, start `gpu.yml` by hand on it, and verify the `gpu` session passes there and a
  deliberately failing test marked `gpu` on a scratch branch fails `gate`, which is the failing-GPU-test scenario, or
  record in the design that no runner exists yet

## 8. Documentation

- [ ] 8.1 Add a devices section to `docs/overview/user_guide/simulate.md` with `device`, `precision`, the cost of
  double precision on consumer GPUs, the deterministic XLA flag, and a runnable example on the CPU device, and verify
  the markdown tests run it
- [ ] 8.2 Add the GPU install lines for CUDA 12, CUDA 13, and ROCm to the installation section of `README.md` and to
  the troubleshooting page, which also states that Apple silicon runs on the CPU, and verify the docs build

## 9. Integration

- [ ] 9.1 Run `nox` (formatting, ruff, mypy, pydoclint, import contracts, tests of the touched modules with the thread
  limits set to 1, doctests, and the strict docs build) and verify every session passes
- [ ] 9.2 Run `openspec validate add-device-execution --strict` and verify it passes
