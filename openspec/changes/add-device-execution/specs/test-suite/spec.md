# Spec Delta

## ADDED Requirements

### Requirement: GPU tests run where a GPU exists
A test that needs a GPU SHALL carry the `gpu` marker. A run where JAX sees no GPU SHALL skip it with a reason that says
so, and the `gpu` session of nox SHALL run the marked tests and the benchmarks and SHALL fail when JAX sees no GPU.

#### Scenario: Machine without a GPU
- **WHEN** a contributor runs `pytest` on a machine without a GPU
- **THEN** every test marked `gpu` is reported as skipped with the reason that names the missing GPU device

#### Scenario: GPU session without a GPU
- **WHEN** the `gpu` session of nox runs on a machine without a GPU
- **THEN** it fails with a message that names the missing GPU device

### Requirement: Releases wait for the GPU tests
The release workflow SHALL run the `gpu` session on the GPU runner that the repository variable `GPU_RUNNER` names and
SHALL publish only when it passes. When no runner is configured, the GPU job SHALL be skipped with a warning that says
the GPU tests and benchmarks did not run.

#### Scenario: Failing GPU test
- **WHEN** a release tag is pushed with `GPU_RUNNER` set and a test marked `gpu` fails on the runner
- **THEN** the release workflow publishes nothing to PyPI, GitHub, or the documentation site

#### Scenario: No GPU runner
- **WHEN** a release tag is pushed with `GPU_RUNNER` empty
- **THEN** the GPU job is skipped, and the run shows a warning that names `GPU_RUNNER` and says the GPU tests and
  benchmarks did not run
