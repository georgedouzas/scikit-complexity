# Spec Delta

## ADDED Requirements

### Requirement: GPU tests run where a GPU exists
A test that needs a GPU SHALL carry the `gpu` marker. A run where JAX sees no GPU SHALL skip it with a reason that says
so, and the `gpu_tests` session of nox SHALL run only the marked tests and SHALL fail when JAX sees no GPU.

#### Scenario: Machine without a GPU
- **WHEN** a contributor runs `pytest` on a machine without a GPU
- **THEN** every test marked `gpu` is reported as skipped with the reason that names the missing GPU device

#### Scenario: GPU session without a GPU
- **WHEN** the `gpu_tests` session of nox runs on a machine without a GPU
- **THEN** it fails with a message that names the missing GPU device
