# Spec Delta

## MODIFIED Requirements

### Requirement: Laws compile to a chosen backend
Compiling a system SHALL accept a backend, `numba`, `numpy` as the reference, `jax` on a chosen device, or `auto` by
default, which picks `numba` for systems whose groups hold at least 1000 members in total and `numpy` otherwise, and
never picks `jax` or a GPU. The `numba` and `numpy` backends SHALL produce the same numeric functions for the same
system, and the `jax` backend in double precision SHALL produce them within 1e-12 relative error.

#### Scenario: Backends agree
- **WHEN** an ideal gas with pair interactions is compiled with both backends and evaluated on the same state
- **THEN** the drift, boundary, and update results agree to within 1e-12 relative error

#### Scenario: Automatic choice
- **WHEN** a system with a group of 1000 members and a system with two single elements are compiled with `auto`
- **THEN** the first uses the Numba backend and the second uses the NumPy backend

#### Scenario: Automatic choice ignores a GPU
- **WHEN** a system with a group of 100000 members is compiled with `auto` on a machine where JAX sees a GPU
- **THEN** it uses the Numba backend

#### Scenario: Unknown backend
- **WHEN** a user asks for a backend that is not registered
- **THEN** a named error is raised that lists the registered backends
