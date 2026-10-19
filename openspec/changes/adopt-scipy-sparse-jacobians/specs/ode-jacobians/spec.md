# Spec Delta

## Purpose

Lets implicit ODE integrators solve stiff finite-difference and finite element fields fast, by deriving the Jacobian's
sparsity pattern and an exact sparse Jacobian from the compiled symbolic laws and passing them to SciPy.

## ADDED Requirements

### Requirement: The Jacobian pattern is derived from the laws
The framework SHALL derive the sparsity pattern of the Jacobian of a compiled ODE problem from its symbolic laws:
member reads, shared reads, field neighbour tables, sparse operators of finite-difference, finite element, and
finite-volume fields, and dense blocks for couplings known only at run time. The pattern SHALL be a superset of the
true pattern, include the diagonal, and leave fixed rows empty.

#### Scenario: Superset of the true pattern
- **WHEN** the finite-difference Jacobian is taken at three random states of every library model with an ODE form
- **THEN** every entry above 1e-8 in absolute value lies inside the derived pattern

#### Scenario: Stencil pattern
- **WHEN** Fisher-KPP on 1000 points uses the three-point Laplacian with zero-flux edges, and again with periodic edges
- **THEN** the pattern has 2998 entries and is tridiagonal with zero-flux edges, and has 3000 entries with periodic edges

#### Scenario: Mesh pattern
- **WHEN** heat with a reaction term runs on a triangular mesh with fixed boundary temperatures
- **THEN** the pattern equals the stiffness matrix's pattern with the diagonal on free nodes, and fixed nodes have empty
  rows

#### Scenario: Run-time couplings
- **WHEN** a group couples its members through a pair interaction or a spectral operator
- **THEN** the pattern holds the dense block of the group or field

### Requirement: The exact sparse Jacobian is derived from the laws
The framework SHALL derive an exact sparse Jacobian of a compiled ODE problem by differentiating its symbolic laws,
evaluated at the current parameters, with the derived pattern as its structure. Laws without a symbolic form SHALL make
the exact Jacobian unavailable with a named error.

#### Scenario: Exact Jacobian matches finite differences
- **WHEN** the exact Jacobian of Gray-Scott on a 64 by 64 lattice and of heat with a reaction term on a mesh is
  evaluated at a random state
- **THEN** it agrees with central finite differences within 1e-6 relative error in the Frobenius norm

#### Scenario: Parameters change
- **WHEN** a parameter changes between two calls of a run with the exact Jacobian
- **THEN** the second call's Jacobian uses the new value

#### Scenario: Callable law
- **WHEN** a user asks for the exact Jacobian of a system with a callable law
- **THEN** initializing raises a named error that names the law and suggests the automatic option

### Requirement: SciPy integrators receive the Jacobian structure
The SciPy ODE solver SHALL pass the sparsity pattern to its BDF and Radau methods by default, the half bandwidths of a
narrow banded pattern to LSODA, and the exact sparse Jacobian when asked. It SHALL keep the dense path on request and
when the pattern is denser than one half. Explicit methods SHALL ignore the option and build no structure.

#### Scenario: Speed over the dense path
- **WHEN** Fisher-KPP on 4000 points with diffusion making its largest eigenvalue about 1600 runs to `T = 10` with BDF
  at `rtol = 1e-6`, with the pattern and with the dense path
- **THEN** the pattern path is at least 20 times faster, and both agree within 1e-5 relative error

#### Scenario: Radau and exact Jacobians
- **WHEN** the same problem runs with Radau and the pattern, and with BDF and the exact Jacobian
- **THEN** each is at least 10 times faster than BDF on the dense path, and the exact run uses at least 5 times fewer
  drift evaluations than the pattern run

#### Scenario: Banded LSODA
- **WHEN** Fisher-KPP on 4000 points with zero-flux edges runs with LSODA
- **THEN** LSODA receives half bandwidths of 1 and is at least 10 times faster than without them

#### Scenario: Explicit methods ignore the option
- **WHEN** a field problem runs with RK45 with each value of the option
- **THEN** no structure is built and the trajectories are identical

### Requirement: The hybrid path keeps the Jacobian structure
The SciPy ODE solver SHALL pass the same Jacobian structure when it integrates step by step for events, breakpoints,
and delays, at every restart.

#### Scenario: Event on a stiff field
- **WHEN** Fisher-KPP on 4000 points runs with BDF and a terminal event when the total population crosses half the
  domain size
- **THEN** the event time agrees with the dense path within 1e-6, and the run is at least 10 times faster

#### Scenario: Delayed stiff field
- **WHEN** a diffusive Hutchinson equation with delay 1 on 2000 points runs with BDF and the pattern
- **THEN** the final state agrees with the dense path within 1e-5 relative error
