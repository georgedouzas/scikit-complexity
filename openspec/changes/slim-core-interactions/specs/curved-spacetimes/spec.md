## MODIFIED Requirements

### Requirement: Einstein interactions reduce the field equations of an ansatz
`EinsteinFieldEquationsInteraction` of `skcomplex.systems.physics.cosmology` SHALL take a `PseudoRiemannianSpace` with curvature whose metric holds functions of the
coordinates named after states or parameters of its target, one component `(μ, ν)` per state, a source function that
returns the mixed energy-momentum tensor, the coupling `κ`, and the cosmological constant `Λ`. It SHALL solve the
equations `G^μ_ν + Λ δ^μ_ν = κ T^μ_ν` of the kept components for the second time derivatives of the states and return
one rate law per state. A time derivative SHALL become the derivative of the state, and a derivative along a coordinate
that `axes` maps to a lattice axis SHALL become `partial` of the field along that axis. A mixed derivative, a derivative
of order above 2, a derivative along a coordinate without an axis, a kept coordinate, a function that names nothing, or
components that do not fix the second derivatives SHALL raise a `ValidationError` that names the problem.

#### Scenario: Einstein-de Sitter universe
- **WHEN** a flat Robertson-Walker ansatz holds dust of density `3 / (8 π a³)` and the component `(1, 1)` drives the
  scale factor from `a = 1` with `a' = 1`
- **THEN** the scale factor follows `(3 t / 2 + 1)^(2/3)` within a relative 1e-8

#### Scenario: Field in 1 + 1 dimensions
- **WHEN** the ansatz `-dt² + dx² + u(t, x)² (dy² + dz²)` keeps the component `(2, 2)` with `x` on the axis 0 of a
  periodic lattice
- **THEN** the law is `u'' = partial(partial(u, 0), 0)` and a standing sine wave oscillates as `cos t`

#### Scenario: Constraint component
- **WHEN** the same ansatz keeps only the component `(0, 0)`
- **THEN** running the interaction raises a `ValidationError` that says the components do not fix the second
  derivatives

#### Scenario: Field equations live in physics
- **WHEN** a user lists `skcomplex.core.components.interactions`
- **THEN** it holds `GeodesicInteraction` and no interaction of the field equations, and the cosmology systems build
  `EinsteinFieldEquationsInteraction` from their own module
