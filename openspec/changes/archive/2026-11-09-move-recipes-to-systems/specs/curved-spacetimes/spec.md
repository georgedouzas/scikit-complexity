## MODIFIED Requirements

### Requirement: Pseudo-Riemannian spaces derive their curvature
`SpacetimeSpace` of `skcomplex.systems.physics.common.relativity` SHALL take a symmetric, nondegenerate Sage metric and
its coordinates, and `prepare` SHALL derive through EinsteinPy, loaded as the extra `relativity`, the Christoffel
symbols `christoffel_[μ][α][β] = Γ^μ_αβ` and, unless `curvature=False`, the Riemann tensor, the Ricci tensor, the Ricci
scalar, and the Einstein tensor, as Sage expressions. It SHALL be an unbounded `ContinuousSpace` with one dimension per
coordinate, and a bad metric or bad coordinates SHALL raise a `ValidationError` from `prepare`.

#### Scenario: Unit sphere
- **WHEN** a user prepares the space of the metric `diag(1, sin²θ)` in `(θ, φ)`
- **THEN** `christoffel_[0][1][1]` simplifies to `-sin θ cos θ` and `ricci_scalar_` is 2

#### Scenario: Schwarzschild is Ricci-flat
- **WHEN** a user prepares the Schwarzschild metric of a symbolic mass `M`
- **THEN** every entry of `ricci_` and `einstein_` simplifies to 0 and `christoffel_[1][0][0]` to `M (r - 2M) / r³`

#### Scenario: Connection only
- **WHEN** a user prepares a space with `curvature=False`
- **THEN** it has `christoffel_` and its curvature attributes are `None`

#### Scenario: Prepared again
- **WHEN** a prepared space is prepared again with the same metric, coordinates, and curvature flag
- **THEN** it keeps the same `christoffel_` object, and after `set_params(curvature=True)` it derives the tensors again

#### Scenario: Degenerate metric
- **WHEN** a user prepares the metric `diag(1, 0)`
- **THEN** `prepare` raises a `ValidationError` that says the metric is degenerate

#### Scenario: Missing extra
- **WHEN** a user prepares a `SpacetimeSpace` without `einsteinpy` installed
- **THEN** the error names the extra `relativity`, and the core imports without `einsteinpy`

### Requirement: Geodesic interactions move points along geodesics
`GeodesicInteraction` of `skcomplex.systems.physics.common.relativity` SHALL return one rate law
`x''^μ = -Γ^μ_αβ(x) x'^α x'^β` on a vector state of order 2 of its target, with the Christoffel symbols of the
`SpacetimeSpace` it takes, for timelike and null tangents alike. It SHALL raise a `SpaceError` from `prepare` without
such a space, and a `ValidationError` for a position whose number of components differs from the number of
coordinates.

#### Scenario: Straight line in polar coordinates
- **WHEN** a point of the polar plane `diag(1, r²)` starts at `(1, 0)` with the tangent `(0, 1)`
- **THEN** it follows the line `x = 1` at unit speed

#### Scenario: Null tangent stays null
- **WHEN** a point starts with a null tangent
- **THEN** `g_μν x'^μ x'^ν` stays 0 within the tolerance of the solver

### Requirement: Einstein interactions reduce the field equations of an ansatz
`EinsteinFieldEquationsInteraction` of `skcomplex.systems.physics.cosmology` SHALL take a `SpacetimeSpace` with
curvature whose metric holds functions of the coordinates named after states or parameters of its target, one
component per state, a source of the mixed energy-momentum tensor, the coupling `κ`, and the cosmological constant `Λ`.
It SHALL solve `G^μ_ν + Λ δ^μ_ν = κ T^μ_ν` of the kept components for the second time derivatives of the states and
return one rate law per state, and it SHALL raise a `ValidationError` that names any derivative or component it cannot
reduce.

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
- **WHEN** a user lists `skcomplex.core.components.interactions` and `skcomplex.core.components.spaces`
- **THEN** they hold no geodesic, no spacetime, and no interaction of the field equations, and the physics systems
  build all three from `skcomplex.systems.physics`
