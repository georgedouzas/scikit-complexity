# curved-spacetimes Specification

## Purpose
TBD - created by archiving change add-general-relativity. Update Purpose after archive.

## Requirements

### Requirement: Pseudo-Riemannian spaces derive their curvature
`PseudoRiemannianSpace` SHALL take a symmetric, nondegenerate Sage metric and its coordinates, and `prepare` SHALL
derive through EinsteinPy the Christoffel symbols `christoffel_[μ][α][β] = Γ^μ_αβ` and, unless `curvature=False`, the
Riemann tensor, the Ricci tensor, the Ricci scalar, and the Einstein tensor `R_μν - R g_μν / 2`, as Sage expressions.
Symbolic functions of the coordinates in the metric SHALL come back as the same functions, whatever their names. The
space SHALL be an unbounded `ContinuousSpace` with one dimension per coordinate. A metric that is not square with one
row per coordinate, not symmetric, or identically degenerate, or coordinates that are not distinct variables, SHALL
raise a `ValidationError` from `prepare`.

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

### Requirement: Geodesic interactions move points along geodesics
`GeodesicInteraction` SHALL return one rate law `x''^μ = -Γ^μ_αβ(x) x'^α x'^β` on a vector state of order 2 of its
target, with the Christoffel symbols of the `PseudoRiemannianSpace` it takes, for timelike and null tangents alike. It
SHALL raise a `SpaceError` from `prepare` without such a space, and a `ValidationError` for a position whose number of
components differs from the number of coordinates.

#### Scenario: Straight line in polar coordinates
- **WHEN** a point of the polar plane `diag(1, r²)` starts at `(1, 0)` with the tangent `(0, 1)`
- **THEN** it follows the line `x = 1` at unit speed

#### Scenario: Null tangent stays null
- **WHEN** a point starts with a null tangent
- **THEN** `g_μν x'^μ x'^ν` stays 0 within the tolerance of the solver

### Requirement: Einstein interactions reduce the field equations of an ansatz
`EinsteinInteraction` SHALL take a `PseudoRiemannianSpace` with curvature whose metric holds functions of the
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

### Requirement: Relativity systems reproduce known results
The catalog SHALL hold `schwarzschild_precession`, `light_bending`, and `kerr_frame_dragging` in
`physics/celestial_mechanics` and `flrw_cosmology` and `oppenheimer_snyder_collapse` in `physics/cosmology`, built on
these interactions, with inventory entries and gallery examples.

#### Scenario: Perihelion precession
- **WHEN** `schwarzschild_precession(1000, 0.5)` runs for two turns
- **THEN** its periapsis advances per turn by `6 π M / (a (1 - e²))` within 2 percent

#### Scenario: Light bending
- **WHEN** `light_bending(100, 1e5)` runs past the mass
- **THEN** the ray turns by the exact deflection of a null geodesic within 0.2 percent, close to `4 M / b`

#### Scenario: Frame dragging
- **WHEN** `kerr_frame_dragging(10, spin=0.9)` runs
- **THEN** `dφ/dt` equals `2 M a / (r³ + a² r + 2 M a²)` within a relative 1e-7 along the fall

#### Scenario: Friedmann scale factor
- **WHEN** `flrw_cosmology(0.5)` runs with only matter, only radiation, or only a cosmological constant
- **THEN** the scale factor follows `t^(2/3)`, `t^(1/2)`, or `e^t` within a relative 1e-8

#### Scenario: Oppenheimer-Snyder collapse
- **WHEN** `oppenheimer_snyder_collapse(8)` runs to proper time 34.5
- **THEN** every shell keeps `R = r a(t)` on the cycloid within a relative 1e-7 and the surface crosses `2 M` at the
  time the cycloid gives
