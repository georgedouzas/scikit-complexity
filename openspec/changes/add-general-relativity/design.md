# Design

## Context

Schwichtenberg 5.2.1 presents gravity as the dynamics of curvature: the Einstein equation `G_μν = κ T_μν` says how
energy-momentum curves spacetime, and the covariant derivative, through the Christoffel symbols, says how particles
move in it. Satz (chapters 2, 4, and 6) tells the same story through the expanding universe and the collapse of stars
to black holes. The core has Sage expressions as its symbolic layer, laws that the engine compiles, and no notion of a
metric. Full `3 + 1` numerical relativity, with gauge choices and constraint damping, is out of scope by the user's
choice. What remains are the spacetimes that reduce to ordinary differential equations or to `1 + 1` fields: test
particles on fixed metrics, homogeneous cosmologies, and spherical dust.

## Goals / Non-Goals

**Goals:**

- A general object for a coordinate chart with a symbolic metric, from which the connection and the curvature follow.
- The geodesic equations as rate laws of a test particle, timelike or null.
- The Einstein equations of a metric ansatz reduced to rate laws of its unknowns, as ODEs in time or as `1 + 1` fields.
- Probe systems with known analytic results: perihelion precession, light bending, frame dragging, the Friedmann scale
  factor, and Oppenheimer-Snyder collapse.

**Non-Goals:**

- `3 + 1` evolution of the full metric, gauge conditions, constraint damping, or horizons found numerically.
- Ansatzes whose equations mix time and space derivatives, which are not in the explicit form `∂²u/∂t² = F(u, ∂u/∂t,
  ∂u/∂r, ∂²u/∂r²)` that the engine integrates.

## Decisions

### Wrap EinsteinPy, keep Sage at the boundary

EinsteinPy 0.5.0 (released 2026-10-21, after four years without a release, and maintained again) computes the
Christoffel symbols, the Riemann and Ricci tensors, and the Ricci scalar from a SymPy metric. The core converts each
Sage entry with `_sympy_()` and converts the results back with `SR(...)`, so no SymPy import appears in the package,
which the ruff ban on `sympy` keeps out. SymPy reads some function names as its own, such as `E` as Euler's number, so
the functions of an ansatz are renamed to private names on the way and restored on return. The Einstein tensor is
assembled in Sage as `R_μν - R g_μν / 2`, since EinsteinPy's own version carries the factor `1/2` as a float.
Alternatives: writing the curvature formulas in Sage (a reimplementation of a mature library, which the project avoids)
or SageManifolds (not part of the passagemath distributions the core installs).

The core uses EinsteinPy, so it is a runtime dependency. It brings astropy, plotly, and pyerfa and changes neither NumPy
nor SymPy.

### A space, named after the geometry

`PseudoRiemannianSpace` names the mathematical object, a manifold chart with a metric of any signature, never a
physical one such as a spacetime. It subclasses `ContinuousSpace` as an unbounded space of one dimension per
coordinate, so a point on it has a vector state of that dimension and every solver that runs continuous spaces runs it.
`prepare` sets `coordinates_`, `metric_`, `inverse_metric_`, `christoffel_`, and, with `curvature=True`, `riemann_`,
`ricci_`, `ricci_scalar_`, and `einstein_`. The curvature of Kerr takes minutes to simplify, so `curvature=False` leaves
it out for systems that need only geodesics. A subclass overrides `_get_metric` and `_get_coordinates`, as
`SchwarzschildSpace` and `KerrSpace` in `physics/celestial_mechanics` do.

### Geodesics as a rate law

`GeodesicInteraction` returns one rate law on the vector state of order 2, `x''^μ = -Γ^μ_αβ(x) x'^α x'^β`, with the
Christoffel symbols evaluated at the state. The time of the system is the affine parameter. Timelike and null geodesics
differ only in the norm of the initial tangent, which the equation conserves, so the interaction has no kind: the
systems set the initial four-velocity. The interaction takes the space as an argument and prepares it, since the
compiler prepares interactions without the space of the system. The system usually takes the same space, which gives
the position its dimension.

### Einstein equations by ansatz

`EinsteinInteraction` takes a space whose metric holds functions of the coordinates named after states or parameters of
its target, the components `(μ, ν)` of the mixed equations `G^μ_ν + Λ δ^μ_ν = κ T^μ_ν` to keep, one per state, and a
source function that returns the mixed energy-momentum tensor in the symbols of the target. Mixed components are used
because the diagonal ones of symmetric ansatzes are free of the spatial coordinates, unlike the covariant ones, which
carry factors such as `r²`. The interaction maps every derivative of an unknown along time to the derivative of its
state, every derivative along a coordinate that `axes` maps to a lattice axis to `partial` of the field, and every other
coordinate to an error. The equations are linear in the second time derivatives, so it solves them as a linear system
in Sage and returns one rate law per state. The constraint components, such as the Friedmann equation `G^0_0`, are not
laws: the systems choose initial values that satisfy them, and the evolution keeps them, which the tests check.
Alternatives: equation laws solved by the compiler (field elements do not take them), or a `kind` switch for
cosmologies and collapse (the ansatz carries the difference).

### Probe systems in the existing subfields

The geodesic systems go in `physics/celestial_mechanics` and the cosmologies in `physics/cosmology`, the subfields that
exist, since a `physics/relativity` subfield would be a new package, which needs the user's approval.
`oppenheimer_snyder_collapse` is the Lemaître-Tolman-Bondi dust ball on a radial lattice of comoving shells, with the
energy of every shell set so the ball starts at rest with uniform density, the Oppenheimer-Snyder case, where every
shell follows the same cycloid.

## Risks / Trade-offs

- EinsteinPy simplifies every component, which is slow for metrics with many entries. → `curvature=False`, and the
  systems that need the curvature use diagonal ansatzes.
- The compiler prepares an interaction several times, and every preparation derives the connection again. → The
  derivations of the probe systems take well under a second each, apart from the curvature of the ansatzes.
- Constraints are not enforced during the run. → The adaptive solvers keep them to the tolerance of the run, and the
  tests check them.
