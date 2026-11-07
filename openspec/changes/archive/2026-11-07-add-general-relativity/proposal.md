# Proposal

## Why

The catalog has gravity only as Newtonian forces and a Friedmann acceleration written by hand, and the audit lists the
Einstein field equations of Schwichtenberg 5.2.1 as not supported. General relativity reduces to equations the engine
already runs whenever a symmetry ansatz or a test particle leaves ordinary differential equations or `1 + 1` fields, so
a symbolic metric with its derived curvature, and the laws that come from it, open a whole field without full `3 + 1`
numerical relativity.

## What Changes

- Add the runtime dependency EinsteinPy (0.5.0, released 2026-10-21), which derives the Christoffel symbols and the
  Riemann, Ricci, and Einstein tensors from a symbolic metric. The core converts Sage expressions to SymPy and back at
  the boundary, so laws stay in Sage.
- Add `PseudoRiemannianSpace` to `skcomplex.core.components.spaces`: an unbounded continuous space of one dimension per
  coordinate, built from a symbolic metric that may hold unknown functions of the coordinates, which `prepare` turns
  into `christoffel_`, `riemann_`, `ricci_`, `ricci_scalar_`, and `einstein_`.
- Add `GeodesicInteraction` to `skcomplex.core.components.interactions`: the geodesic equation of a point of a
  pseudo-Riemannian space as a rate law of its vector position of order 2, for timelike and null geodesics alike.
- Add `EinsteinInteraction`: the mixed Einstein equations `G^μ_ν + Λ δ^μ_ν = κ T^μ_ν` of a metric ansatz, whose
  unknowns are states of an element or of a field, solved for their second time derivatives as rate laws. Derivatives
  along a coordinate mapped to a lattice axis become field derivatives, so spherical ansatzes give `1 + 1` fields.
  Equations that keep other coordinates, mix time and space derivatives, or do not fix the second derivatives raise.
- Add catalog systems: `schwarzschild_precession`, `light_bending`, and `kerr_frame_dragging` in
  `physics/celestial_mechanics`, with the spaces `SchwarzschildSpace` and `KerrSpace`, and `flrw_cosmology` and
  `oppenheimer_snyder_collapse` in `physics/cosmology`, each with tests against known results, an inventory entry,
  and a gallery example.
- Add the primitive `curved-spacetimes` to the inventory.

## Capabilities

### New Capabilities

- `curved-spacetimes`: symbolic metrics with derived curvature, geodesic motion, and Einstein equations reduced by a
  symmetry ansatz, with the catalog systems that exercise them.

### Modified Capabilities

## Impact

- `pyproject.toml` and `pdm.lock` gain `einsteinpy>=0.5.0`, which brings astropy, plotly, and pyerfa. NumPy and SymPy
  keep their versions. The lock resolves for Python 3.12 to 3.14, and EinsteinPy computes the Schwarzschild Einstein
  tensor with NumPy 2.5.4 and SymPy 1.14.0 on each of them.
- New modules `spaces/_pseudo_riemannian.py`, `interactions/_geodesic.py`, and `interactions/_einstein.py`. No new
  packages. The relativity systems go in the existing subfields `celestial_mechanics` and `cosmology`, since a new
  `physics/relativity` subfield would be a new package.
- The curvature of a metric with many entries, such as Kerr's, takes minutes in EinsteinPy, so the space takes
  `curvature=False` to derive only the connection.
