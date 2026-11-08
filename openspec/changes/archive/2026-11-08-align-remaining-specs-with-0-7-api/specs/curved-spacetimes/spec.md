## MODIFIED Requirements

### Requirement: Relativity systems reproduce known results
The catalog SHALL hold `SchwarzschildPrecessionSystem`, `LightBendingSystem`, and `KerrFrameDraggingSystem` in
`physics/celestial_mechanics` and `FLRWCosmologySystem` and `OppenheimerSnyderCollapseSystem` in `physics/cosmology`, built on
these interactions, with inventory entries and gallery examples.

#### Scenario: Perihelion precession
- **WHEN** `SchwarzschildPrecessionSystem(1000, 0.5)` runs for two turns
- **THEN** its periapsis advances per turn by `6 π M / (a (1 - e²))` within 2 percent

#### Scenario: Light bending
- **WHEN** `LightBendingSystem(100, 1e5)` runs past the mass
- **THEN** the ray turns by the exact deflection of a null geodesic within 0.2 percent, close to `4 M / b`

#### Scenario: Frame dragging
- **WHEN** `KerrFrameDraggingSystem(10, spin=0.9)` runs
- **THEN** `dφ/dt` equals `2 M a / (r³ + a² r + 2 M a²)` within a relative 1e-7 along the fall

#### Scenario: Friedmann scale factor
- **WHEN** `FLRWCosmologySystem(0.5)` runs with only matter, only radiation, or only a cosmological constant
- **THEN** the scale factor follows `t^(2/3)`, `t^(1/2)`, or `e^t` within a relative 1e-8

#### Scenario: Oppenheimer-Snyder collapse
- **WHEN** `OppenheimerSnyderCollapseSystem(8)` runs to proper time 34.5
- **THEN** every shell keeps `R = r a(t)` on the cycloid within a relative 1e-7 and the surface crosses `2 M` at the
  time the cycloid gives
