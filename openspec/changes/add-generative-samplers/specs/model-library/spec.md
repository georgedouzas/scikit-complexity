## ADDED Requirements

### Requirement: Lattice φ⁴ field theory
The library SHALL provide `LatticePhi4System` in `physics/statistical_mechanics`, the real scalar field on a periodic
lattice of side `size` in `dims` dimensions with the action `Σ_x [Σ_μ (φ_{x+μ} - φ_x)² / 2 + m² φ_x² / 2 + λ φ_x⁴ / 24]`
of 2504.18126, sampled by local Gaussian moves of a set width, and declaring the energy of a site so that flow
proposals can score whole configurations.

#### Scenario: Free propagator
- **WHEN** the system runs with `λ = 0` and `m² = 1` on an 8 × 8 lattice for 5,000 sweeps
- **THEN** the mean of `φ²` over the sites and frames lies within 3% of the exact value
  `(1 / V) Σ_p 1 / (Σ_μ 4 sin²(p_μ / 2) + m²)`

#### Scenario: Broken symmetry
- **WHEN** the system runs with `λ = 0.5` and `m² = -4` on an 8 × 8 lattice from random values
- **THEN** the mean of `|φ|` over the sites settles above 1.5

### Requirement: The lattice φ⁴ system is recorded in the inventory
The inventory SHALL record `lattice_phi4` as implemented with 2504.18126 as its source and the primitives `monte-carlo`
and `generative-samplers`, and the gallery SHALL hold its example, which compares local and flow proposals.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for `lattice_phi4`
