# Spec Delta

## ADDED Requirements

### Requirement: Lattice Yang-Mills model samples SU(2) and SU(3) links
The library SHALL provide `lattice_yang_mills` in `physics/waves`, after Satz (2022 §7) and Schwichtenberg (2020 §4.4).
It SHALL sample SU(2) or SU(3) link matrices on the cell complex of a periodic lattice by the Wilson action
`-(beta / N) sum_p Re tr U_p` under discrete time, through `WilsonSamplingInteraction` with a matrix group, from Haar
links or given links, with an extent of the last axis for finite temperature. Its catalog entry SHALL declare the mean
plaquette and the Polyakov loop as observables, and `GaugeAnalysis` SHALL measure Wilson loops of given extents and the
Polyakov loop of matrix links.

#### Scenario: Four-dimensional SU(2)
- **WHEN** `lattice_yang_mills` samples SU(2) at `beta = 2.3` on a periodic 4^4 lattice for 100 sweeps
- **THEN** the mean plaquette after 30 sweeps is within 0.03 of 0.60

#### Scenario: Deconfinement
- **WHEN** `lattice_yang_mills` samples SU(2) on a 4 × 4 × 4 × 2 lattice at `beta = 1.5` and at `beta = 2.6`
- **THEN** the mean Polyakov loop at `beta = 2.6` is more than three times that at `beta = 1.5`

#### Scenario: Gauge-invariant loops of matrix links
- **WHEN** a random gauge transformation is applied to random SU(2) or SU(3) links
- **THEN** the 2 × 1 Wilson loop and the Polyakov loop are unchanged within 1e-12

### Requirement: Lattice Yang-Mills is recorded in the system inventory
The inventory SHALL record `lattice_yang_mills` as implemented with its sources, the primitives `link-fields` and
`matrix-groups`, and the matrix domain, and the gallery SHALL hold its curated example.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for `lattice_yang_mills`
