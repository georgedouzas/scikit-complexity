## ADDED Requirements

### Requirement: Lyapunov spectra of fields
`lyapunov(kind='spectrum')` SHALL compute the spectrum of a spectral or lattice field system, above the member limit of
symbolic expansion as well as below it.

#### Scenario: Kuramoto Sivashinsky
- **WHEN** the spectrum of a Kuramoto Sivashinsky system of length 22 is computed
- **THEN** its largest exponent is close to 0.04

#### Scenario: Default modes
- **WHEN** the spectrum of `KuramotoSivashinskySystem()` with 128 modes is computed
- **THEN** it returns exponents instead of an error about the member limit

### Requirement: Dispersion of spectral derivatives
`stability(kind='dispersion')` SHALL linearise `partial` derivatives of any order as well as Laplacians.

#### Scenario: Kuramoto Sivashinsky growth rate
- **WHEN** the dispersion relation of `KuramotoSivashinskySystem()` is computed
- **THEN** the growth rate at every wavenumber `q` is `q^2 - q^4`

### Requirement: Defect charges sum to zero on periodic lattices
`defects` SHALL give every edge one wrapped angle difference whose reverse is its negative, so the charges on a
periodic lattice sum to 0.

#### Scenario: Half period ties
- **WHEN** the levels of a Hodgepodge run with 100 levels are mapped to phases `2 pi s / 100` and their defects are found
- **THEN** the total charge is 0
