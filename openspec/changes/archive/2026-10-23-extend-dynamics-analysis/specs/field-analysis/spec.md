# Spec Delta

## ADDED Requirements

### Requirement: Dispersion relations of field models
The analysis SHALL compute the dispersion relation of a field model, site groups on a lattice with rate laws, around a
uniform state from its laws, the growth rate of each wavenumber of the model's discretization.

#### Scenario: Turing band
- **WHEN** the dispersion relation of the Brusselator above its Turing threshold is computed
- **THEN** the band of growing wavenumbers matches the analytic band within one grid wavenumber
