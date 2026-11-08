## MODIFIED Requirements

### Requirement: Lyapunov spectra of laws with absolute values
`lyapunov_spectrum` SHALL differentiate `abs(x)` as `sign(x)`, so laws with absolute values have a spectrum.

#### Scenario: Sprott jerk system
- **WHEN** a user computes the spectrum of `SprottJerkSystem` until 200
- **THEN** it returns three finite exponents whose sum is negative
