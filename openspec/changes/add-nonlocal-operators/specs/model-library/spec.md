# Spec Delta

## ADDED Requirements

### Requirement: Nonlocal field models
The library SHALL provide, as registered factories:
- the Amari neural field, under `biology/neuroscience` (2112.12147 §1.3.1);
- the quasi-geostrophic double gyre, under `earth/oceanography` (1910.00583 §III.6.1);
- the gas-kinetic-based non-local traffic model, under `engineering/traffic` (cond-mat/0012229 §III.E.4).

#### Scenario: Amari bump
- **WHEN** the Amari neural field `tau u_t = -u + convolve(H(u), w) + h` runs on a line with a lateral-inhibition
  kernel `w`, a Heaviside firing rate, and a resting level `h < 0`, from a localized excitation
- **THEN** a stationary bump forms, and its width is within 5% of the root `a` of `W(a) + h = 0` with `w(a) < 0`,
  where `W(a)` is the integral of `w` from 0 to `a`

#### Scenario: Double gyre circulation
- **WHEN** the quasi-geostrophic double gyre runs in a closed square basin from rest under the symmetric double-gyre
  wind stress `-tau0 cos(2 pi y / L)` in its viscous, steady regime
- **THEN** the streamfunction forms two gyres of opposite sign, one on each side of the mid-basin latitude, with each
  extremum in the western fifth of the basin, and along the central latitude of each gyre the interior streamfunction
  is within 10% of the Sverdrup balance `beta psi_x = curl(tau)`

#### Scenario: Stop-and-go waves
- **WHEN** the gas-kinetic-based traffic model runs on a ring road with a small local density perturbation, once at a
  density of 15 vehicles per kilometre and once at a density in its unstable range
- **THEN** the perturbation decays at the low density, and at the unstable density it grows into congested clusters
  whose downstream fronts move upstream at between 10 and 20 kilometres per hour
