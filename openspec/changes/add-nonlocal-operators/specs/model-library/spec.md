# Spec Delta

## ADDED Requirements

### Requirement: Nonlocal field models
The library SHALL provide, as registered factories built on the general interactions and the nonlocal operators:
- `amari_neural_field` under `biology/neuroscience` (2112.12147 §1.3.1);
- `double_gyre` under `earth/oceanography` (1910.00583 §III.6.1);
- `gkt_traffic` under `engineering/traffic` (cond-mat/0012229 §III.E.4);
- `active_viscoelastic_gel` under `physics/active_matter` (2306.08868 VI).

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
  whose downstream fronts move upstream at between 10 and 25 kilometres per hour

#### Scenario: Contractile gel
- **WHEN** the active viscoelastic gel runs from a small perturbation of uniform cell density above its contractile
  threshold
- **THEN** the density concentrates into peaks, and the velocity at every step solves the force balance within 1e-10

### Requirement: Field couplings are interactions
`legion` and `nanotube_gap_junction_exchange` SHALL write the coupling of their fields with `CouplingInteraction`, so
`TermanWangElement` and `ExchangingCellElement` hold only the dynamics of one site, and SHALL keep their trajectories.

#### Scenario: Rebuilt couplings keep the trajectories
- **WHEN** `legion` and `nanotube_gap_junction_exchange` run with their catalog defaults
- **THEN** the trajectory of `nanotube_gap_junction_exchange` equals that of the previous elements within 1e-10, and the
  rates of `legion` equal those of the previous elements within 1e-12 at many random states, with its trajectories
  equal within 1e-10 over a short window before rounding grows
