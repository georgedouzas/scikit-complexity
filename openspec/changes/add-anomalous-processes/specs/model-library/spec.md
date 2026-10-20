# Spec Delta

## ADDED Requirements

### Requirement: Anomalous random walk models
The library SHALL provide, as registered factories under the field path `physics/soft_matter`:
- the continuous-time random walk, Gaussian jumps after power-law waiting times (Argun et al. 2021 §6.3, Metzler and
  Klafter 2000);
- the Lévy walk, flights at constant speed in random directions for power-law flight times (Argun et al. 2021 §6.3,
  Zaburdaev, Denisov, and Klafter 2015).

#### Scenario: Subdiffusive random walk
- **WHEN** the continuous-time random walk with waiting-time exponent `α = 0.5` runs for 10⁴ walkers to time 10⁴
- **THEN** the ensemble mean squared displacement grows with a scaling exponent within 0.1 of 0.5

#### Scenario: Superdiffusive Lévy walk
- **WHEN** the Lévy walk with flight-time exponent `σ = 1.5` runs for 2000 walkers to time 2000
- **THEN** the ensemble mean squared displacement grows with a scaling exponent within 0.1 of `3 - σ = 1.5`

### Requirement: Anomalous diffusion models with memory or heterogeneity
The library SHALL provide, as registered factories under the field path `physics/soft_matter`:
- annealed transient time motion, Brownian motion whose diffusion coefficient is redrawn at renewals with waiting
  times that depend on it (Argun et al. 2021 §6.3, Massignan et al. 2014);
- fractional Brownian motion of a given Hurst index (Argun et al. 2021 §6.1, Mandelbrot and Van Ness 1968).

#### Scenario: Fractional Brownian motion
- **WHEN** the fractional Brownian motion model with Hurst index 0.3 runs for 1000 members
- **THEN** the ensemble mean squared displacement grows with a scaling exponent within 0.05 of 0.6, and the Hurst
  exponent of the increments by detrended fluctuation analysis lies within 0.05 of 0.3

#### Scenario: Annealed transient time motion
- **WHEN** annealed transient time motion with diffusion coefficients of density `P(D) ~ D^(σ-1)` for `σ = 0.5`, and
  waiting times of mean `D^(-γ)` for `γ = 1`, runs for 10⁴ walkers to time 2000
- **THEN** the ensemble mean squared displacement grows with a scaling exponent within 0.1 of `σ / γ = 0.5`, as
  derived by Massignan et al. (2014)

### Requirement: No inventory model is blocked by anomalous processes
The model inventory SHALL mark the continuous-time random walk, the Lévy walk, annealed transient time motion, and
fractional Brownian motion in `physics/soft_matter` as implemented.

#### Scenario: Inventory status
- **WHEN** the inventory is read after this change
- **THEN** no entry has the status `blocked add-anomalous-processes`, and each of the four models resolves to a
  registered factory under `physics/soft_matter`

### Requirement: Lévy flights and trap models
The library SHALL provide `levy_flight` in `physics/soft_matter` and `trap_model` in `physics/statistical_mechanics`, as Bertin 2021 poses them and the model inventory records them, built on this
change's capability, each with a gallery example.

#### Scenario: Registered under the inventory names
- **WHEN** the change is applied and the inventory check runs
- **THEN** both models are registered under their field paths and marked implemented

