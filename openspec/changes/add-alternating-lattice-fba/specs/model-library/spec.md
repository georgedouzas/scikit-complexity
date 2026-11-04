# Spec Delta

## MODIFIED Requirements

### Requirement: COMETS model

The library SHALL provide `comets`, dynamic FBA on a 2D lattice in which metabolites and biomass diffuse, and only sites
that hold biomass are solved, through the activity of the linear program law.

#### Scenario: Expanding colony

- **WHEN** `comets` runs from a central inoculum on uniform glucose
- **THEN** the colony radius grows with a nearly constant speed after a transient. The glucose consumed over the lattice
  matches the integral of uptake flux times biomass over sites and time within 1%, and sites without biomass have zero
  fluxes.

## ADDED Requirements

### Requirement: COMETS alternates growth and diffusion

`comets` SHALL alternate a growth step, in which each site solves its linear program, and a diffusion step, with the
coefficients, time step, and inoculum of Harcombe et al. (2014), and each default SHALL name its source.

#### Scenario: Mass balance

- **WHEN** `comets` runs on a closed lattice
- **THEN** the glucose taken up over the run equals the glucose consumed by growth, within 1%

#### Scenario: Stable at the published step

- **WHEN** `comets` runs with steps of 0.1 h at the published diffusion coefficients
- **THEN** biomass and glucose stay finite and non-negative within 1e-9 at every frame

#### Scenario: Colony grows from its centre

- **WHEN** `comets` runs from the inoculum at the centre of the lattice
- **THEN** the biomass stays symmetric about the centre within 1e-6, and the radius of the colony grows over the run
