# Spec Delta

## ADDED Requirements

### Requirement: Tacrolimus population pharmacokinetics
The library SHALL provide `tacrolimus_population_pk`, a group of patients with four transit compartments, central and
peripheral compartments, oral doses at given times, covariates, and log-normal random effects, after 2602.03215. It
SHALL have mechanistic forms with linear or saturable elimination and a latent neural form, a gallery example, and an
inventory entry.

#### Scenario: Superposition of doses
- **WHEN** the mechanistic form with linear elimination runs one patient without random effects for ten doses 12 hours
  apart
- **THEN** the central concentration equals the superposition of the single-dose response within 1e-8

#### Scenario: Covariate on clearance
- **WHEN** two patients that differ only in their genotype covariate run the same doses
- **THEN** the clearance of the expresser is that of the non-expresser times the multiplier of the system, and its area
  under the curve is smaller

#### Scenario: Reproducible population
- **WHEN** the system is built twice with the same `random_state`
- **THEN** the drawn covariates and random effects are equal

### Requirement: Oral and intravenous pharmacokinetics of compounds
The library SHALL provide `oral_iv_pk_hybrid`, a group of compounds, each given orally and intravenously, with gut
absorption, hepatic first pass, and elimination, whose rates follow from descriptors through a fixed reference map or a
neural map, after 2310.09167, with a gallery example and an inventory entry.

#### Scenario: Bioavailability from areas
- **WHEN** the reference form runs one compound until its concentrations vanish
- **THEN** the observable bioavailability equals the ratio of the oral to the intravenous area under the curve, scaled
  by the doses, and equals the product of the absorbed fraction and the hepatic escape fraction within 1e-6

#### Scenario: Untrained neural form
- **WHEN** the neural form runs with the weights of its last layer at zero
- **THEN** every compound has the typical rates of the system, and its concentrations equal those of the reference form
  for a compound whose descriptors map to the typical rates, within 1e-8
