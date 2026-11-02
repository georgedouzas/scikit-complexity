# Spec Delta

## MODIFIED Requirements

### Requirement: Noise follows a declared convention
Each noise law SHALL declare Itô, Stratonovich, or anti-Itô noise through its convention, and the framework SHALL add
the drift correction for the convention, with Milstein and Heun steppers available. The correction SHALL cover shared
and correlated Wiener processes and states that several noise laws drive. Laws that share one process SHALL declare
one convention, and validation SHALL report laws that share a process with different conventions.

#### Scenario: Geometric Brownian motion in Stratonovich form
- **WHEN** `dX = μ X dt + σ X ∘ dW` runs with a noise law declared `convention='stratonovich'` over an ensemble
- **THEN** the mean of `X` at time 1 matches `exp(μ + σ² / 2)` within 2 standard errors

#### Scenario: Cross terms of a shared process
- **WHEN** `dX = σ Y ∘ dW` and `dY = σ X ∘ dW` share one process in one element, both laws declared Stratonovich, and
  run over an ensemble
- **THEN** the mean of `X + Y` at time 1 matches `(X₀ + Y₀) exp(σ² / 2)` within 2 standard errors

#### Scenario: Several laws on one state
- **WHEN** two Stratonovich noise laws with independent processes drive one state
- **THEN** the system compiles, and its drift correction is the sum of both laws' corrections

#### Scenario: Mixed conventions on one process
- **WHEN** an Itô noise law and a Stratonovich noise law name the same process
- **THEN** validation reports one issue that names both laws and the process, with the fix of one convention for both

#### Scenario: Amplitude outside the law's scope
- **WHEN** a Stratonovich noise law depends on a state that another component drives through a shared process
- **THEN** validation reports an issue that names both laws and suggests the Itô convention or one component
