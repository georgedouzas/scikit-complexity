# Spec Delta

## MODIFIED Requirements

### Requirement: Noise follows a declared convention
Systems SHALL declare Itô, Stratonovich, or anti-Itô noise, and the framework SHALL add the drift correction for the
convention, with Milstein and Heun steppers available. The correction SHALL cover shared and correlated Wiener
processes and states that several noise laws drive. Milstein SHALL reject noise that is not diagonal.

#### Scenario: Geometric Brownian motion in Stratonovich form
- **WHEN** `dX = μ X dt + σ X ∘ dW` runs as Stratonovich over an ensemble
- **THEN** the mean of `X` at time 1 matches `exp(μ + σ² / 2)` within 2 standard errors

#### Scenario: Cross terms of a shared process
- **WHEN** `dX = σ Y ∘ dW` and `dY = σ X ∘ dW` share one process in one element and run as Stratonovich over an
  ensemble
- **THEN** the mean of `X + Y` at time 1 matches `(X₀ + Y₀) exp(σ² / 2)` within 2 standard errors

#### Scenario: Several laws on one state
- **WHEN** two noise laws with independent processes drive one state in Stratonovich form
- **THEN** the system compiles, and its drift correction is the sum of both laws' corrections

#### Scenario: Amplitude outside the law's scope
- **WHEN** a Stratonovich noise law depends on a state that another component drives through a shared process
- **THEN** compiling raises a named error that names both laws and suggests the Itô form or one component

#### Scenario: Milstein with a shared process
- **WHEN** `Milstein` integrates a system in which one process drives two states
- **THEN** it raises a named solver error that names the process and suggests `EulerMaruyama` or `Heun`
