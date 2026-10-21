# Design

## Context

The stochastic analysis has entropy production and large deviations from tilted generators, the field analysis has structure factors, and the symbolic layer linearises laws. The new measures extend these.

## Goals / Non-Goals

**Goals:**
- Sample rare events where tilted generators are too large.
- Predict pattern wavelengths from a model's laws before running it.

**Non-Goals:**
- Transition path sampling, which a later change may add.

## Decisions

### Cloning

The cloning algorithm of Giardinà, Kurchan, and Peliti with a fixed population, run through the simulation's steppers and seeds, estimates the scaled cumulant generating function; its bias shrinks with population size, which the function reports.

### Dispersion relations

The symbolic layer linearises the field laws around a uniform state and substitutes Fourier modes, so the growth rate of each wavenumber is the largest eigenvalue of a small matrix per wavenumber, exact for the model's discretization.

## Risks / Trade-offs

- [Cloning needs large populations] → The function warns when the population is small for the bias.
