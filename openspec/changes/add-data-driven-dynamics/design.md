# Design

## Context

Trajectories expose states as arrays through the adapters, and the compiler accepts symbolic laws. A fitted model is useful only if it runs through the same interface as a hand-written one.

This design is rebased on `redesign-system-model`: a system is its components, interactions, space, and time, and
`Simulation` validates it before choosing a solver and backend. Surrogates are plain systems in that vocabulary, so they
need no declaration of their own.

## Goals / Non-Goals

**Goals:**
- Fit linear, sparse symbolic, and reservoir surrogates to trajectories.
- Return surrogates that `Simulation` runs, so forecasts and comparisons use the existing tooling.

**Non-Goals:**
- Gradient training of model parameters, which `add-learning-dynamics` covers.

## Decisions

### Libraries

PySINDy (maintained by the Brunton and Kutz groups) returns sparse symbolic models; the surrogate converts its feature library and coefficients to Sage expressions, so the compiler lowers them like any law. PyDMD covers DMD variants and PyKoopman covers EDMD with dictionaries; the survey picks one. ReservoirPy provides echo state networks and next-generation reservoirs.

### Surrogates as elements

A linear surrogate becomes an element with a matrix `Rate` law, and a SINDy surrogate an element with the found `Rate`
laws, each in a `System` with `ContinuousTime()`. A reservoir surrogate becomes an element whose `Update` law calls the
trained readout through a numeric function, in a `System` with `DiscreteTime()`. Each surrogate system has no space and
no interactions, and it validates and selects its solver and backend like any system. A surrogate fitted to discrete
samples of a continuous system keeps continuous time, and its sampling step is a parameter of the forecast, not of the
system.

## Risks / Trade-offs

- [Fitted models can be unstable outside the data] → Fitting reports the training and validation errors, and examples forecast within the attractor only.
