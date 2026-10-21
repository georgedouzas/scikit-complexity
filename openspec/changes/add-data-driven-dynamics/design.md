# Design

## Context

Trajectories expose states as arrays through the adapters, and the compiler accepts symbolic laws. A fitted model is useful only if it runs through the same interface as a hand-written one.

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

A linear surrogate becomes an element with a matrix rate law, a SINDy surrogate an element with the found laws, and a reservoir surrogate an element whose update calls the trained readout through a numeric function.

## Risks / Trade-offs

- [Fitted models can be unstable outside the data] → Fitting reports the training and validation errors, and examples forecast within the attractor only.
