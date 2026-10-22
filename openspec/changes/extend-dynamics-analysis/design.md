# Design

## Context

The stochastic analysis has entropy production and large deviations from tilted generators, the field analysis has structure factors, and the symbolic layer linearises laws. The new measures extend these.

This design is rebased on `redesign-system-model`. Fields are site groups on a lattice, systems declare their time,
and `Simulation` validates a system before choosing its solver. The analyses add no primitive, solver, or backend, so
they need no declaration.

## Goals / Non-Goals

**Goals:**
- Sample rare events where tilted generators are too large.
- Predict pattern wavelengths from a model's laws before running it.

**Non-Goals:**
- Transition path sampling, which a later change may add.

## Decisions

### Cloning

The cloning algorithm of Giardinà, Kurchan, and Peliti with a fixed population, run through the simulation's steppers and seeds, estimates the scaled cumulant generating function; its bias shrinks with population size, which the function reports.

Cloning runs through `Simulation`, so the system is validated, and its solver and backend chosen, before any clone
is advanced.

### Dispersion relations

The symbolic layer linearises the field laws around a uniform state and substitutes Fourier modes, so the growth rate of each wavenumber is the largest eigenvalue of a small matrix per wavenumber, exact for the model's discretization.
The field laws are the rate laws of the site groups on a `LatticeSpace`, and the discretization is that lattice's
spacing and neighbourhood.

### Probe models

The probe models use the redesigned vocabulary and existing primitives only:
- `kinouchi_copelli`: a fixed group of excitable elements on a random `NetworkSpace` under `DiscreteTime()`, with
  update laws over neighbours;
- `nonreciprocal_cahn_hilliard`: two site groups on a periodic `LatticeSpace` under `ContinuousTime()`, with rate laws
  that couple them nonreciprocally;
- `nonreciprocal_vicsek`: two groups of self-propelled elements in a `ContinuousSpace(boundary='periodic')` under
  `DiscreteTime()`, aligned by an interaction with the binding `'pairs'` within a cutoff;
- `tipping_cascade_network`: a group of tipping elements on a `NetworkSpace` under `ContinuousTime()`, with rate laws
  coupled over neighbours.

## Risks / Trade-offs

- [Cloning needs large populations] → The function warns when the population is small for the bias.
