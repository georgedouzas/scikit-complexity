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
  `DiscreteTime()`, whose `Update` laws align each heading with the mean heading of the neighbours of each species
  within a radius, `neighbors(..., over=Radius(r), reduce='mean')`, with a species coupling that differs one way from
  the other;
- `tipping_cascade_network`: a group of tipping elements on a `NetworkSpace` under `ContinuousTime()`, with rate laws
  coupled over neighbours.

### Dependency on neighbourhood reductions

A pair interaction takes only rate laws, which need continuous time, so the discrete Vicsek rule cannot be written as
one. The neighbourhood reductions of `add-adaptive-networks` (its task 2.1) are the general primitive it needs: a law
of any time class reduces an expression over the neighbours within a radius. `nonreciprocal_vicsek` is built after
that task lands, and the other three probe models do not depend on it.

## Risks / Trade-offs

- [Cloning needs large populations] → The function warns when the population is small for the bias.
