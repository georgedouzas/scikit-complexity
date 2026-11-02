# Design

## Context

The inventory records the four models as pending this change, with tags that predate the general rewrite:
`szabo_cells` and `active_passive_flocking` carry a network and an edge group, which the systems they build do not
have. The general classes they now build on are listed in the proposal. `ModeDecompositionAnalysis` fits the exact DMD
of the one-sample map `g(t + dt) = A g(t)` with PyDMD, on the variables or on the observables of a scikit-learn
dictionary, and its `system_` follows `dg/dt = L g` under continuous time. It takes no inputs.

## Goals / Non-Goals

**Goals:**
- Build the four models from the general classes, with tests of their known behaviour.
- Add the one general operation the Koopman controller needs, a mode decomposition with control inputs.

**Non-Goals:**
- A receding-horizon solver in the engine. The controller is a discrete-time partition of the model, which
  `CoSimulationSolver` already runs beside the plant.
- Any other new core class. A model that turns out to need one stops, and its need becomes a change of its own.

## Decisions

### Mode decomposition with control

`ModeDecompositionAnalysis(inputs=None)` takes the names of the variables that are control inputs. With inputs, it fits
`g(t + dt) = A g(t) + B u(t)` with PyDMD's `DMDc` on the pairs of consecutive samples, where `g` holds the observables
of the dictionary computed from the other variables, and it sets `operator_` and `input_matrix_`, the matrices `A` and
`B` of the one-sample map. Its `system_` follows `dg/dt = L g + M u` with `L` from the eigenvalues of `A` as today and
`M` the input matrix that gives the same one-sample map under an input held over the sample, and it declares every input
as a bounded adaptive parameter of the surrogate element, bounded by the range of the inputs in the samples. PyDMD is
already a dependency of the analysis layer, and its `DMDc` is the mature implementation, so nothing is written in house
beyond the adapter. The audit of 2026-11-01 recorded that PyDMD's last release, 2025.8.1, is of 2025-08-01; the
manifest already requires `pydmd>=2025.8.1`, and the risk is recorded below.

### szabo_cells

Szabó et al. (2006) cells in a periodic `ContinuousSpace`: a group of cells whose position `x`, a vector of order 1,
moves at `v0 n + mu F`, where `n` is the unit vector of the heading `theta` and `F` the sum of the pair forces over the
neighbours within twice the cell radius, `neighbors(force, over=RadiusNeighborhood(2 R))`. The heading turns towards the
velocity, `theta' = asin((n × v / |v|) · e_z) / tau`, with angular noise. Both are `RateInteraction`s, the second with
`noise`, and both read the same reduction, so the velocity needs no state of its own. The force is the linear
repulsion and attraction of the paper, a function of the pair namespace written in the package.

### active_passive_flocking

The model of Silverberg et al. (2013) as Charbonneau (2017, ch. 10) writes it: two groups of `PointElement`s in a
periodic box, active and passive. Both feel a soft repulsion between overlapping agents, a `PairCouplingInteraction`
with a potential, and noise. Only the active agents propel themselves towards the speed `v0`, a `RateInteraction`, and
align with the mean velocity of the agents within the flocking radius, a `RateInteraction` reading
`neighbors(velocity, over=RadiusNeighborhood(r_f), reduce='mean')`. The system runs under continuous time with
Euler-Maruyama steps.

### koopman_model_predictive_control

Korda and Mezić (2018), as Brunton et al. (2021, §6.1.1 and §6.3) review it, for the Duffing oscillator, the Van der Pol
oscillator, and a bilinear DC motor, chosen by the variant `plant`:
- The factory runs the plant with a bounded adaptive control input held at random values over random intervals, and
  fits `ModeDecompositionAnalysis(dictionary=..., inputs=('plant.u',))` to the runs.
- The returned system nests the plant under `ContinuousTime` and a controller element under `DiscreteTime` with the
  sample step. The controller's `MapInteraction` sets its state `u` to a `CallSpec` of the plant states. The called
  function lifts the states with the dictionary, condenses the quadratic tracking cost over the horizon with `A` and
  `B`, and solves the bounded least-squares problem with `scipy.optimize.lsq_linear`, returning the first input.
- The plant reads `u` from the controller through a role, and `CoSimulationSolver` holds it between samples.

The predictive problem lives in the model package as a private class, because it is the one model that uses it.

### next_generation_reservoir

Gauthier et al. (2021), as the reservoir review of 2509.11338 writes it: an element whose variables keep `delay - 1`
lags, `StateSpec(lags=...)`, and one `MapInteraction` per variable in the increment mode, whose function is the ridge
readout of the constant, linear, and quadratic monomials of the variables and their lags. The factory does not fit a
second readout. It fits `ReservoirAnalysis(kind='next_generation', delay=..., order=..., ridge=...)` of
`skcomplex.analysis` on a trajectory of the Lorenz flow, which the package builds from `RateInteraction`s since model
packages do not import each other, or on a series the caller passes. That analysis already fits ReservoirPy's `NVAR`
features and `Ridge` readout in `src/skcomplex/analysis/_reservoirs.py`, and the factory reads the fitted readout from
its `step_`: the weights and bias of the `Ridge` node, the standardisation `mean` and `scale`, and the order of the
`NVAR` monomials. It writes them as the symbolic increment of every variable over its lags, so the model is a readable
system of the core while the fit stays in one place. A test checks that the forecast of the system agrees with the
`forecast_` of the same analysis.

## Risks / Trade-offs

- [Fitting inside a factory makes it slow] → The default runs are short, and the factory takes the training series or
  the fitted readout as arguments to skip the fit.
- [The readout depends on the feature layout of ReservoirPy's `NVAR`] → The factory reads the monomial order from the
  fitted node rather than assuming it, and the agreement scenario fails if the layouts diverge.
- [PyDMD has not released since 2025-08-01] → `DMDc` is a small, stable part of PyDMD, the adapter is the one place
  that calls it, and the linear system scenario detects a break.
- [The surrogate of a controlled plant may track poorly] → The scenario compares the controlled with the uncontrolled
  plant rather than fixing an error bound, as the review does.
