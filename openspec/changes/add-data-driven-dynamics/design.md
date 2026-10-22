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

The survey, run against the project's NumPy 2 stack:

- PySINDy 2.1 is chosen. It depends only on NumPy, SciPy, scikit-learn, and `derivative`, and its
  `PolynomialLibrary.powers_` gives the exponents of every feature, so the adapter builds the Sage monomials from them
  without parsing feature names.
- PyDMD 2025.8 is chosen. Its `DMD.fit(X, Y)` takes separate snapshot pairs, so several series fit together, and it
  exposes exact modes, eigenvalues, and one-step predictions. It brings Matplotlib and xarray as dependencies.
- PyKoopman 1.2 is rejected. It pins NumPy below 1.27, scikit-learn 1.1.3, SciPy 1.11, PyDMD 0.4, and PyTorch 2.1, which
  the project's NumPy 2.1 floor excludes. Extended DMD instead lifts the variables with any scikit-learn transformer as
  the dictionary, such as `PolynomialFeatures`, and runs PyDMD on the observables, which is EDMD by definition.
- ReservoirPy 0.4 is chosen. Its `NVAR`, `Reservoir`, and `Ridge` nodes give next-generation reservoirs, echo state
  networks, and the ridge readout, and both reservoirs expose their memory, the window `NVAR.store` and the unit
  activities `Reservoir.state['out']`, as public attributes.

All three are runtime dependencies, imported on first use through `skcomplex.core.analysis._lazy.module`.

### Surrogates as elements

A linear surrogate becomes an element with a matrix `Rate` law, and a SINDy surrogate an element with the found `Rate`
laws, each in a `System` with `ContinuousTime()`. A reservoir surrogate becomes an element whose `Update` law calls the
trained readout through a numeric function, in a `System` with `DiscreteTime()`. Each surrogate system has no space and
no interactions, and it validates and selects its solver and backend like any system. A surrogate fitted to discrete
samples of a continuous system keeps continuous time, and its sampling step is a parameter of the forecast, not of the
system.

Decisions taken while applying the change:

- The fits take a trajectory, a `Series`, or a sequence of them, with `names` choosing the variables of trajectories.
  Every series must be sampled at one uniform step, and all must share the variables and the step.
- The states of a surrogate take the last dotted part of every label, such as `x` for `lorenz.x`, or the whole label
  made an identifier where those clash.
- The linear surrogate holds one matrix `Rate` law, `dg/dt = L g` with `L = Phi diag(ln(mu) / dt) Phi^+`, on a vector
  target of its scalar states. The sparse surrogate holds one `Rate` law per variable, a sum of monomials. Both start at
  the first sample of the first series, so a run reproduces the data.
- The reservoir surrogate holds the variables and then the memory of the reservoir as scalar states, `memory_0` and
  on. One `Update` law on the vector of all states calls the trained map through a `Call`, so the system carries its
  whole state and restarts like any other. It starts at the last sample of the last series with the memory the series
  left, so a run is the autonomous forecast. The readout gives the increment of the standardized variables over one
  sample, and the default ridge penalty is 1e-4: without the standardization, a next-generation reservoir fitted to
  accurate Lorenz samples diverged within a tenth of a Lyapunov time.
- A `Call` on a vector target used to run once per component in the NumPy kernels, so a map over `n` states ran `n`
  times per step. The kernels now share the evaluation of the outputs of one call, as the Numba kernels already did,
  which the spec's "each step applies the readout once" needs.
- The errors are relative errors of the predicted increments over one sample for DMD and reservoirs, and of the
  predicted derivatives for SINDy. `validation`, 0.2 by default, holds out the end of every series, and the validation
  error is NaN when nothing is held out.

## Risks / Trade-offs

- [Fitted models can be unstable outside the data] → Fitting reports the training and validation errors, and examples forecast within the attractor only.
