## ADDED Requirements

### Requirement: Event loops run serial kernels
The Numba backend SHALL compile the kernels that an event loop calls once per event as serial code, whatever its
parallel setting, and SHALL keep parallel loops for the drift and pair kernels of large groups.

#### Scenario: Compiled Gillespie with the default backend
- **WHEN** a stochastic epidemic of 1000 individuals runs with the compiled Gillespie stepper on a Numba backend with
  the default parallel setting and on one with `parallel=False`, after compilation
- **THEN** the default backend takes at most 1.5 times as long as the serial one, and both give the same trajectory
  for the same random state

### Requirement: Kernels read the layout without per-call arithmetic
Compiled numeric functions SHALL compute the positions and shapes of their variables once, when they are compiled,
and SHALL NOT recompute them at every evaluation.

#### Scenario: Small system of many elements
- **WHEN** the NumPy drift of 20 populations, each with its own logistic growth interaction, is evaluated after
  compilation
- **THEN** one evaluation takes at most 250 microseconds and returns the logistic rate of every population
