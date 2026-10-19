# Proposal

## Why

Update laws give deterministic maps only. A law cannot draw a random number from the simulation's random stream, so
the framework has no discrete-time stochastic process and no discrete-time Markov chain. Lattice rules and Mesa rules
draw random numbers, but only on lattices and agents. The model inventory of `organize-model-library` records three
models as blocked on this gap:

- the persistent random walk, whose step keeps its direction with a given probability (Jensen 2023 §10.4);
- Markov and hidden Markov sources, whose symbols follow a transition matrix or are emitted from a hidden chain
  (Lindgren 2024 §3.2);
- iterated function systems drawn by the chaos game, which applies a randomly chosen affine map at each step
  (Dilão 2023 §5.4).

## What Changes

- Add random draws to the symbolic layer: uniform, normal, Bernoulli, and categorical draws, each named, that update
  laws use like any other expression. A draw takes a fresh value from the simulation's random stream at every step,
  one per member of a group, and every use of the same name in the same component and step reads the same value.
- Add discrete-time Markov chains: a transition law moves a state over a finite alphabet with the row of a transition
  matrix that its current value selects, and an emission law sets an observed state from the row of an emission matrix
  that a hidden state selects, which gives hidden Markov sources.
- Check transition and emission matrices: entries are nonnegative and rows sum to one, at compile time and when a
  running simulation changes a parameter that appears in a matrix.
- Compile stochastic maps on the NumPy and Numba backends, and run them with the `iteration` stepper, which passes the
  simulation's random generator to the compiled update. Runs with the same random state give the same trajectory.
- Reject draws in laws other than update laws, with a named error that points to `Noise` and `Jump` for continuous
  time.
- Add library models: the persistent random walk, Markov and hidden Markov sources, and iterated function systems
  drawn by the chaos game.

## Capabilities

### New Capabilities

- `stochastic-maps`: random draws in update laws, discrete-time Markov chains with transition and emission laws, the
  matrix checks, and reproducible compiled runs.

### Modified Capabilities

- `model-library`: the persistent random walk, Markov and hidden Markov sources, and chaos-game iterated function
  systems, which this change unblocks.

## Impact

- `skcomplex.core.model` gains the draw functions `uniform`, `normal`, `bernoulli`, and `categorical`, and the laws
  `Transition` and `Emission`. The existing `ValidationError`, `ParameterError`, and `IncompatibleLawsError` of
  `skcomplex.core.base` report invalid matrices and misplaced draws, so no new exception is needed.
- `skcomplex.core.compile` lowers draws and Markov laws to NumPy and Numba kernels that take a
  `numpy.random.Generator`. The `Problem` records whether its update is stochastic, and the `iteration` stepper in
  `skcomplex.core.solvers` passes its generator through.
- No new runtime dependency: draws are taken from NumPy's `Generator`, which Numba accepts inside compiled kernels.
  `quantecon` becomes a test dependency, used as the reference for stationary distributions.
- `skcomplex.models.mathematics.stochastic_processes` gains the persistent random walk and the Markov and hidden
  Markov sources, and `skcomplex.models.mathematics.fractals` gains the chaos game. The inventory marks the three
  entries as implemented.
