# Design

## Context

See proposal.md for the motivation. The relevant parts of the core today:
- `Update(target, value)` in `skcomplex.core.model` gives the discrete map form. The compiler prints its Sage
  expressions to a NumPy function `update(t, x, p)` and to a Numba kernel, and the `iteration` stepper in
  `skcomplex.core.solvers` applies it once per integer step. The stepper receives the simulation's
  `numpy.random.Generator` in `initialize` but never uses it.
- Randomness already flows from `Simulation(random_state=...)` to the `euler-maruyama` and `gillespie` kernels, which
  take the generator as an argument and call `rng.normal`, `rng.exponential`, and `rng.uniform` inside Numba code.
  Lattice rules and Monte Carlo samplers draw from the same generator. The experiments runner spawns one stream per run
  from a `SeedSequence`.
- States take categorical domains, `Domain.categorical(...)`, stored as label indices, and lags, so a state over a
  finite alphabet and a map that reads past values exist already.
- `where`, `either`, and `both` give piecewise expressions, which is how a chaos game picks one affine map among
  several.
- `skcomplex.core.analysis` has `entropy_rate`, `block_entropies`, and `box_counting_dimension`, which the model
  scenarios use.

## Goals / Non-Goals

**Goals:**

- Write a discrete-time stochastic process as update laws, symbolically, as the references write it, and compile it to
  the same NumPy and Numba kernels as a deterministic map.
- Share one random stream between draws, the other stochastic steppers, and the experiments runner, so seeds keep their
  meaning.
- Express Markov chains and hidden Markov sources with their matrices, without the user writing the inverse cumulative
  lookup.

**Non-Goals:**

- Inference for Markov models, such as fitting transition matrices or decoding hidden states. A later analysis change
  can wrap `hmmlearn` for that.
- Continuous-time Markov chains. Jump laws and the `gillespie` stepper cover them.
- Random draws inside ODE or SDE laws. Noise and jump laws already model randomness in continuous time.

## Decisions

### Library survey

The constitution requires a mature library where one fits. The feature has two parts, drawing random variates inside
compiled update kernels, and simulating Markov chains. The survey, checked in October 2026:

| Library | Maturity and licence | Wheels for Linux and macOS, Python 3.12 and 3.13 | Fit | Verdict |
|---|---|---|---|---|
| NumPy `Generator` with PCG64 | The standard, BSD, already a dependency | Yes | Vectorised draws of every distribution needed, the stream the simulation already owns | Wrapped for the NumPy backend |
| Numba support for `numpy.random.Generator` | Numba 0.56 and later, BSD, already a dependency | Yes | Compiled kernels accept a `Generator` argument and call `random`, `normal`, `uniform`, `integers`, and more, with NumPy's algorithms | Wrapped for the Numba backend |
| `quantecon` `MarkovChain` | Active, 0.13.1 released October 2026, MIT, Numba based | Pure Python wheel | Simulates whole paths of one chain from its own seed, with no coupling to other states within a step and no live parameters | Rejected as the engine, used in tests as the reference for stationary distributions |
| `PyDTMC` | Active, 10.0.0 released October 2026, MIT, pure Python | Pure Python wheel | Analysis and path simulation of one chain, and it pulls in matplotlib | Rejected |
| `hmmlearn` | Limited maintenance, 0.3.3 released October 2024, BSD | Yes, cp312 and cp313 | Fits and decodes hidden Markov models, and samples whole sequences from its own random state | Rejected for simulation, a candidate for a future inference change |
| `pomegranate` | Active, MIT | Yes | Probabilistic models on PyTorch, a heavy dependency for categorical draws | Rejected |
| SciPy `stats` distributions | The standard, BSD, already a dependency | Yes | Draws through NumPy, but cannot be called inside Numba kernels | Rejected for kernels |

The path simulators fail on the same point. A stochastic map in this framework is one step of a coupled system: the
persistent walker's position reads the same draw as its direction, an emission reads the hidden state of the same
step, and matrices hold parameters that change while the simulation runs. A library that produces whole paths cannot
take part in that step. The core therefore keeps only the adapter that lowers named symbolic draws to calls on NumPy's
`Generator`, in NumPy code and in Numba code. No new runtime dependency is added, and `quantecon` joins the test
dependencies.

### Draws are named symbolic functions

`uniform(name, low=0, high=1)`, `normal(name, mean=0, deviation=1)`, `bernoulli(name, p)`, and
`categorical(name, weights)` are Sage `BuiltinFunction`s in `skcomplex.core.model`, like `where` and `mod`. They never
evaluate and have no derivative. The name is part of the call, so Sage's simplification and the printer's common
subexpression elimination merge two uses of one name, which is the declared meaning, and keep different names apart.
An anonymous draw was rejected, because Sage would merge `u() + u()` into `2 u()`. A draw declared as a component field
was rejected, because a distribution's arguments often depend on the current state, as a transition row does.

### Lowering

The compiler collects the draws of each component and checks that every name has one distribution and one argument
list. It replaces each draw by a fresh scalar symbol per component, or a group variable per group, filled at the start
of the step. The draw block runs before the update expressions in a fixed order, by component path, draw name, and
member index:
- uniform and Bernoulli draws use `rng.random(n)`, compared or scaled;
- categorical draws use `rng.random(n)` and a search in the cumulative weights, as the compiled `gillespie` kernel does
  already;
- normal draws use `rng.normal(mean, deviation, n)`.

The NumPy backend draws one vector per name. The Numba backend loops in the same order and calls the same `Generator`
methods, so uniform-based draws give equal trajectories on both backends. `Problem` gains `stochastic: bool`, the
update signature becomes `update(t, x, p, rng)` when it is set, and the `iterate` kernel takes the generator. The
`iteration` stepper passes its generator through. A deterministic map keeps its present signature and kernel, so
existing models and their cached kernels are unchanged.

### Transition and emission laws

`Transition(target, matrix)` and `Emission(target, source, matrix)` are update laws in `skcomplex.core.model`. The
compiler rewrites them before printing:
- a transition becomes `Update(target, categorical(key, row))`, where the row is the matrix row the target's current
  value selects;
- an emission becomes `Update(target, categorical(key, row))`, where the row is the one the source's new value selects,
  obtained by substituting the source's update expression.

The key is derived from the law's path, so it never collides with a user's draw name. Matrices of numbers lower to
constant arrays. Matrices with parameters lower to an array assembled from the parameter vector once per step, and the
row lookup is an index into it. Emission substitution reuses the source's draws, so the hidden draw is taken once.

The matrix check runs at compile time with the initial parameter values. `Simulation.update_parameters` runs it again
for every matrix that a changed parameter enters, before it applies the change, and raises `ParameterError`. Compile
time failures raise `ValidationError`, and draws in other law kinds raise `IncompatibleLawsError`.

### Visualization and runtime

Draws are not states, so the trajectory and Rerun do not record them, and a model that wants to record a draw
assigns it to a state. Markov states are scalar states, so `ScalarsVisual` logs them as time series, and categorical
domains keep their labels in the trajectory. A chaos game on a group of points uses `PointsVisual`, which shows the
attractor as the point cloud of each frame. Clocks, frames, live parameters, and co-simulation partitions need no
change, since a stochastic map is still the discrete map form.

### Library models

The models live in `skcomplex.models.mathematics.stochastic_processes` and `skcomplex.models.mathematics.fractals`,
each self-contained:
- `persistent_random_walk(n, persistence)` is a group with a direction `d` in `{-1, 1}` and a position `x`, with
  `d ← where(bernoulli('turn', 1 - q), -d, d)` and `x ← x + where(bernoulli('turn', 1 - q), -d, d)`, which share the
  draw;
- `markov_source(matrix, labels)` and `hidden_markov_source(transition, emission, hidden_labels, labels)` use
  `Transition` and `Emission` on categorical states;
- `chaos_game(maps, weights, n)` is a group of points with `k = categorical('map', weights)` and a piecewise affine
  update chosen by `k`, with the Sierpiński triangle and the Barnsley fern as presets.

## Risks / Trade-offs

- [Numba's normal variates could differ from NumPy's in the last bits] → The spec requires equal trajectories only for
  uniform-based draws, and equal distributions for normal draws, tested with a two-sample test.
- [A draw block per step costs time on small systems] → Draws are vectorised per name, and the deterministic path is
  untouched. The compiled speed scenario bounds the overhead.
- [Users may expect two uses of one name to be independent] → The docstrings and the user guide state the sharing
  rule, and conflicting declarations of one name fail at compile time.
- [Seeded results of existing stochastic models change] → They do not, since only maps with draws consume the stream
  in the `iteration` stepper.
