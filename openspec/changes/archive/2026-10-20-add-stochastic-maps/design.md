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
list. Every draw is written in terms of one standard value: a uniform value `U` in `[0, 1)` for uniform, Bernoulli,
and categorical draws, and a standard normal value `Z` for normal draws:
- `uniform(a, b)` becomes `a + (b - a) U`, and `bernoulli(p)` becomes `where(U < p, 1, 0)`;
- `categorical(w)` becomes the number of cumulative weights `w_0 + ... + w_{k-1}`, for `k` from 1, that `U` times the
  total weight reaches, the inverse cumulative lookup the compiled `gillespie` kernel does already;
- `normal(m, s)` becomes `m + s Z`.

The standard values are stored after the parameters in an extended parameter vector, one per component and name, or
one per member for a group, so the kernels read them like parameters and every backend lowers them without change.
The reductions read by laws move after the draws. `Problem.draws` holds the offset of the draws and which of them are
normal, in the fixed order by component path, draw name, and member index, and `Problem.stochastic` is set when it
exists. The update of a stochastic problem becomes `update(t, x, p, rng)`, which extends `p`, fills the draws, and
evaluates the map. The NumPy backend fills each run of uniform or normal draws with one `rng.random(n)` or
`rng.standard_normal(n)` call, and the Numba loop `iterate_random(t, x, p, L, until, rng, offset, normal)` calls
`rng.random()` or `rng.standard_normal()` once per draw in the same order, so uniform-based draws give equal
trajectories on both backends. The `iteration` stepper passes its generator through. A deterministic map keeps its
present signature and kernel, so existing models and their cached kernels are unchanged. The equations of a stochastic
map show the standard value of each draw as a symbol named after the draw kind, the draw name, and the component.

### Transition and emission laws

`Transition(target, matrix)` and `Emission(target, source, matrix)` are update laws in `skcomplex.core.model`. The
compiler rewrites them before printing:
- a transition becomes `Update(target, categorical(key, row))`, where the row is the matrix row the target's current
  value selects;
- an emission becomes `Update(target, categorical(key, row))`, where the row is the one the source's new value selects,
  obtained by substituting the source's update expression.

The key is derived from the law's target and starts with an underscore, so it never collides with a user's draw name.
The row lookup is a nested condition on the selecting value, whose entries are numbers or expressions of parameters,
so a live parameter change reaches the next step. Emission substitution reuses the source's draws, so the hidden draw
is taken once. `Problem.matrices` holds every matrix as a function of the parameter vector with the parameter indices
it reads.

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

The models are self-contained, each in the package of its field:
- `persistent_random_walk(n_walkers, persistence)` in `mathematics/stochastic_processes` is a group with a direction
  `d` in `{-1, 1}` and a position `x`, with `d ← d (1 - 2 b)` and `x ← x + d (1 - 2 b)` for one shared draw
  `b = bernoulli('turn', 1 - q)`;
- `markov_source(transition)` and `hidden_markov_source(transition, emission)` in `mathematics/stochastic_processes` use
  `Transition` and `Emission` on states over the integers `0` to `n - 1`. The alphabet is fixed by the matrices rather
  than by labels, since a categorical domain is declared on the element type and a type built per call would not
  pickle;
- `random_walk_with_memory(n_walkers, alpha, sigma)` in `mathematics/stochastic_processes` is Bertin's walk
  `x ← x + v`, `v ← alpha v + xi` with a normal draw;
- `chaos_game(preset, maps, weights, n_points)` in `mathematics/fractals` is a group of points with
  `k = categorical('map', weights)` and a piecewise affine update chosen by `k`, with the Sierpiński triangle and the
  Barnsley fern as presets;
- `probabilistic_boolean_network(network, inputs, active)` in `biology/systems_biology` draws the truth table of every
  node with `categorical` and records the drawn index in a state;
- `intrinsic_instability` in `biology/ecology` is Jensen's pair of populations, Eqs. (13.3) and (13.4), with normal
  noise and a Bernoulli mutation;
- `diluted_hopfield` in `computer_science/neural_networks` updates every neuron in parallel to `±1` with a uniform draw
  compared with `1 / (1 + exp(-2 beta h))`, for the local field `h` as a weighted sum over the group.

## Risks / Trade-offs

- [Numba's normal variates could differ from NumPy's in the last bits] → The spec requires equal trajectories only for
  uniform-based draws, and equal distributions for normal draws, tested with a two-sample test.
- [A draw block per step costs time on small systems] → Draws are vectorised per name, and the deterministic path is
  untouched. The compiled speed scenario bounds the overhead.
- [Users may expect two uses of one name to be independent] → The docstrings and the user guide state the sharing
  rule, and conflicting declarations of one name fail at compile time.
- [Seeded results of existing stochastic models change] → They do not, since only maps with draws consume the stream
  in the `iteration` stepper.
