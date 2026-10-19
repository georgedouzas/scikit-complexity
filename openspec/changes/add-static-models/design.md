# Design

## Context

The parts of the framework that this change builds on:
- Every system compiles to one `Form`, and a stepper per form advances a `Simulation` frame by frame in simulated time.
  The `Trajectory` records `times`, states, parameters, and observables, and `RerunObserver` logs each frame on the
  `step` and `sim_time` timelines.
- Laws are Sage expressions, lowered to NumPy and Numba. `Equation` already solves implicit relations for the highest
  derivatives.
- `schedule` sets a parameter from a function of time at every frame. The archived `add-experiments` runs replicates
  with independent streams, and designs and summaries over them.
- Groups give vectorised members, reductions such as `total` and `mean` reduce over them, and `PointsVisual` draws
  members in a continuous space.
- The Monte Carlo form already compiles Metropolis, Glauber, and heat-bath acceptance in `Sampler`.

The six models (see proposal.md) have no time derivative and no update law. They need a frame axis that is a swept
parameter or the iteration of an algorithm, and adapters to the libraries that solve them.

## Goals / Non-Goals

**Goals:**

- Fit solved, sampled, enumerated, and optimised models into the existing `System`, `Simulation`, and `Trajectory`
  interface, without a parallel runtime, and without a fake time axis.
- Take every numerical method from a mature library, with the core holding only the adapters.
- Keep each construct domain-agnostic. A game is not an oligopoly, a constraint problem is not a perceptron, and a
  point pattern is not a set of disks.

**Non-Goals:**

- Equilibria of games with more than two players. The m-firm pricing model is solved from its indifference
  condition, which is algebraic. Gambit, the one mature solver for n-player games, is noted below.
- Message-passing solvers for constraint problems, such as belief or survey propagation. SAT solvers decide the
  sizes the gallery needs.
- Gradient training of deep networks. `Learner` wraps estimators that learn by `partial_fit`. The Elman network stays
  blocked on its own change.
- Multi-objective search. pymoo supports it, and a later change can expose it.

## Decisions

### How a model without time fits the runtime

Three frame axes cover the six models. Each is the real axis of the model, not time in disguise.

| Axis | Meaning of a frame | `run` | Trajectory | Models |
|---|---|---|---|---|
| sweep of a parameter | one value of the control parameter, solved once | `Simulation(..., sweep=Sweep('lq.dose', values)).run()` | `axis == 'lq.dose'`, `times` hold the doses | `lq_cell_survival`, `random_energy_model`, `oligopoly_equilibria`, `continuum_percolation` |
| iteration of an algorithm | one generation, epoch, or round | `run(until=n)` with `dt=1` | `axis == 'iteration'` | `evolutionary_optimisers`, the learning dynamics of games, the real perceptron |
| growing problem | one more revealed constraint, or more revealed points | a sweep of the revealed count | `axis` is the count parameter | `perceptron_satisfiability`, `continuum_percolation` |

- `Form.STATIC` ('static solution') holds systems whose laws are `Define` and `Balance`, and the elements solved as a
  whole, `Satisfiability` and `PointProcess`. Its stepper `Evaluation` solves the system once per frame from the
  current parameters and sets the state vector to the solution. With no sweep, `run(until=0)` gives one frame.
- `Form.PROCEDURE` ('iterative procedure') holds systems with `Search`, `Learner`, or a `Game` with a learning
  dynamic. `ProcedureStepper` calls one `step` of each such element per frame with the run's random stream.
- `Sweep(parameter, values)` is a run setting, stored unmodified. `Simulation` sets the parameter through the same
  path as `update_parameters`, so domain checks apply, and records the value in place of the time. The catalog's
  defaults may hold a `sweep`, so `run('lq_cell_survival')` sweeps the dose with no arguments. A sweep of a growing
  problem sweeps an integer parameter, such as the number of revealed patterns.
- `Trajectory` gains `axis`, `'time'` by default, so analysis code and the gallery's plots label the axis correctly.
- Disorder averages are ensembles of the existing experiments over seeds. A quenched parameter is drawn once per run,
  so replicates average over draws with no new machinery.

Alternatives considered:
- A separate `Solver` API outside `Simulation`, returning a result object. It would split the library into two
  interfaces, and the models would lose Rerun streaming, observers, ensembles, and the catalog.
- The `schedule` argument with time standing for the swept value. It works mechanically, but labels a dose or a
  temperature as time in the trajectory and in Rerun.

### Library survey

Checked on 2026-10-19 against PyPI's JSON API, GitHub, and the libraries' documentation, with spike installs in a
Python 3.13 environment on macOS arm64:

| Library | Version (date) | Licence | Wheels for Linux and macOS, 3.12–3.13 | Maintenance | Role | Decision |
|---|---|---|---|---|---|---|
| SciPy | 1.18.1 (2026-08-21) | BSD-3-Clause | cp312 and cp313, manylinux and macOS | very active | `optimize.root`, `special.logsumexp`, `stats` distributions, `spatial.cKDTree`, `sparse.csgraph.connected_components` | use, already a dependency |
| scikit-learn | already pinned | BSD-3-Clause | yes | very active | `partial_fit` estimators, such as `Perceptron` | use, already a dependency |
| Sage (passagemath) | already pinned | GPL | yes | active | exact `solve` of algebraic equations | use, already the symbolic layer |
| nashpy | 0.0.43 (2025-11-06) | MIT | pure Python | maintained by its author, releases 2023–2025 | support and vertex enumeration, Lemke-Howson, fictitious play, stochastic fictitious play, replicator dynamics, Moran processes, two players only | adopt as a runtime dependency |
| pygambit (Gambit) | 16.7.0 (2026-07-11) | GPL-2.0-or-later | Windows only. Linux and macOS build the C++ library from source | active | n-player equilibria | reject, no wheels on the supported platforms and not needed by any model |
| PySAT (`python-sat`) | 1.9.dev16 (2026-10-15) | MIT, with permissively licensed bundled solvers (CaDiCaL, Glucose, MiniSat, Lingeling, Maple family) | cp312 and cp313 for manylinux x86_64 and aarch64 and macOS x86_64 and arm64 | very active, last push 2026-10-15, 462 stars | incremental SAT with `add_clause`, assumptions, `enum_models`, and cardinality encodings in `pysat.card` without `pypblib` | adopt as a runtime dependency |
| OR-Tools CP-SAT | 9.15 (2026-01-14) | Apache-2.0 | cp312 and cp313 | very active | constraint programming and linear Boolean constraints | reject, pulls `protobuf`, `absl-py`, and `pandas` pins for a feature PySAT covers |
| pymoo | 0.6.2 (2026-06-28) | Apache-2.0 | cp312 and cp313, manylinux and macOS | active, releases 2024–2026 | GA, DE, PSO, ES, CMA-ES, pattern search, with an ask-and-tell interface | adopt as a runtime dependency |
| DEAP | 1.4.4 (2026-04-17) | LGPL-3.0 | pure Python | active | evolutionary toolkit with hand-written loops | reject, pymoo gives the same algorithms ready-made with ask and tell |
| pygmo | 2.20.0 (2026-10-06) | MPL-2.0 | manylinux only, no macOS wheels | active | many algorithms through `evolve` | reject, no macOS wheels |
| nevergrad | 1.0.12 (2025-04-23) | MIT | pure Python | slowing | ask-and-tell optimisers, no GA or SA by those names | reject, pins `bayesian-optimization==1.4.0` |
| `cma` | 4.5.0 (2026-09-13) | BSD-3-Clause | pure Python | active | CMA-ES | transitive through pymoo |
| Gradient-Free-Optimizers | 1.13.0 (2026-05-15) | MIT | pure Python | active | ask-and-tell SA, GA, DE, PSO | reject, far less used than pymoo, and SA reuses the framework's own acceptance |
| simanneal | 0.5.0 (2019-08-03) | BSD | pure Python | inactive since 2019 | SA | reject |

Under the project's rule that a library the core uses is a runtime dependency whatever its licence, and that only a
library a model alone needs goes into an extra, the three adopted libraries are runtime dependencies. No model needs
an extra.

The spike measured:
- Random energy model, `N = 20`, 10 draws of 2^20 energies, with `logsumexp`: entropy per spin 0.074, 0.108, 0.211,
  0.450, and 0.631 at `T` = 0.3, 0.4, 0.6, 1, and 2, against the theory 0, 0, 0, 0.443, and 0.631, with
  `T_g = 0.6006`. All draws and temperatures took 0.8 s.
- Binary perceptron, patterns revealed one at a time into one CaDiCaL 1.9.5 solver with sequential-counter
  cardinality encodings, 20 draws: mean critical load 0.76, 0.83, 0.85, and 0.85 at `N` = 11, 15, 21, and 25, against
  0.83 in the limit. One draw took 0.11 s at `N = 21` and 0.5 s at `N = 25`.
- Real-weight perceptron, `N = 25`, separability by a HiGHS feasibility LP over 60 draws: 1.0, 0.97, 0.40, 0.017,
  and 0 at loads 1, 1.5, 2, 2.5, and 3, against Cover's critical load 2. scikit-learn's `Perceptron.partial_fit`
  reached zero training error on 12 patterns within 4 epochs in all 20 draws.
- Continuum percolation, disks of radius 1, `cKDTree.query_pairs` and `connected_components`, 50 draws: spanning
  fractions 0, 0.06, 0.52, 0.94, and 1.0 at filling factors 0.9, 1.0, 1.128, 1.25, and 1.4 in a square of side 60.
  All draws took 0.16 s.
- pymoo ask and tell on the 2-D Rastrigin function, 40 candidates, seed 1: DE reached 0.0 by generation 50, PSO 0.0,
  and GA 1.4e-4 by generation 100, at about 1.2 ms per generation.
- nashpy: matching pennies gave the single equilibrium (1/2, 1/2) by support enumeration, and fictitious play reached
  frequencies within 0.02 of 1/2 after 1000 rounds in three seeds. Lemke-Howson enumeration on a 100 by 100
  discretised pricing game did not terminate in 10 minutes, because the game is degenerate. The pricing model is
  therefore solved from its indifference condition, and the discretised game runs fictitious play.

### Definitions and algebraic equations

- `Define(state, expression)` is a law. The compiler orders definitions by their dependencies and rejects a cycle,
  naming the states. They lower through the existing printers to one NumPy or Numba function.
- `Balance(relations, unknowns, guess=None)` is a law. At compile time, Sage's `solve` is tried. A unique explicit
  solution turns the balance into definitions. Otherwise the stepper calls `scipy.optimize.root` with the compiled
  residual and its symbolic Jacobian, starting from the previous frame's solution or `guess`. A failure raises
  `SolveError`, a named error in `skcomplex.core.base`.
- `best_response(payoff, strategy)` returns the relation `diff(payoff, strategy) == 0`, so a continuous game is a
  `Balance` of its players' best responses.

### Random ensembles

- `Quenched(distribution, shape)` declares a parameter drawn from a `scipy.stats` frozen distribution with the run's
  generator, once, in `Simulation._prepare`. Its draw is part of `parameters_`, so it appears in frames and is
  recorded once in the trajectory.
- `log_partition(energy, temperature)` and `boltzmann_mean(value, energy, temperature)` are reductions over a group,
  lowered to `scipy.special.logsumexp` on NumPy and to a max-shifted sum on Numba.
- `PointProcess(space, intensity=None, count=None, maximum=...)` is a group of points in a `ContinuousSpace`. It
  draws the largest pattern of the sweep once, in arrival order, and a frame reveals the first `Poisson(intensity *
  volume)` points of it, drawn by a quenched count, so sweeps are nested.
- `clusters(points, radius, span=axis)` in `skcomplex.core.analysis` builds the contact graph with
  `cKDTree.query_pairs` and labels it with `connected_components`. It returns the labels, the sizes, and whether one
  label touches both faces. It is also an observable, so the largest cluster and the spanning flag stream per frame.

### Games

- `Game(players=('row', 'column'), strategies=..., payoffs=...)` is an element. Payoffs are arrays or Sage
  expressions of the strategy symbols, tabulated on the strategy grids.
- `equilibria(game, method='support' | 'vertex' | 'lemke-howson')` in `skcomplex.core.analysis` returns the mixed
  equilibria through `nashpy.Game`.
- `Game(dynamics='fictitious' | 'stochastic-fictitious' | 'replicator')` makes the element a procedure. Its step
  advances the matching nashpy generator by one item, with NumPy's legacy seed set from the run's stream because
  nashpy draws from the global state. The states are the play counts or the population shares.

### Constraint satisfaction and learning

- `Satisfiability(variables, constraints, reveal=None)` is an element. Its constraints are a sequence of `clause(...)`,
  `at_least(literals, k)`, and `at_most(literals, k)`. Literals are Boolean states or their negation by `~`. `reveal`
  names the integer parameter that sets how many constraints are active, so a sweep of it adds constraints frame by
  frame.
- The adapter keeps one PySAT solver (CaDiCaL 1.9.5 by default) for the run. Each frame adds only the newly revealed
  constraints, encoding cardinalities with `pysat.card.CardEnc` and a shared `IDPool`, and calls `solve`. A frame that
  reveals fewer constraints than the last rebuilds the solver.
- `satisfiable` is an observable, and the states hold the model. `enumerate_solutions(system, limit)` uses
  `enum_models` on a copy of the solver.
- `Learner(estimator, data, target)` is an element that clones a scikit-learn estimator with `partial_fit`. Each step
  is one epoch over the data in a random order from the run's stream. The states are `coef_` and `intercept_`, and
  `training_error` is an observable.

### Search processes

- `Search(variables, bounds, objective, algorithm, population=40)` is an element. The objective is a Sage expression
  of the variables, lowered to a vectorised function over the population.
- For `'ga'`, `'de'`, `'pso'`, `'es'`, and `'cmaes'`, the adapter sets up the pymoo algorithm with the run's seed and
  calls `ask`, evaluates the compiled objective, and calls `tell`, once per frame. The candidates are the members of
  a group, so `PointsVisual` draws them on the landscape, and `best` and `mean` are observables.
- `'annealing'` reuses the Metropolis acceptance that `Sampler` already compiles, applied to a Gaussian proposal per
  candidate, with a temperature parameter that the run's `schedule` cools. No library offers a steppable classical
  simulated annealing that is widely used, and the framework already has the acceptance rule.

### Rerun

- `RerunObserver` sets a timeline named after the sweep's parameter. Rerun timelines hold integers or times, so the
  value is logged as a duration in the parameter's units, and `step` is logged as before.
- No new visual is needed. Curves against the swept parameter are time series on that timeline. Search candidates and
  point patterns are points, perceptron weights and payoff tables are images, and the energy histogram of the random
  energy model is a bar chart through `ScalarsVisual`.

### Library models

| Model | Field path | Construction | Default run | Thumbnail and analysis plot |
|---|---|---|---|---|
| `lq_cell_survival` | `biology/oncology` | `Define` of survival `exp(-alpha D - beta G D^2)`, the Lea-Catcheside factor `G = 2(x - 1 + e^-x)/x^2` with `x = lambda T`, and BED (Eqs. 6.6–6.8) | sweep of the dose over 0–10 Gy | survival on a log scale against dose, and BED of fractionation schemes |
| `random_energy_model` | `physics/statistical_mechanics` | `Quenched` Gaussian energies of variance `N J^2/2` on 2^N members, with `log_partition` observables | sweep of `T` over 0.2–2 | entropy and free energy against `T` with the theory and `T_g`, averaged over draws |
| `oligopoly_equilibria` | `economics/markets` | Cournot best responses as `Balance` (Eqs. 21.5–21.6), the m-firm indifference condition as `Balance` for `F(P)`, and the discretised duopoly as a `Game` with fictitious play | sweep of the incentive weight `lambda` | sales against `lambda`, and the empirical price distribution of fictitious play against `F(P)` |
| `perceptron_satisfiability` | `computer_science/neural_networks` | `Satisfiability` with one `at_least` per pattern, and a real-weight `Learner` variant | sweep of the number of revealed patterns | satisfiable fraction against `p/N` for several `N`, with the critical loads 0.83 and 2 |
| `evolutionary_optimisers` | `computer_science/evolutionary_computation` | `Search` on Rastrigin or Rosenbrock with a chosen algorithm | 100 generations | the population on the landscape, and the best value per generation for each algorithm |
| `continuum_percolation` | `physics/statistical_mechanics` | `PointProcess` of disks and `clusters` | sweep of the filling factor over 0.8–1.5 | the disks coloured by cluster, and the spanning probability against the filling factor for two sizes, crossing near 1.128 |

The inventory gains `random_energy_model` and `perceptron_satisfiability`, and the four existing entries change from
out of scope to pending this change.

## Risks / Trade-offs

- [PySAT publishes only `.dev` versions, which resolvers skip by default] → The manifest pins `python-sat>=1.9.dev16`.
  A specifier that names a pre-release admits pre-releases for that package in pip, uv, and PDM.
- [nashpy draws from NumPy's global random state] → The adapter seeds it from the run's stream before every step,
  and the runs in parallel experiments use processes, not threads.
- [Lemke-Howson stalls on degenerate games] → The documentation says so, `equilibria` takes a step limit and raises
  `SolveError` at it, and the library's pricing model does not depend on it.
- [The SAT cost grows exponentially near the critical load] → The gallery uses `N` of at most 21, 0.11 s per draw in
  the spike.
- [Three new runtime dependencies] → nashpy and PySAT are small, and pymoo brings `cma`, `autograd`, `moocore`, and
  `matplotlib`, which the documentation already installs. deptry learns that they are used.
