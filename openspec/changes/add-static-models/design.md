# Design

## Context

The parts of the current code this change builds on:
- `StaticTime` in `skcomplex.core.spaces`, whose `_get_frames` gives one frame. `EquationLaw(lhs, rhs, target=)` in
  `skcomplex.core.laws` requires `ContinuousTime` or `StaticTime`, but no solver supports `StaticTime`, so a static
  system does not compile today.
- `PassCompiler` in `skcomplex.engine.compile`, whose report already holds a `law-time` issue for a law whose time
  requirement the system does not meet, and `SupportSpec(time=, space=, laws=)`, the support of a solver.
- `Experiment(source, design=GridDesign(), bounds=, replicates=, observables=)` in `skcomplex.experiments`, which runs
  a model over a grid of parameter values and replicates with independent random streams, and `ExperimentResult`.
- Factories that draw quenched disorder from their `random_state` argument, such as the spin glasses of
  `physics/statistical_mechanics`.
- The reductions `total`, `mean`, `maximum`, and `neighbors` of `skcomplex.core.laws.utils`.
- `ChoiceInteraction(rule='best' | 'logit')`, `ReinforcementInteraction(rule='cumulative')`, `RateInteraction`, and
  `EvolutionInteraction` in `skcomplex.core.interactions`, and `SamplingInteraction` with the `'gaussian'` move.
- Agent rules, `RuleLaw(function)`, which an interaction runs once per step as `function(model)`, and interactions
  that report values per step, such as `LifeCycleInteraction`.
- `ClusterAnalysis` with the kinds `'labels'`, `'sizes'`, `'percolation'`, and `'spanning'` on grids and graphs, and
  `ScalingAnalysis(kind='power')`.

## Goals / Non-Goals

**Goals:**
- Fit solved, sampled, and optimised models into `System`, `Simulation`, `Experiment`, and `Trajectory` without a new
  runtime and without a fake time axis.
- Take every numerical method from a mature library and keep only the adapter in the codebase.
- Name every new core class for its operation, and keep field quantities in the model packages.

**Non-Goals:**
- A sweep axis inside a simulation. A sweep is an `Experiment` with `GridDesign`, and a disorder average is its
  `replicates`.
- Equilibria of finite games by support or vertex enumeration. No model needs them, and the learning dynamics of games
  are compositions of `ChoiceInteraction` and `ReinforcementInteraction`.
- Estimators trained by epochs, which `add-learning-dynamics` covers, so the real-weight perceptron is left to it.
- Message-passing solvers for constraint problems, and multi-objective search.

## Decisions

- **Reference numbers are checked before they become thresholds.** The urban scaling exponent and the fictitious-play
  distance are first reproduced from the cited source; when a run disagrees, the scenario uses the measured value
  with a tolerance from the spread over seeds, and design.md records the source, the measured value, and why they
  differ. Decided on 2026-11-01.

### The static form

`PassCompiler` compiles a system under `StaticTime` to a `ProblemSpec` of the new form `FormKind.STATIC`
(`'static solution'`), whose default solver is `RootSolver`. Its laws are `EquationLaw`s with a `target` and no time
derivative. The equations of one target from every element and interaction add, as they already do for equations in
derivatives. A group target gives one unknown per member, and reductions such as `total` and `neighbors` couple the
members as they do in rates.

At compile time Sage's `solve` is tried on the whole set. A unique explicit solution turns the equations into
expressions printed through the existing printers and evaluated once. Otherwise `RootSolver` calls
`scipy.optimize.root` on the compiled residual and its symbolic Jacobian, from the initial state as the guess. A failure
to converge raises a `SolverError` that names the equations and the largest residual. A system under static time with
no equations solves to its initial state, so observables of quenched states run in one frame.

`RootSolver` declares `SupportSpec(time={StaticTime}, laws={EquationLaw})`. A rate law under static time is already
reported with the code `law-time`. The new code `equation-target` reports an equation under static time with no target,
or with a target that is not a state of the system, with the path of the law.

A sweep of the dose of `lq_cell_survival` is `Experiment(source=lq_cell_survival, design=GridDesign(),
bounds={'dose': values})`, one static run per value.

Alternative considered: a separate definition law and a sweep setting of `Simulation`, as first planned. An
`EquationLaw` with an explicit solution already is a definition, and `Experiment` already is a sweep, so both would be
second ways.

### Library survey

Checked on 2026-10-19 against PyPI, GitHub, and spike installs in Python 3.13 on macOS arm64:

| Library | Licence | Role | Decision |
|---|---|---|---|
| SciPy | BSD-3-Clause | `optimize.root`, `special.logsumexp`, `spatial.cKDTree`, `sparse.csgraph` | already a dependency |
| Sage (passagemath) | GPL | exact `solve` of algebraic equations | already the symbolic layer |
| PySAT 1.9.dev17 | MIT, with permissive bundled solvers | incremental SAT with `add_clause` and cardinality encodings in `pysat.card` | adopt as a runtime dependency |
| OR-Tools CP-SAT 9.15 | Apache-2.0 | constraint programming | reject: pulls `protobuf`, `absl-py`, and `pandas` pins for a feature PySAT covers |
| pymoo 0.6.2 | Apache-2.0 | DE, PSO, ES, CMA-ES with ask and tell | adopt as a runtime dependency |
| DEAP 1.4.4 | LGPL-3.0 | evolutionary toolkit | reject: hand-written loops where pymoo gives the algorithms |
| pygmo 2.20 | MPL-2.0 | many algorithms | reject: no macOS wheels |
| nevergrad 1.0.12 | MIT | ask-and-tell optimisers | reject: pins `bayesian-optimization==1.4.0` |
| nashpy 0.0.43 | MIT | two-player equilibria | reject: no model needs equilibrium enumeration |

PySAT publishes only `.dev` versions: all 177 releases on PyPI are pre-releases, the latest 1.9.dev17 of 2026-10-20,
with wheels for Python 3.12 and 3.13 on Linux and macOS. The manifest already declares `python-sat>=1.8.dev16`, which
admits pre-releases for that package in pip, uv, and PDM, and this change keeps that specifier. Depending on a project
that never cuts a final release is an accepted risk, recorded under Risks. The spike measured a mean critical load of
0.85 at `N = 21` over 20 draws, at 0.11 s per draw, against 0.83 in the limit, and DE reaching 0.0 on the 2-D Rastrigin
function by generation 50 with 40 candidates.

### Log-sum-exp

`log_sum_exp(expression, over=None)` reduces over the members of a group like `total`, as `log sum_j exp(a_j)`, lowered
to `scipy.special.logsumexp` on NumPy and to a max-shifted sum on Numba, so energies of order 10^4 do not overflow. The
Boltzmann averages of the random energy model are written with it in its package, so the core names no Boltzmann
weight.

### Contact clusters of points

`ClusterAnalysis(kind, radius=None)` reads, when `radius` is given, the positions of the agents of a group from a
trajectory or an array of points, builds the contact graph with `cKDTree.query_pairs(2 * radius)`, and labels it with
`connected_components`. The kinds `'sizes'`, `'percolation'`, and `'spanning'` then read that graph, and a cluster spans
along `axis` when it holds points within `radius` of both faces of the space. A random pattern is a group of points
whose positions the factory draws from its `random_state`.

### Satisfiability

`SatisfiabilityInteraction(label, *, target, state, constraints, reveal=1, solver='cadical153')` binds a group whose
`state` is Boolean. A constraint is a frozen spec of `skcomplex.specs`: `ClauseSpec(literals)`, a disjunction, and
`CardinalitySpec(literals, bound, sense='at-least' | 'at-most')`, where a literal is a member index from 1, negative for
the negation. It lowers to a `RuleLaw` of the model that adds the next `reveal` constraints to one PySAT solver each
step, encoding cardinalities with `pysat.card.CardEnc` and a shared `IDPool`, calls `solve`, and sets the state of every
member to the model when one exists. The solver keeps what it learned across steps. The interaction reports
`satisfiable`, 1 or 0, and `revealed`, the number of active constraints, every step. A non-Boolean state raises a
`ValidationError` that the compiler reports with the code `law-domain`.

### Optimisation

`OptimizationInteraction(label, *, target, state, objective, bounds, algorithm=DifferentialEvolutionSpec())` binds a
group of candidates, each holding a vector `state` in the box `bounds`, and an `objective(values)` that takes the array
of candidates and returns their values. The algorithm is a frozen spec of `skcomplex.specs` holding the settings of one
pymoo algorithm: `DifferentialEvolutionSpec(variant, weight, crossover)`, `ParticleSwarmSpec(inertia, cognitive,
social)`, `EvolutionStrategySpec(offspring)`, or `CovarianceAdaptationSpec(sigma)`. The interaction runs each as its own
private method, chosen by the type of the spec, as `MatchingInteraction` runs its rules. It lowers to a `RuleLaw` of the
model that calls `ask`, evaluates the objective, calls `tell`, and writes the population into the members, once per
step, with pymoo seeded from the model's random stream. It reports `best` and `mean` every step. Genetic algorithms
stay with `EvolutionInteraction`, so no genetic algorithm spec is added.

### Annealing

`SamplingInteraction(cooling=None)` takes a function of the time symbol that multiplies the temperature, such as
`lambda t: 0.99**t`. Simulated annealing of a continuous objective is then a group of candidates on a one-dimensional
`LatticeSpace` whose energy reads no neighbours, with the `'gaussian'` move and one `SamplingInteraction` per state.

### Library models

| Model | Field path | Construction |
|---|---|---|
| `lq_cell_survival` | `biology/oncology` | `StaticTime`, `EquationLaw`s for the survival `exp(-alpha D - beta G D^2)`, the Lea-Catcheside factor `G`, and the BED (Eqs. 6.6–6.8) |
| `random_energy_model` | `physics/statistical_mechanics` | `StaticTime`, a group of 2^N members with Gaussian energies drawn from `random_state`, and observables of the log partition function and entropy built on `log_sum_exp` |
| `oligopoly_equilibria` | `economics/markets` | Cournot best responses as `EquationLaw`s (Eqs. 21.5–21.6) and the m-firm price distribution `F(P)` from its indifference condition, and with `dynamics='fictitious-play'` the discretised duopoly as `ChoiceInteraction(rule='best')` on beliefs counted by `ReinforcementInteraction(rule='cumulative')` |
| `von_thunen` | `economics/resources` | `StaticTime`, a `FieldElement` around one market whose crop per cell is the `EquationLaw` of the highest bid rent `Y_i (p_i - c_i) - Y_i f_i d` |
| `perceptron_satisfiability` | `computer_science/neural_networks` | a group of `N` Boolean weights and a `SatisfiabilityInteraction` with one `CardinalitySpec` per random pattern, one pattern revealed per step |
| `evolutionary_optimisers` | `computer_science/evolutionary_computation` | `OptimizationInteraction` on Rastrigin or Rosenbrock with a chosen algorithm, and annealing through `SamplingInteraction` with `cooling` |
| `continuum_percolation` | `physics/statistical_mechanics` | `StaticTime`, a group of disks drawn from `random_state` in a `ContinuousSpace`, read by `ClusterAnalysis(radius=...)` |
| `gravity_urban_scaling` | `social/organizations` | `StaticTime`, a group of residents in a disc of area `A ∝ N^alpha` with the `EquationLaw` of each resident's interaction, a sum of `r^-gamma` over the others |
| `mandelbrot_least_effort` | `social/language` | `StaticTime`, a group of words with costs, probabilities `exp(-lambda c_i) / Z` as `EquationLaw`s, and `lambda` solved from the stationarity of cost over entropy |

The sweeps of the gallery examples are `Experiment`s with `GridDesign`, and the disorder averages are their
`replicates`.

## Risks / Trade-offs

- [The SAT cost grows exponentially near the critical load] → The gallery uses `N` of at most 21.
- [Two new runtime dependencies] → PySAT is small and already declared. pymoo 0.6.2 brings the transitive runtime
  dependencies `autograd`, `alive_progress`, `Deprecated`, `moocore`, `cma`, and `matplotlib`, of which only
  `matplotlib` is already installed, by the `docs` dependency group. deptry learns that pymoo is used.
- [PySAT publishes only pre-releases] → Accepted. The specifier `python-sat>=1.8.dev16` opts into pre-releases for that
  package alone, the wheels cover every supported platform, and the lock file pins the exact `.dev` version, so an
  install is reproducible. A final release, if one appears, satisfies the same specifier.
- [Sage's `solve` may be slow on large groups] → The compiler tries it only on systems below a size limit and falls back
  to the numerical solve, which a scenario checks.
