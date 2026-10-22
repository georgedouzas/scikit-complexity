# Proposal

## Why

The inventory records six models as out of scope, or leaves them out, because they are not integrated in time. They
are solved, sampled, enumerated, or optimised:
- the linear-quadratic cell survival curve is a closed form in the dose (Padmanabhan et al. 2021 §6.2, Eqs. 6.6–6.8);
- the random energy model is an exact Boltzmann sum over 2^N quenched random energies, with a glass transition
  (Bertin 2021 §1.5.3);
- static oligopolies are Nash equilibria of pricing and quantity games (Forrest 2018 ch. 16, 20, 21);
- the binary perceptron is a constraint satisfaction problem whose satisfiability collapses at a critical load
  (Bertin 2021 §6.3.3, Ex. 6.4);
- simulated annealing, genetic algorithms, and differential evolution are search processes over a landscape
  (Raman 2021 §6.3.3.3, §6.3.4);
- continuum percolation is the cluster structure of random overlapping disks (Satz 2022 ch. 3).

The user ruled that nothing is out of scope. The framework only advances states in time, so none of these fits
today. Each fits once the frame axis can be something other than time: a swept parameter, the iteration of an
algorithm, or a growing number of constraints or points.

**Rebased on `redesign-system-model`.** The redesign already defines static time, `StaticTime()`, validates that a
static system holds only constraint laws, and places algebraic `Equation` and `Objective` in the constraint class. It
also replaces `Sampler` with `Sample`, whose Metropolis acceptance annealing reuses, and moves observables from systems
to simulations and catalog entries. This change no longer adds a static time of its own: it adds the static form and
its evaluation and root solvers, the definition law, sweeps, iterative procedures, random ensembles, games, constraint
problems, learners, and searches. The algebraic-equation law `Balance` is dropped in favour of the redesign's algebraic
`Equation` with declared unknowns. Field realizations leave the core: the Boltzmann reductions are built in
`skcomplex.models.physics.realizations` on a general log-sum-exp reduction, and the learning dynamics of games
(fictitious play, stochastic fictitious play, replicator dynamics) live in `skcomplex.models.economics.realizations`.

## What Changes

- Add a static form. A system with `StaticTime()` whose laws only define states as closed forms or as solutions of
  algebraic equations is solved again at every frame from its current parameters, with no time derivative.
- Add the definition law `Define`, in the constraint class, and let algebraic `Equation` laws declare the states they
  solve for.
- Add sweeps. A simulation can step a parameter through a list of values, and each frame is one value. The trajectory
  names its axis after the parameter, and Rerun shows it on a timeline of that name.
- Add iterative procedures. An element whose state is advanced by one step of an external algorithm, such as a
  generation of a search, a training epoch, or a round of a learning dynamic, runs one step per step of discrete time.
- Add random ensembles: quenched random parameters drawn from the simulation's seed, a log-sum-exp reduction over the
  members of a group, and random point patterns with the clusters of their contact graphs. The Boltzmann reductions
  built on log-sum-exp go to the physics realizations.
- Add games: finite two-player games with their equilibria, and continuous games solved from their best-response
  conditions. Their learning dynamics are economics realizations.
- Add constraint satisfaction: Boolean decision states under clauses and cardinality constraints, solved
  incrementally as constraints arrive, and learning of estimators that train by epochs.
- Add search processes: populations of candidate solutions moved by genetic algorithms, differential evolution,
  particle swarms, evolution strategies, and simulated annealing, minimising an `Objective` law.
- Declare `Requires` for every new primitive and `Supports` for the two new solvers, with the validation issues of
  their misuse.
- Add six library models: `lq_cell_survival`, `random_energy_model`, `oligopoly_equilibria`,
  `perceptron_satisfiability`, `evolutionary_optimisers`, and `continuum_percolation`.

## Capabilities

### New Capabilities

- `algebraic-models`: closed-form definitions, algebraic equations with declared unknowns, and the static form.
- `random-ensembles`: quenched random parameters, log-sum-exp and Boltzmann reductions, random point patterns, and
  their clusters.
- `games`: finite and continuous games, their equilibria, and learning dynamics from the library.
- `constraint-satisfaction`: Boolean constraint problems solved incrementally, and estimators trained by epochs.
- `search-processes`: populations of candidate solutions advanced by optimisation algorithms, one generation per
  step.

### Modified Capabilities

- `simulation`: sweeps of a parameter as the frame axis, and iterative procedures as a form.
- `rerun-visualization`: the sweep axis as a timeline.
- `model-library`: the six models and the realizations they use.

## Impact

- `skcomplex.core.base` gains `Form.STATIC` and `Form.PROCEDURE` as internal forms, the law kinds `DEFINE`
  (constraint class) and `PROCEDURE` (discrete class), and the issue codes of the new primitives.
- `skcomplex.core.laws` gains `Define` and the `unknowns` of algebraic `Equation`, and the core's reductions gain
  `log_sum_exp`.
- `skcomplex.core.components` gains `Quenched`, `PointProcess`, `Game`, `Satisfiability` with `clause`, `at_least`,
  and `at_most`, `Learner`, and `Search`.
- `skcomplex.core.runtime` gains `Sweep`, and `Trajectory` gains its `axis`.
- `skcomplex.core.solvers` gains `Evaluation` and `ProcedureStepper`, and `skcomplex.core.analysis` gains `clusters`
  and `equilibria`.
- `skcomplex.models.physics.realizations` gains `log_partition` and `boltzmann_mean`, and
  `skcomplex.models.economics.realizations` gains the game learning dynamics.
- New runtime dependencies: `nashpy`, `python-sat`, and `pymoo`. SciPy (`optimize`, `spatial`, `sparse.csgraph`,
  `special`, `stats`) and scikit-learn are already dependencies.
- Depends on `redesign-system-model`, and on the archived `add-experiments` for disorder averages over replicates.
