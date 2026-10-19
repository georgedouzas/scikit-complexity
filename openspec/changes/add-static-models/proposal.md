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

## What Changes

- Add a static form. A system whose laws only define states as closed forms or as solutions of algebraic equations is
  solved again at every frame from its current parameters, with no time derivative.
- Add sweeps. A run can step a parameter through a list of values, and each frame is one value. The trajectory names
  its axis after the parameter, and Rerun shows it on a timeline of that name.
- Add iterative procedures. An element whose state is advanced by one step of an external algorithm, such as a
  generation of a search, a training epoch, or a round of a learning dynamic, runs one step per frame.
- Add random ensembles: quenched random parameters drawn from the simulation's seed, Boltzmann reductions over
  enumerated configurations, and random point patterns with the clusters of their contact graphs.
- Add games: finite two-player games with their equilibria and learning dynamics, and continuous games solved from
  their best-response conditions.
- Add constraint satisfaction: Boolean decision states under clauses and cardinality constraints, solved
  incrementally as constraints arrive, and learning of estimators that train by epochs.
- Add search processes: populations of candidate solutions moved by genetic algorithms, differential evolution,
  particle swarms, evolution strategies, and simulated annealing.
- Add six library models: `lq_cell_survival`, `random_energy_model`, `oligopoly_equilibria`,
  `perceptron_satisfiability`, `evolutionary_optimisers`, and `continuum_percolation`.

## Capabilities

### New Capabilities

- `algebraic-models`: closed-form definitions, algebraic equations, and the static form.
- `random-ensembles`: quenched random parameters, Boltzmann reductions, random point patterns, and their clusters.
- `games`: finite and continuous games, their equilibria, and learning dynamics.
- `constraint-satisfaction`: Boolean constraint problems solved incrementally, and estimators trained by epochs.
- `search-processes`: populations of candidate solutions advanced by optimisation algorithms, one generation per
  frame.

### Modified Capabilities

- `simulation`: sweeps of a parameter as the frame axis, and iterative procedures as a form.
- `rerun-visualization`: the sweep axis as a timeline.
- `model-library`: the six models.

## Impact

- `skcomplex.core.base` gains `Form.STATIC` and `Form.PROCEDURE`.
- `skcomplex.core.model` gains `Define`, `Balance`, `Quenched`, `PointProcess`, `Game`, `Satisfiability`,
  `Learner`, `Search`, and the reductions `log_partition` and `boltzmann_mean`.
- `skcomplex.core.runtime` gains `Sweep`, and `Trajectory` gains its `axis`.
- `skcomplex.core.solvers` gains `Evaluation` and `ProcedureStepper`, and `skcomplex.core.analysis` gains `clusters`
  and `equilibria`.
- New runtime dependencies: `nashpy`, `python-sat`, and `pymoo`. SciPy (`optimize`, `spatial`, `sparse.csgraph`,
  `special`, `stats`) and scikit-learn are already dependencies.
- Depends on the archived `add-experiments` for disorder averages over replicates.
