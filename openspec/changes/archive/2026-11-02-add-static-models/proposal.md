# Proposal

## Why

The inventory records nine models as blocked on this change, or pending it, because they are solved, sampled,
enumerated, or optimised rather than integrated in time:
- The linear-quadratic cell survival curve, a closed form in the dose (Padmanabhan et al. 2021 §6.2, Eqs. 6.6–6.8).
- The random energy model, an exact Boltzmann sum over 2^N quenched random energies with a glass transition (Bertin
  2021 §1.5.3).
- Static oligopolies, equilibria of pricing and quantity games (Forrest 2018 ch. 16, 20, 21).
- Von Thünen land use, the crop of highest bid rent at every distance from a market (2407.02181 §7).
- The binary perceptron, a constraint satisfaction problem whose satisfiability collapses at a critical load (Bertin
  2021 §6.3.3, Ex. 6.4).
- Simulated annealing, genetic algorithms, and differential evolution, search processes over a landscape (Raman 2021
  §6.3.3.3, §6.3.4).
- Continuum percolation, the clusters of random overlapping disks (Satz 2022 ch. 3).
- The gravity model of urban scaling, pair interactions summed over a city and fitted across sizes (2404.02642 §4).
- Mandelbrot's least-effort word frequencies, the minimum of cost over entropy on the probability simplex (2407.02181
  §6).

The current code already provides much of what these need. `StaticTime` exists in `skcomplex.core.spaces`, gives one
frame, and `EquationLaw` accepts it and names its `target`. `Experiment` with `GridDesign` and `replicates` in
`skcomplex.experiments` already sweeps a parameter and averages over independent random streams, so no sweep axis is
needed. Factories already draw quenched disorder from a `random_state` argument, as the spin glasses do.
`ChoiceInteraction`, `ReinforcementInteraction`, and `RateInteraction` already give fictitious play and replicator
dynamics, and `EvolutionInteraction` already gives genetic algorithms. `SamplingInteraction` already gives Metropolis
acceptance with a Gaussian move, `ClusterAnalysis` already labels the components of a graph, and
`ScalingAnalysis(kind='power')` fits power laws across a parameter. What is missing is a solver that solves the
algebraic equations of a static system, the log-sum-exp reduction, contact clusters of points, incremental
satisfiability, population optimisers beyond genetic algorithms, a cooling temperature, and the models.

## What Changes

- Add the static form, `FormKind.STATIC`, and `RootSolver` in `skcomplex.engine.solvers`: a system under `StaticTime`
  whose laws are `EquationLaw`s with a `target` and no time derivative is solved once, exactly when Sage finds a unique
  explicit solution and with `scipy.optimize.root` otherwise, per element and per member of a group.
- Add the compile issue `equation-target` for an equation under static time with no target or with a target that is not
  a state.
- Add the reduction `log_sum_exp` to `skcomplex.core.laws.utils`, beside `total` and `mean`.
- Let `ClusterAnalysis` read the positions of points with a contact `radius`, and label the clusters of their contact
  graph.
- Add `SatisfiabilityInteraction` to `skcomplex.core.interactions`, with the constraint specs `ClauseSpec` and
  `CardinalitySpec` in `skcomplex.specs`: it reveals constraints on the Boolean states of a group step by step and
  decides them incrementally with PySAT.
- Add `OptimizationInteraction` to `skcomplex.core.interactions`, with the algorithm specs
  `DifferentialEvolutionSpec`, `ParticleSwarmSpec`, `EvolutionStrategySpec`, and `CovarianceAdaptationSpec` in
  `skcomplex.specs`: it moves a population of candidates one generation per step through pymoo's ask and tell.
- Add `cooling` to `SamplingInteraction`, a function of time that multiplies the temperature, for simulated annealing.
- Add nine models: `lq_cell_survival`, `random_energy_model`, `oligopoly_equilibria`, `von_thunen`,
  `perceptron_satisfiability`, `evolutionary_optimisers`, `continuum_percolation`, `gravity_urban_scaling`, and
  `mandelbrot_least_effort`, with their field cases in their packages.

## Capabilities

### New Capabilities

- `algebraic-models`: the static form, `RootSolver`, and the equation target check.
- `random-ensembles`: the log-sum-exp reduction and contact clusters of points.
- `constraint-satisfaction`: `SatisfiabilityInteraction` and its constraint specs.
- `search-processes`: `OptimizationInteraction`, its algorithm specs, and annealing by a cooling temperature.

### Modified Capabilities

- `model-library`: the nine models.

## Impact

- `skcomplex.types.FormKind` gains `STATIC`, `skcomplex.engine.solvers` gains `RootSolver`, and
  `skcomplex.engine.compile.PassCompiler` compiles static systems and gains the issue `equation-target`.
- `skcomplex.core.laws.utils` gains `log_sum_exp`, and `skcomplex.analysis.ClusterAnalysis` gains `radius`.
- `skcomplex.core.interactions` gains `SatisfiabilityInteraction` and `OptimizationInteraction`, and
  `SamplingInteraction` gains `cooling`.
- `skcomplex.specs` gains `ClauseSpec`, `CardinalitySpec`, `DifferentialEvolutionSpec`, `ParticleSwarmSpec`,
  `EvolutionStrategySpec`, and `CovarianceAdaptationSpec`.
- A new runtime dependency `pymoo`, which the core uses, with its transitive dependencies. `python-sat`, which
  publishes only pre-releases, is already declared as `python-sat>=1.8.dev16`, and SciPy is already a dependency.
- The package `skcomplex.models.social.language` is new, and `docs/overview/inventory.toml` marks the nine entries
  implemented.
