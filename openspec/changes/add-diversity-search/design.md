## Context

`sweep` in `core/experiments/_sweep.py` runs a system over rows of parameters from bounds or a design, with replicates,
analyses, outputs, and executors, and returns an `ExperimentResult`. `calibrate` fits parameters to data with optuna,
pyabc, and numpyro. A name without a dot is a model parameter set with `set_params`, and a dotted path names an initial
state or a part parameter, so a search can reuse the same row handling.

`LeniaSystem` in `computer_science/automata/_lenia.py` runs Lenia with ring kernels. Particles with pair forces in
periodic boxes run through `PairInteraction` and the particle helpers of `physics/common/particles`, but `physics` is
another field, so particle life uses only core parts.

## Goals / Non-Goals

**Goals:**
- One experiment that searches parameters for diverse behaviour, with the standard quality-diversity algorithms taken
  from a maintained library, and the goal exploration of the artificial-life papers.
- One analysis that measures open-endedness in the two senses the papers use.
- A probe whose parameter is a matrix, so the search handles shaped parameters.

**Non-Goals:**
- A pretrained vision-language embedding inside the core. The user passes the embedding. Torch and open_clip are not
  added here.
- The curriculum and the gradient inner loop of 2402.10236, which train an initial pattern to reach a goal through a
  differentiable run. `identify` and `TrainingInteraction` of `add-learning-foundation` already fit differentiable
  runs, and goal exploration here perturbs parameters only.
- `sensorimotor_lenia`, Lenia with an obstacle channel. It is a probe of the curriculum, which is out of scope.
- Supervised target search of ASAL. `calibrate` already fits a target, and an embedding-based loss is a user function.

## Decisions

### Placement

| Part | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| `diversity_search`, `DiversitySearchResult` | 2, an experiment the layer runs over any system | `skcomplex.core.experiments` | artificial life, robot morphology and control in engineering, design of materials and molecules in chemistry, exploration of ecological and economic regimes, policy design in social systems |
| `open_endedness`, `OpenEndednessResult` | 2, an analysis that reads trajectories by their tags | `skcomplex.core.analysis` | artificial life, the evolution of technologies and cultures, the training of learning systems, long runs of evolutionary models in biology |
| `ParticleLifeSystem` | 3, one system | `computer_science/automata/_particle_life.py` | computer science |

### The search

- Signature: `diversity_search(system, parameters, *, descriptors, objective=None, method='map-elites', cells=10,
  iterations=100, batch=32, until=None, dt=None, random_state=None, executor='serial')`. `parameters` are bounds as in
  `sweep`. A matrix parameter is given by its dotted entries or as one bound per entry with the shape kept, so particle
  life searches its matrix entry by entry.
- `descriptors` is a mapping from a name to a function of a trajectory with a range, which sets the bounds of the
  archive. `objective` is a function of a trajectory or `None`, which scores every run as 1.
- Runs go through the same private runner as `sweep`, so executors, outputs, and seeds behave the same.
- A parameter may be the weight vector of a `learned_function` or of a `PolicyInteraction` of
  `add-learning-foundation`, given one pair of bounds for every entry. `'cma-me'` searches it as covariance matrix
  adaptation does in neuroevolution, and the archive keeps policies or learned laws that behave differently.
- `'map-elites'` uses `ribs.archives.GridArchive`, `ribs.emitters.GaussianEmitter`, and `ribs.schedulers.Scheduler`.
  `'cma-me'` swaps the emitter for `EvolutionStrategyEmitter` with the improvement ranker. `'novelty'` uses
  `ProximityArchive` with `k` nearest neighbours and a novelty threshold.
- `'goal-exploration'` is written in the module. The first batch draws random rows. Every next row draws a goal
  uniformly in the descriptor bounds, finds the nearest archived run with `sklearn.neighbors.NearestNeighbors`, and
  adds Gaussian noise of a given scale to its parameters inside the bounds. pyribs has no goal-conditioned emitter, so
  this loop is a short private method of the result builder.
- The result holds `parameters`, `descriptors`, and `objectives` of the archive as arrays, `coverage` (filled cells over
  all cells, or for the proximity archive the share of a reference grid within the novelty threshold), `qd_score` (the
  sum of the objectives shifted to be positive), `diversity` (the mean distance to the nearest other archived
  descriptor, equation 4 of 2412.17799), and `history`, these values per iteration.

### Open-endedness

- `'historical-novelty'`: the embedding defaults to the flattened frame of the named variables, scaled to unit norm.
  The novelty of frame `T` is `1 - max_{T' < T} cos(z_T, z_T')`, and the score is its mean over frames after the first,
  the expectation of equation 3 of 2412.17799 with its sign turned so that larger means more open-ended.
- `'observer'`: artefacts are the embedded frames. For every `t` in a set of checkpoints, a clone of `observer`
  (default `sklearn.mixture.GaussianMixture(n_components=4)`) is fitted to the artefacts up to `t`, and `l(t, T)` is
  the mean of `-score_samples` over a window around `T`. The observer may also be a learned density model of
  `add-learning-foundation`, an Equinox module trained on the shared Optax loop by the negative log density of the
  artefacts, whose `score_samples` is its log density. Novelty holds when the Spearman correlation of `l(t, T)` with
  `T` is positive with a p-value below `alpha` for most checkpoints `t`, and learnability when the correlation of
  `l(t, T)` with `t` is negative in the same sense. The result holds the matrix, both correlations per checkpoint, and
  the verdicts.

### Libraries surveyed

- `ribs` (pyribs) 0.12.1, released 2026-11-07, is chosen. It is the reference library of quality diversity, with grid,
  CVT, and proximity archives and the emitters of MAP-Elites, CMA-ME, and CMA-MAE, written on NumPy.
- `qdax` 0.5.0, released 2025-05-27, implements the same algorithms on JAX, but it releases less often and expects the
  evaluation to be a JAX function, which most systems of the library are not.
- `evosax` 0.3.2, released 2026-10-11, gives evolution strategies but no archives. It is not needed, since CMA-ME comes
  with pyribs.
- No maintained library implements intrinsically motivated goal exploration over the parameters of a simulator, so it
  is written in the module.

### Particle life

`ParticleLifeSystem(n_species=4, n_particles=400, matrix=None, core=0.3, cutoff=1.0, friction=0.5, force=1.0,
box=40.0, random_state=None)`. A group of particles holds position, velocity, and species. One `PairInteraction` gives
the force `F(r) = r / core - 1` for `r < core` and `a_ij (1 - |2 r - 1 - core| / (1 - core))` for `core <= r < 1`, in
units of `cutoff`, along the unit vector between the pair, times `force`. A `RateInteraction` gives the friction. A
`matrix` of `None` draws entries uniformly in `[-1, 1]` from `random_state`. Primitives: `particles`, `groups`,
`interactions`, `continuous-spaces`, `draws`.

## Risks / Trade-offs

- [Searches run thousands of simulations] → The executor runs them in parallel, the gallery uses small systems, and
  the guide gives the cost per iteration before the call.
- [Grid archives grow with the power of the number of descriptors] → The docstring advises two or three descriptors,
  and `'novelty'` for more.
- [Observer verdicts depend on the observer] → The verdicts are relative to the observer by definition, and the result
  records the observer and its settings.

## Migration Plan

None. The new functions add to the surface.

## Open Questions

- Should the CVT archive of pyribs be offered as a fourth option for many descriptors? The design starts with grids
  and the proximity archive.
