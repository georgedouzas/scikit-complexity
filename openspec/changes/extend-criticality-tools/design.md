# Design

## Context

See proposal.md for the motivation. Today the only events of a discrete run come from the relaxation operator `relax`
of `core/components/laws/utils/_relaxations.py`. The compiler turns an `UpdateLaw` whose value is a relaxation into a
`CascadeKernel` of `core/engine/problems/_cascade_kernel.py`, which drives the lattice once, relaxes it round after
round until it is stable, and returns a private `_CascadeRecordSpec(label, time, size, duration, area)`. The
`LatticeSolver` of `core/engine/solvers/_lattice.py` appends these records to the events of the trajectory.
`events(kind='excursions')` of `core/analysis/_events.py` reads such records as excursions with sizes and durations but
no shapes, and `crackling_noise` of `systems/physics/criticality/_crackling_noise.py` reads them through it.

`SandpileSystem` and `MannaSandpileSystem` build their cascade from `AvalancheInteraction`, a subclass of
`TopplingInteraction` in `systems/physics/common/avalanches`. `ForestFireSystem` and `NagelSchreckenbergSystem` are
local rules of `UpdateLaw(..., neighborhood=...)` that the `LatticeSolver` runs, and `SelfOrganizedBranchingSystem` is
three symbolic updates of one element that the `IterationSolver` runs, one generation per step. None of the three
records an event, since a fire, a jam or a branching avalanche lasts many steps and no part of the engine follows
activity across steps.

`spectrum(kind='power')` calls `scipy.signal.welch` with `nperseg=min(size, 1024)`. `ExtinctionInteraction` of
`BakSneppenSystem` already takes `neighbors='ring'` or `'graph'`.

## Goals / Non-Goals

**Goals:**

- One general mechanism for events that span several steps, used by the fires, the jams and the branching process.
- Profiles in every cascade record, so the fast-time activity and the avalanche shapes come from the records.
- The toppling matrix, the Green function and Dhar's mean size from mature sparse linear algebra.

**Non-Goals:**

- Toppling matrices of other sandpiles than Bak-Tang-Wiesenfeld, such as Manna or Zhang, whose relaxations are not
  abelian with a fixed matrix.
- A change of the time scale of `SelfOrganizedBranchingSystem`, which keeps one generation per step.

## Decisions

### Placement

- The `activity` of an `UpdateLaw` and the recorder of its events go to the core, by step 2 of the placement rule. The
  engine handles them by the kind of the law, and they are cross-domain: forest fires, traffic jams, epidemics on
  lattices, directed percolation and neuronal avalanches all form bursts of activity connected in space and time.
- The profile of a cascade record and the kind `'activity'` of `events` stay in the core, where the records and
  `events` already live.
- `segment_length` is a setting of `spectrum` in `core/analysis`.
- `abelian_sandpile` is an analysis of the abelian sandpile, one model of the physics criticality subpackage, so it goes
  to a new module `systems/physics/criticality/_abelian_sandpile.py`, beside `_crackling_noise.py`, which is the
  precedent for an analysis of this subpackage. The core keeps no sandpile name. Methods on `SandpileSystem` were
  rejected because a ready-made system exposes `prepare` and `simulate` like an estimator, and the analyses of the
  library are functions with a `kind`.
- Random neighbours of the sandpile are an option of `TopplingInteraction` in `physics/common/avalanches`, which the
  sandpile systems share, and of `SandpileSystem`. Random neighbours of Bak-Sneppen are a value of `neighbors` of
  `ExtinctionInteraction` in its own module.

### Events of activity (#77)

- `UpdateLaw` gains `activity: Expression | None = None`, a symbolic expression of the values of the member the law
  runs for, such as `where(cover == 2, 1, 0)` for a burning site, `where(velocity == 0, 1, 0)` for a stopped car, or
  `where(g < generations, z, 0)` for the branching tree. A value above 0 marks the member as active and adds that value
  to the size. A symbolic expression was chosen over a NumPy function like the `activity` of `relax`, because it reads
  any value of the member, such as `g` beside `z`, and compiles on every backend the laws already reach. The events
  take the label of the law, so a law with an activity needs a label, and `UpdateLaw._check` raises otherwise.
- The compiler lowers the activity like an observable of the member and hands it to the solver with the label. The
  `LatticeSolver` and the `IterationSolver` evaluate it after every step and once on the initial state, and pass the
  active members to a private `_ActivityRecorder` in `core/engine/problems/_activity_recorder.py`, beside
  `_draw_recorder.py`.
- The recorder keeps the open events of the previous step. At every step it joins the active members to each other by
  the nearest neighbour pairs of the space, the von Neumann pairs of a lattice or the edges of a network, and to the
  open events through the same members and their nearest neighbours. It labels the components with
  `scipy.sparse.csgraph.connected_components`, merges open events that a component touches, and closes every open
  event that no component touches as a `_CascadeRecordSpec` whose time is its first step. Events still open when the run
  ends are dropped, as an avalanche cut by the end of a run would be.
- The nearest neighbours of the space were chosen over the neighbourhood of the law. The rule of Nagel-Schreckenberg
  reads five cells ahead, which would join jams that are far apart, while a jam moves back one cell per step, which the
  nearest neighbours follow. A member of no space has no neighbours and connects only to itself at the next step, so
  its events are the runs of positive activity.
- `ForestFireSystem` labels the update of `FireInteraction` `'fire'` with the activity of burning sites, so the
  hexagonal forest and the forest on a graph record fires too.
- `NagelSchreckenbergSystem` gives the activity of stopped cars to its movement law and labels it `'jam'`, so the
  stopped cars are read after the cars have moved. The acceleration law keeps its label.
- `SelfOrganizedBranchingSystem` gives the activity `where(g < generations, z, 0)` to the update of `z` with the label
  `'avalanche'`. The step of the root has activity 1, and the step that reaches the last generation or finds no active
  site has activity 0, so consecutive avalanches never merge. The new state `sigma` takes the update
  `where(ends, where(g >= generations, z, 0), sigma)`, so it holds the dissipation of the last avalanche that ended.
  Rebuilding the process as a relaxation with one avalanche per step was rejected, since it would change the time
  scale that the docstring and the gallery example describe.
- Libraries surveyed for the labelling: `scipy.ndimage.label` labels whole arrays but not a growing stack of frames with
  graph neighbours, and `networkx` components are slower for every step of a large lattice. `scipy.sparse.csgraph`
  labels a sparse graph of the active members of two steps in one call, and SciPy is already a runtime dependency.

### Profiles of cascades (#78)

- `_CascadeRecordSpec` gains `profile: tuple[int, ...]`, the active members of every round, which `CascadeKernel._relax`
  already counts as `updated.sum()`. The events of activity fill the same field with the activity of every step, so
  both kinds of event read alike.
- Every record holds its profile, with no option. The profile holds `duration` integers, so it costs no more than the
  record of the activity series it replaces, and one way to read the activity is simpler than an option that turns it
  on. The request for an option to record the activity per relaxation step is met by the profiles of every record and
  the kind `'activity'`.
- `events(records, kind='excursions')` returns the profiles as the shapes, so `crackling_noise` collapses the shapes of
  sandpile avalanches with no change of its own.
- `events(records, kind='activity')` returns the series `0, profile_1, 0, profile_2, ...`, where every 0 stands for the
  drive round of the step that starts the cascade. Steps whose drive starts no cascade leave no record, so they are not
  in the series, and the docstring says so. The result holds the series as `values` and its round index as `times`,
  and `spectrum(result.values)` gives the power spectrum of Downey.

### Spectrum segments (#78)

- `spectrum(segment_length=1024)` passes `nperseg=min(size, segment_length)` to `scipy.signal.welch`. The name says what
  it holds in plain words, where SciPy's `nperseg` is an abbreviation. It belongs to the settings of the kind `'power'`
  only, so `'structure-factor'` refuses a changed value as `_check_settings` does for other settings.

### Toppling matrix and Green function (#76)

- `abelian_sandpile(system, kind='toppling' | 'green' | 'mean-size')` reads a `SandpileSystem`, prepared or not, and
  returns an `AbelianSandpileResult` with `toppling`, a `scipy.sparse.csr_array`, `green`, a dense array, or
  `mean_size`, a float, by its kind.
- The matrix is `threshold * I - A`, where `A` is the adjacency matrix of the lattice from
  `nx.to_scipy_sparse_array(nx.grid_graph(dim=shape))`, in the row-major order of the sites of the lattice. The open
  edges make it nonsingular, since an edge site loses the grains its missing neighbours would receive.
- `kind='green'` inverts the matrix densely with `scipy.linalg.inv` and refuses lattices above `MAX_GREEN_SITES =
  10_000` sites with an `AnalysisError` that names `kind='mean-size'`, since the inverse of a lattice of side 128 would
  take 2 GB.
- `kind='mean-size'` solves `Delta x = 1` with `scipy.sparse.linalg.spsolve` and returns `mean(x)`, which is
  `(1 / N) Σ_ij G_ij` because the matrix is symmetric.
- A system that is no `SandpileSystem`, or one with `neighbors='random'`, raises an `AnalysisError`. A sandpile with a
  periodic lattice cannot occur, since `SandpileSystem` always has open edges.

### Forest fire boundary (#77)

- `ForestFireSystem(boundary='periodic')` passes the boundary to its `LatticeSpace` and keeps today's behaviour as the
  default. The parameter and its values follow `ReactionDiffusionSystem`, `ParallelHopInteraction` and the other
  lattice systems that take `boundary=`, with the values of `LatticeSpace`. The docstring states the periodic default,
  that a fire at one edge reaches the opposite edge in one step, and that a fixed edge holds no trees.

### Random neighbours (#79)

- The variants are string options of the systems, as the user asked: `neighbors='fixed'` for the lattice, ring or graph
  of today and `neighbors='random'` for neighbours drawn afresh. The spelling `neighbors` follows the parameter that
  `ExtinctionInteraction` already has and the `neighbors` function of the laws, rather than the British spelling of the
  issue. `annealed=True` of `RandomBooleanNetworkSystem` was considered as a precedent for a boolean, but the project
  writes variants as string options.
- `SandpileSystem(neighbors='random', conservation=0.99)` follows the rule R2-MF of Jensen (2023, chapter 12): a site
  at the threshold loses it, and every grain goes to a site drawn uniformly among all sites with the probability
  `conservation`, and leaves the pile otherwise. A grain that is kept or lost by chance was chosen over Jensen's whole
  number `q_max = nu E_max` of receiving sites, so any `conservation` in `(0, 1)` works with the threshold 4. A pile
  with `conservation=1` gains a grain every step and loses none, so it cannot stay stable, and `_check` raises.
  `conservation` is unused with fixed neighbours, and the docstring says so.
- `TopplingInteraction(neighbors='random', conservation=...)` implements it as an array rule beside the scatter of the
  mode `'random'`, drawn from the generator of the run, so the record of the relaxation stays the same.
- `BakSneppenSystem(neighbors='random', k=3)` passes `neighbors='random'` and `k` to `ExtinctionInteraction`, whose rule
  then renews the least fit species and `k - 1` distinct other species drawn uniformly at every update, the model of
  Gros (2015, Eqs. 6.37 to 6.42) whose fitnesses gather above `1 / k`. It is an array update of the fitnesses, since the
  random others are no neighbourhood the symbolic `neighbors` reads. `k` is unused with fixed neighbours.

## Risks / Trade-offs

- [The recorder runs after every step of every law with an activity, which slows long lattice runs] → It works on the
  active members only, which are few in a driven forest or a jam, and the tests time a forest of `100 x 100` for 500
  steps against the run without activity.
- [Merging events by nearest neighbours may join two fires that meet] → Two fires that touch are one cluster of burning
  sites, which is the usual definition of a fire in a forest fire model, and the docstring says so.
- [The profile adds a field to every cascade record] → The record is private, and the only readers are `events` and the
  tests, which this change updates.
- [The scenario on crackling-noise exponents of the sandpile is statistical] → It uses a wide band around the values
  of the literature, and the test fixes the seed.
