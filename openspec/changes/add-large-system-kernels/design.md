## Context

- The JAX backend supports systems without a space. Its reductions lower to sums, means, and matrix products, and it
  rejects reductions over neighbourhoods. `add-device-execution` gives it loops on a device for maps, ODEs, SDEs, and
  jumps.
- The Numba backend supports every space. `_numba_neighbours.py` bins members into cell grids, with periodic boxes,
  cones, and nearest neighbours, and `_numba_stencils.py` writes lattice stencils. `LatticeSolver` applies every
  `UpdateLaw` of a lattice once per step: local rules with `neighborhood=`, updates of whole arrays with `arrays=True`,
  drives, and relaxations written with the core operator `relax`. `MetropolisSolver` samples an `ObjectiveLaw` and
  sweeps the sites of a lattice through the private `_MemberMetropolisSolver`, a `MonteCarloSolver`, in the order of
  its `schedule`, `'checkerboard'`, `'random-sequential'`, or `'colored'`.
- Since `redesign-core-api`, laws are values of seven kinds. A transition of agents is an `UpdateLaw` whose value holds
  a draw (`DrawSpec`), a Monte Carlo sampler is an `ObjectiveLaw` run by `MetropolisSolver`, and an agent rule is an
  `UpdateLaw` on a group whose value is a Python function `rule(agent, system)`. The engine lowers them onto its private
  records, such as `_SampleLaw`, `_RuleLaw`, and `_LocalRuleLaw`.
- The neighbourhoods of the core live in `core/components/spaces`: `LatticeNeighborhood`, `RadiusNeighborhood`, with
  its cone, `NearestNeighborhood`, `AdjacencyNeighborhood`, `DelaunayNeighborhood`, and `CofaceNeighborhood`. The
  engine reads them through its private reduction and read records.
- A mesh is a `CellComplexSpace(points=..., cells=...)`, whose fields assemble sparse matrices with scikit-fem on the
  host.
- Observables are evaluated on the host from every recorded frame, and a trajectory holds every state of every frame.
- `add-neural-particle-simulators` uses the Numba neighbour search for its graph networks on the CPU and keeps GPUs out
  of scope.

## Goals / Non-Goals

**Goals:**
- Every large system of the library whose laws have a parallel form runs on the JAX backend: lattices, fields on
  lattices, particles, flocks, networks, explicit fields on meshes, and agent groups whose laws are symbolic.
- Neighbour search on the device in memory linear in the members.
- Runs that copy only their observables to the host.

**Non-Goals:**
- Implicit and exponential field steppers on the device, such as Crank-Nicolson and ETDRK4, whose linear solves need
  sparse factorisations that JAX lacks. They stay on the NumPy and Numba backends.
- Relaxations and extremal updates on the device. Both are serial by nature: a relaxation by `relax` topples one site
  after another, and an extremal update, such as Bak-Sneppen, changes the site where `argmin` falls once per step.
- Delaunay and coface neighbourhoods on the device, which need geometry libraries on the host, and the neighbourhoods
  of `skcomplex.systems.computer_science.common.automata`, which bring their own engine support.
- Splitting one system over devices. `add-multi-device-sharding` does it on these kernels.

## Decisions

### Placement

| Component | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| JAX cell list and neighbour reads | 2, backend lowerings of the neighbourhoods of `core/components/spaces` | new private module `core/engine/backends/_jax_neighbours.py` | physics (particles, granular matter), active matter and collective behaviour (flocks, swarms), ecology (foraging agents), epidemiology (spatial contacts), chemistry (Brownian reactants) |
| Lattice gathers and the `lattice` loop | 2, backend lowerings and a loop the solvers read | `core/engine/backends/_jax.py`, `_jax_loops.py`, and the private `_AcceleratedSpec` of `core/engine/problems/_accelerated_spec.py` | statistical physics (Ising, Potts), epidemiology (contact process), ecology (forest fires), computer science (cellular automata), chemistry (pattern formation) |
| Sparse sums over edges and mesh operators | 2, backend lowerings of network and mesh reductions | `core/engine/backends/_jax.py` | network science, epidemiology (contact networks), neuroscience (connectomes), power systems (grids), engineering (meshes) |
| Lowering of update laws with draws and of lattice sampling | 2, backend lowerings of the law kinds `UpdateLaw` and `ObjectiveLaw` | `core/engine/backends/_jax.py` | social science (opinion and choice), economics (agents choosing), epidemiology (agent states), ecology (behaviour) |
| `record_states` and observables on the device | 2, an option of `System.simulate` and a backend lowering of observables | `core/systems/_base.py`, `core/engine/runtime/_simulate.py`, `_recorder.py`, and `core/engine/backends/_jax.py` | every field with large systems |

No new package and no new concept. Each part lowers a concept the core already has to one more backend.

### Dependencies

None. Checked on PyPI on 2026-11-11, the libraries for neighbour lists:

| Library | Version and date | Licence | Why rejected |
| --- | --- | --- | --- |
| jax-md | 0.2.29, 2026-06-10 | Apache-2.0 | Has JAX cell and neighbour lists, but requires flax, e3nn-jax, ml-collections, pymatgen, and jraph, whose last release is of 2022-08-12, past the 24 month limit |
| vesin | 0.6.2, 2026-09-28 | BSD-3-Clause | Fast, but computes on the host from NumPy arrays in three dimensions, so it cannot run inside a traced loop |
| matscipy | 1.3.1, 2026-10-13 | LGPL | Host neighbour lists for ASE atoms, not traceable |

The cell list is therefore written on `jax.numpy`, about 150 lines, after the sorted-cell method of jax-md (Schoenholz
and Cubuk 2020). It lowers concepts the core already has, and the change records the survey as the constitution asks.
Sparse sums use `jax.ops.segment_sum`, a stable API, rather than `jax.experimental.sparse`, which is still experimental.

### Lattices

- A `LatticeSpace` with its neighbourhood gives a table of neighbour indices of shape `(sites, neighbours)`, built once
  on the host from the geometry the Numba stencils use and put on the device. A read of a neighbour is a gather from
  the table, which covers every lattice geometry and boundary, where a shift by `jnp.roll` would cover only periodic
  square lattices.
- The `lattice` loop runs the update laws of one step in the order they are declared, as `LatticeSolver` does, and
  the sweeps of `MetropolisSolver` on a lattice. The schedules `'synchronous'`, `'checkerboard'`, and `'colored'` of
  local rules, and `'checkerboard'` and `'colored'` of `MetropolisSolver`, run in parallel, one colour after the other.
  `'random-sequential'` and `'random-site'` have no parallel form and raise `BackendError` that names
  `schedule='checkerboard'` and `'colored'`.
- Local rules given as a `TableSpec`, a `TotalisticSpec`, an `OuterTotalisticSpec`, a `LinearRuleSpec`, or a symbolic
  expression lower to gathers and lookup tables. A Python function of a local rule, `f(*neighbours, **kwargs)`, or of a
  whole array, `f(values, **kwargs)` with `arrays=True`, is called on JAX arrays inside the loop, so it runs on the
  device when JAX can trace it. A function that JAX cannot trace raises `BackendError` that names the NumPy and Numba
  backends. This choice is an open question below.
- Draws use one key per step, split per site, so a seeded lattice run repeats on one device. It agrees with the Numba
  run in distribution only.

### Neighbour search

- `_jax_neighbours.py` sorts members by the index of their cell, gives every cell `capacity` slots, and gathers the
  candidates of a member from the cells within its reach. Periodic boxes wrap the cell indices and the distances.
  Cones mask candidates by angle. Nearest neighbours take `jax.lax.top_k` of the negated distances among the
  candidates.
- `capacity` is the largest count of members in a cell at the first frame times 1.5, at least 8. A step that overflows
  a cell sets a flag the loop returns, and the solver raises `BackendError` after the frame, naming `capacity` of
  `JaxBackend`, a new setting whose default `None` keeps the automatic value.
- The list is rebuilt every step inside the loop. Rejected: Verlet lists with a skin rebuilt only when a member moved
  half the skin. They save work when steps are small, but need a second capacity and a check per step, and the cell
  rebuild costs one sort, which is cheap on a GPU. A skin can follow if the benchmarks call for it.

### Networks and meshes

- A `NetworkSpace` gives its edges as two index arrays. A reduction over neighbours becomes
  `segment_sum(values[sources] * weights, targets, num_segments=nodes)`, and a mean divides by the degree.
- The explicit operators of mesh fields, the stiffness and mass products that scikit-fem assembles in CSR form, become
  the same sums over their nonzero entries. Implicit steppers on meshes raise `BackendError` that names the NumPy and
  Numba backends.

### Symbolic agent laws

- An `UpdateLaw` on the members of a group whose value holds draws lowers to draws per member: a categorical draw of
  the next discrete state from its row of probabilities with `jax.random.categorical`, and Bernoulli, uniform, and
  normal draws with the matching functions of `jax.random`. The laws read neighbour reductions through the parts above.
  A recipe such as `ChoiceInteraction` without a `schedule` sets these laws, so it runs on the device.
- An `UpdateLaw` whose value is a Python rule, `rule(agent, system)` on a group or `rule(system)` on the whole system,
  needs sequential orders and Python calls. It raises `BackendError`, whose message names symbolic update laws and a
  recipe without `schedule`. Pairs of agents matched with `pairs` inside a rule fall under the same error.

### Observables on the device

- The JAX backend traces every observable of the problem into one function of `(t, x, p)` that returns their values,
  and the recorder calls it on the device after every frame.
- `system.simulate(..., record_states=True)`. With `False` the trajectory holds the times and the observables and no
  states.
  Reading a state then raises `ValidationError` naming `record_states`. On the NumPy and Numba backends the observables
  are evaluated on the host as today and only the states are dropped.
- The experiments set `record_states=False` when every output is an observable path or a function of observables, so
  large ensembles never copy whole states.

### Probes

The targets are ratios to the Numba backend on the same machine, measured in task 1.1. The GPU targets carry the `gpu`
marker, so the GPU job of `add-device-execution`, which the user decided on 2026-11-11 gates releases, runs them on
its runner before every release.

| Probe | Size | Run | Target on the CPU device | Target on a GPU |
| --- | --- | --- | --- | --- |
| `IsingSystem` | 4096 by 4096 sites | 100 checkerboard sweeps | at most twice the Numba time | at most a tenth of the Numba time |
| `VicsekSystem` | 1000000 particles at density 1 | 100 steps | at most three times | at most a fifth |
| `LennardJonesFluidSystem` | 1000000 particles | 100 velocity Verlet steps | at most three times | at most a fifth |
| `QuenchedMeanFieldSISSystem` | Barabási-Albert network of 1000000 nodes, 3 edges per new node | 100 RK4 frames | at most twice | at most a fifth |
| `GrayScottSystem` | 2048 by 2048 sites | 100 RK4 frames | at most twice | at most a fifth |

Each probe also checks a known result at a smaller size: the Ising magnetisation at temperature 2 within three standard
errors of Onsager's value, the Vicsek polarisation near 1 at low noise, the SIS prevalence above and below the
epidemic threshold, and the agreement of the Gray-Scott field with the Numba run within 1e-10 in double precision.

## Risks / Trade-offs

- [Cells overflow when members cluster, as flocks do] → The overflow flag stops the run with a named error and the
  setting to raise, and the automatic capacity leaves half again of the first frame.
- [The in-house cell list] → It is small, tested against the Numba search on random, clustered, and periodic sets of
  members, and its survey is recorded.
- [Scatter sums on a GPU change order between runs] → The tests compare within 1e-10, and the deterministic XLA flag of
  `add-device-execution` makes them exact.
- [Lattice draws differ from Numba draws] → Lattice probes compare in distribution.
- [Runs without states cannot be replayed frame by frame] → `record_states=False` is opt-in for `system.simulate`, and
  the experiments drop states only when no output needs them.

## Migration Plan

Nothing breaks. Systems that ran on the NumPy and Numba backends run there unchanged, and `record_states=True` keeps
every trajectory as it was.

## Open Questions

- Should the automatic backend choice ever pick the JAX backend for very large systems when a GPU is present? This
  change keeps the answer of `add-device-execution`, no.
- Python functions of local rules and of whole arrays are NumPy functions today. Should the JAX backend call them on
  JAX arrays and raise `BackendError` when JAX cannot trace them, as this design proposes, or should it reject every
  Python function and run only tables and symbolic rules on the device?
