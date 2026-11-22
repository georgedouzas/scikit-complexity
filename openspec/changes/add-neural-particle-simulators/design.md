# Design

## Context

A particle group is one `Element` with `size=N` whose vector position `x` has a second derivative set by laws, and a
value named after an axis of its space is its position. Pair forces come from an interaction bound with `among` and
`within=RadiusNeighborhood(r)` that gives its pair energy as `ObjectiveLaw(V(r))`, a symbolic function of the distance,
from which the engine derives the forces `RateLaw(x.diff(t, 2), -dV/dx / m)` of both members.
`LennardJonesFluidSystem` is in `systems/physics/statistical_mechanics/_gases.py`. `VelocityVerletSolver` and
`Yoshida4Solver` integrate positions with second derivatives symplectically, as `solver` settings of a system. At run time the Numba backend finds pairs with cell lists
(`core/engine/backends/_numba_neighbours.py`), private to the compiled kernels. In the analysis layer `_read_snapshots`
(`core/analysis/_base.py`) gives the positions, velocities, and box of a particle or agent group at every frame, and
`neighbors` (`core/analysis/_neighbors.py`) searches pairs with a periodic `scipy.spatial.cKDTree`. `coarse_grain`
(`core/analysis/_coarse_grain.py`) maps agents to grids or members to densities over a state. The surrogate functions
`sparse_identification` and `reservoir` return a `system` built from an `Element` and an interaction that states its
laws, and `reservoir` runs a fitted Python map through `UpdateLaw` with a `CallSpec`. jax and optax are runtime dependencies.

`add-learning-foundation` is a prerequisite. It adds Equinox as a runtime dependency, `learned_function`, whose
learned pair energies lower to NumPy and JAX and differentiate, with weights as an array parameter of the interaction, the private `_OptaxLoop`, the one training loop of the
library with validation and early stopping, `split_runs`, and `forecast_skill`.
`physics/fluids` holds the fluid systems. There is no `physics/fluid_dynamics` subfield and none is created.

## Goals / Non-Goals

**Goals:**

- Learned particle systems that are ordinary systems, prepared and simulated and read by the agent analyses.
- Force matching whose default result is a symbolic pair energy, so it runs on every backend and in LAMMPS.
- The physical metrics of LagrangeBench, distribution distances, as a general analysis.

**Non-Goals:**

- E(3)-equivariant message passing with tensor products (SEGNN), and pretrained foundation potentials, which are left
  to `add-torch-surrogates`.
- Non-separable learned Hamiltonians and learned symplectic maps of SympGNN's full form. The separable form covers
  particles with known masses.
- Exposing the Numba cell lists of the engine to the analysis layer.
- Training on GPUs. Everything runs on the CPU in double precision.

## Decisions

- **Modules.** `core/analysis/_graph_network.py`, `_hamiltonian_network.py`, `_force_matching.py`, and
  `_distribution_distance.py`, each with its frozen result, exported from `skcomplex.core.analysis` and declaring
  `AnalysisTags(DataKind.AGENTS)`. A new package for learned surrogates was considered and rejected, since every analysis
  is flat in `core/analysis`.
- **Radius graphs.** The periodic `cKDTree` search of `_neighbors.py` is hoisted to a private helper
  `_radius_pairs(positions, radius, box)` in `_base.py`, which returns the pairs and their minimum-image displacements.
  `neighbors` and the new functions share it. The rollout systems call the same helper at every step. Reusing the
  Numba cell lists was rejected, since they are private to compiled kernels and the graphs are needed in Python and
  JAX.
- **Graph network.** `graph_network(source, *, radius, group=None, history=5, kinds=None, latent=64, layers=4,
  noise=0.0, validation=0.2, epochs=100, batch=8, learning_rate=1e-3, horizon=0, random_state=None)`. It is GNS: node
  features are the last `history` velocities, normalised, and a learned embedding of the particle kind, edge features
  are the displacement and distance over `radius`, the processor is `layers` message-passing blocks of `eqx.nn.MLP`
  with residual updates and sums over neighbours by `jax.ops.segment_sum`, and the decoder gives accelerations. It is
  trained on the shared `_OptaxLoop` with Adam on the normalised acceleration error with random-walk input noise of size
  `noise`, as in 2309.16342, with whole runs held out by `split_runs` for validation. Static particles, given by `kinds`, are not moved. The result holds `system`, `network`,
  `training_error`, `validation_error`, `rollout_error`, and `forecast`. `rollout_error` is the result of
  `forecast_skill` of the forecast against the held-out frames, so the error by lead time and the valid time come from
  one analysis. Its `system` is a `System` of one `Element(label, values=...)` with
  the positions and the recent velocities of every particle, and one `Interaction` whose `UpdateLaw` calls a vectorised
  `CallSpec` that builds the graph and applies the network, as `reservoir` does. Its time is discrete, since its law is
  an update.
- **Hamiltonian network.** `hamiltonian_network(source, *, cutoff, masses=1.0, group=None, width=64, depth=3,
  validation=0.2, epochs=500, learning_rate=1e-3, random_state=None)`. The potential is `Σ_{i<j} φ(r_ij)` with `φ` an
  `eqx.nn.MLP` of the distance, smoothly cut off at `cutoff`, which is invariant by construction. Training matches one
  velocity Verlet step of the learned forces, `-jax.grad` of the potential, to the next positions and velocities of the
  data, differentiated through the step, on the shared loop. The result holds `system`, `network`, `radii`, `potential`, and the errors.
  Its `system` is a particle group with an interaction bound with `among` and `within=RadiusNeighborhood(cutoff)`
  whose pair energy is `ObjectiveLaw(learned_function(network, r, weights=s.weights))`, the learned pair energy of
  `add-learning-foundation`, with the trained weights as an array parameter of the interaction. The engine turns the
  energy into rate laws on the second derivatives of the positions, so `VelocityVerletSolver` and `Yoshida4Solver`
  split it, and the system runs on the NumPy and JAX backends and differentiates in the weights. The vectorised `CallSpec` that an earlier version of this design planned is not
  needed.
- **Force matching.** `force_matching(source, *, forces, mapping=None, cutoff, basis='gaussian', size=32,
  width=None, kind='linear', masses=None, group=None, random_state=None)`. Positions and forces are mapped by
  `coarse_grain` with `mapping`, the forces with the transpose rule of the multiscale coarse-graining method, summing
  the forces of the particles of a bead. With `kind='linear'` the bead potential is `Σ_k c_k exp(-(r - r_k)² / (2 w²))`
  on `size` centres up to `cutoff`, and the coefficients solve the linear least squares of the mapped forces with
  `scipy.linalg.lstsq`, which is the method of Izvekov and Voth. The `system` is the beads with an
  interaction bound with `within=` whose pair energy `ObjectiveLaw` is that symbolic sum, so it runs on every backend
  and `LammpsSolver` prints it as a Lepton expression. With `kind='network'` the
  potential is an Equinox MLP of the distance plus a repulsive prior `(σ / r)^12`, trained on the shared loop as in
  2310.18278, and the `system` holds an interaction with `within=` whose pair energy is the prior plus the
  `learned_function` of the network, as the Hamiltonian network does. `forces` is an array of frames by
  particles by dimensions.
- **Coarse-grained positions.** `coarse_grain` gains `mapping` and `forces`. With `mapping` it reads the snapshots of a
  particle group, unwraps every group by the minimum image of its first particle, and returns `positions` of the beads,
  `forces` mapped when given, and the box. `CoarseGrainResult` gains the field `forces`, `None` otherwise.
- **Distribution distances.** `distribution_distance(source, reference, *, kind='sinkhorn', group=None, box=None,
  bandwidth=None, epsilon=0.01, iterations=200)`. MMD uses a Gaussian kernel with the median heuristic as the default
  bandwidth. Sinkhorn is a log-domain loop in `jax.numpy` with `logsumexp`, debiased by subtracting the self terms. The
  exact distance solves the assignment with `scipy.optimize.linear_sum_assignment`. Distances use the minimum image
  when a box is given. POT and ott-jax were considered and rejected, since neither is an allowed dependency and the
  three distances are short.
- **System.** `SphDamBreakSystem` goes in `systems/physics/fluids/_sph_dam_break.py`, with `height`, `spacing`,
  `sound_speed` (ten times the largest velocity `sqrt(2 g H)`), `viscosity`, `gravity`, and `rest_density`. Fluid
  particles fill a column `2 H` wide and `H` high in a tank `5.366 H` wide, wall particles form three layers of fixed
  position, velocity zero, and rest density. Fluid and wall particles are members of one element `particles`, whose
  per-member value `wall` marks the walls. One interaction bound with `among='particles'` and
  `within=RadiusNeighborhood(2 h)` states rate laws on the density and on the second derivative of the position of both
  members of every pair, with the Wendland C2 kernel, the Tait pressure, and Monaghan's artificial viscosity, each
  multiplied by `1 - wall` of the member it changes, and a gravity interaction with one role adds `-g` to the vertical
  second derivative of every fluid particle. Gravity and the kernel width are parameters of the interactions. The generalized wall conditions of Adami are replaced by fixed dummy particles, which
  the docstring states.
- **Placement.** The four functions are analyses that read particle and agent trajectories by their tags, step 2, in
  `skcomplex.core.analysis`. Domains: fluids and soft matter in physics, molecular simulation in chemistry, collective
  behaviour in biology, crowds in social systems. `coarse_grain` gains its particle mapping in the core for the same
  domains. `SphDamBreakSystem` is step 3, one system, in `physics/fluids`.

## Risks / Trade-offs

- [The rollout system of the graph network calls Python at every step] → It runs on the NumPy backend only, which its
  tags and the user guide state, and the `CallSpec` is vectorised so each step is one call. The Hamiltonian and force
  matching systems hold learned pair potentials and run on NumPy and JAX.
- [SPH time steps are small] → The defaults use a coarse spacing so the test run stays short, and the gallery example
  uses a finer one.
- [Graph network training time] → Tests train on few frames and few epochs and check only that the rollout beats
  constant velocity. The long training of the gallery example runs once at docs build time.
- [Force matching with stiff bonds] → The dimer scenario maps only the centres, so the bonds cancel in the mapped
  forces, which is the textbook check of the method.
