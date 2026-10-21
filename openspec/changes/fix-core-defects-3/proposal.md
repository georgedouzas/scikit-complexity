# Proposal

## Why

Curating the gallery examples found a third round of core defects:
- angles that jump by a full turn and are colored by a linear palette;
- 2D views that draw physical positions upside down, and no visual for bodies, rods, and paths;
- analyses that recompile or rebuild per call, and a process executor that crashes from scripts;
- models whose defaults do not show what their descriptions promise.

This change fixes every item of that log that reproduces on the current code. Each fix comes with a regression test
that fails before the fix. It does not repeat the fixes of `fix-core-defects-2`.

## What Changes

Angles:
- **BREAKING:** an `ANGLE` state wraps into `(-pi, pi]` instead of `[0, 2 pi)`, so a small swing around zero stays
  near zero. `Domain.ANGLE.encode` wraps the same way.
- Lattice and Monte Carlo steps apply the boundary laws after every step, so the angles of the XY model wrap. Today
  `xy(16)` run until 50 holds angles from -5.7 to 11.2.
- `ImageVisual` colors a real `ANGLE` state with the cyclic palette of complex phases, so equal angles share a color.

Visuals:
- 2D points visuals log positions with the vertical axis up. Rerun's 2D views point their vertical axis down.
- **New:** `BodiesVisual`, registered as `'bodies'`, draws bodies at positions computed from the states of an element,
  rods that join them from a pivot, fading trails of chosen bodies, and static outlines such as the walls of a table.
  `pendulum`, `double_pendulum`, and `circular_billiard` declare it, so their examples stop subclassing
  `RerunObserver`.
- `ScalarsVisual` gives the series of an element distinct colors from a categorical palette. Today Rerun picks colors
  from the entity paths, and the white and black daisies of `daisyworld` come out in nearly the same color.

Analysis and experiments:
- **New:** `replicate(system, size, **values)` returns a system whose single elements become groups of `size`
  members, with per-member values by dotted path, so many initial conditions or parameter values run as one
  vectorised simulation. `orbit_diagram` and `basin_stability` take `vectorized=True` to run their sweeps that way.
- `lyapunov_spectrum` compiles its tangent dynamics once per model structure, with the parameters as an argument.
  Today every call of a one-dimensional map spends 0.8 s compiling Numba functions.
- An experiment with `executor='process'` raises the named `ExperimentError` when its worker processes die, and the
  message names the missing `if __name__ == '__main__':` guard. Today it raises `BrokenProcessPool` after the workers
  print `RuntimeError` tracebacks.

Lattices:
- A relaxation that declares its active sites updates only those sites and their neighbours in each iteration after
  the first. Today `sandpile((64, 64))` costs 3.6 s per 1000 grains in its steady state, and `manna_sandpile` 5.2 s,
  so 60000 grains take more than three minutes.

Models:
- `chemical_langevin_equation` takes the Schlögl rate constants of a volume five times smaller, so its noise switches
  between the two states within 200 time units, as its description says. Today no run switches.
- `hodgepodge` takes `g = 34`, which forms spirals on a 100 × 100 lattice. With `g = 28` the waves break up and the
  ill fraction falls to 0.003.

Gallery tooling:
- `gallery_tools run` prints the path of the thumbnail it writes.

Visuals of agents, networks, and oscillators, from the second log:
- **New:** `TraitsVisual`, registered as `'traits'`, is the default visual of agent groups without a space. It logs
  the histogram of a numeric trait, and the mean and spread of every numeric trait as series. Today the default
  blueprint of `moran_process`, `nk_landscape`, `adaptive_climbing`, `tangled_nature`, `deffuant`,
  `galam_majority`, `naming_game`, `prisoners_dilemma_tournament`, `bit_string_evolution`, `robby_ga`, and
  `bernoulli_process` shows only the constant number of agents, since `AgentsVisual.views` returns no view without a
  space.
- Agents whose position is a state of one coordinate, such as the cars of `charbonneau_traffic`, are drawn as a
  space-time diagram, one line per agent.
- A `Field` on a `NetworkSpace` without a visual takes the `'sites'` visual, a colored graph, as `Element.lattice`
  already does. Today `network_diffusion` and `pagerank` log one series per node.
- A visual's `color_by` without `color_range` widens its range as new values appear, instead of fixing it from the
  first frame. Today agents that all start with the same value, as in `bonabeau_hierarchies` and
  `ring_road_drivers`, keep one color for the whole run.
- **New:** `PhaseRingVisual`, registered as `'phase_ring'`, logs the phases of a group of oscillators against their
  position as points, so chimeras show their coherent and incoherent halves.
- `BodiesVisual` draws bodies of three coordinates in a 3D view with their fading paths, so `stormer_problem` shows
  its 3D path, and `PointsVisual` sizes single particles that share a space by the whole drawing, so the masses of
  `spring_chain` are readable.

Fields:
- The spectral Laplacian gives the Nyquist mode `k = pi` its eigenvalue `-pi^2`. Today a checkerboard mode of
  `complex_ginzburg_landau` grows instead of decaying, while finite differences are right.
- `complex_ginzburg_landau` draws its initial `v` from `random_state`. Today it passes `None`.

Laws, solvers, and analysis:
- **New:** state conditions of `where` and `step` in rate laws compile into hidden events without a reset, so
  integrators restart at the crossing instead of stalling. Today `bathtub(water=20)` stalls LSODA at the empty tub,
  and the stall guard of `fix-core-defects-2` warns and finishes with `RK45`. Hidden events never appear in
  `trajectory.events`.
- `lyapunov_spectrum` differentiates `abs(x)` to `sign(x)`. Today `sprott_jerk` raises `ZeroDivisionError`.

Models:
- `social_impact` flips an opinion when the net pressure to change plus the random field is positive. Today it flips on
  `opinions * impact + field > 0`, which keeps an agent of opinion -1 under pressure and flips it without pressure.
- `gimballed_gyroscope`, `huygens_clocks`, `light_seeking_robots`, `meadow_capacity`, and `stigmergic_predator_prey`
  are checked against their descriptions and fixed in the model where the model is wrong.
- Docstrings of `circadian_clock` (a larger `v_d` lengthens the period, 22.6 h at 0.8 against 29.3 h at 1.6),
  `hypercycles`, `bouchaud_mezard_wealth` (Pareto exponent `1 + 2 J / sigma^2`), and `living_crystals` state what the
  models do.

Remaining items of the second log, each reproduced at the start of its task and dropped with a note in this proposal
when it does not reproduce:
- Hidden lag states and capped series in default blueprints, a blueprint option for components of vector states, and
  per-view grouping of series by magnitude.
- Default point radii from the visual bounds, and brighter default point colors.
- A default solver choice or warning for large diffusive fields, the speed of single scalar SDEs, the named blow-up
  error instead of `LSODA ... Unexpected istate`, and the scheme defects of `rayleigh_benard` and
  `decaying_turbulence`.
- Sanitised entity paths and a separate namespace for user paths in `RerunObserver`, `Trajectory.names`, and a
  vectorised accessor of agent positions over frames.
- Model checks of `restricted_three_body`, `diffusion_in_a_speckle_field`, `legion`, `faraday_disk`, `obesity`,
  `priority_queues`, and `burridge_knopoff`, and the docstring of `kuramoto_sivashinsky`.
- Gallery tooling: whole-script budget, single-channel images, a warning for thumbnails without named series, and
  silenced warnings of `fit_power_law`.

## Dropped items

- Ising Glauber ordering: on a 128 × 128 lattice at `T = 1`, four seeds after 300 sweeps give domain wall densities
  between 0.011 and 0.019, against 0.013 to 0.028 for an independent NumPy checkerboard Glauber reference. Equilibrium
  magnetisations at `T = 2.0, 2.5, 3.0` are 0.915, 0.133, and 0.035. The dynamics are correct, so there is no defect.
- `catalyst_pellet` under LSODA and the gallery runtime of `network_structure_analysis` belong to
  `fix-core-defects-2`.
- The gallery budget covering analysis time is a tooling question outside the core.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `system-modeling`: angles wrap into `(-pi, pi]`, and systems replicate into vectorised ensembles.
- `lattice-dynamics`: lattice steps apply boundary laws, and relaxations update their active neighbourhoods.
- `rerun-visualization`: cyclic colors for angles, 2D positions drawn with the vertical axis up, a bodies visual,
  distinct series colors, agent traits, one-dimensional agents, network fields, widening color ranges, phase rings,
  and 3D paths.
- `spatial-fields`: the Nyquist mode of the spectral Laplacian.
- `hybrid-dynamics`: hidden events at state conditions of rate laws.
- `trajectory-analysis`: fast repeated Lyapunov spectra, and vectorised orbit diagrams and basins.
- `experiments`: a named error for dead worker processes.
- `model-library`: switching in the chemical Langevin equation and spirals in the hodgepodge machine.

## Impact

- `skcomplex.core.base`: `_domains.py` (angle wrapping) and `_errors.py` (`ExperimentError`).
- `skcomplex.core.compile`: `_compiler.py` (angle wrapping) and `_lattices.py` (active neighbourhoods).
- `skcomplex.core.model`: a new `_ensembles.py` with `replicate`.
- `skcomplex.core.solvers`: `_lattices.py` (boundary laws after lattice steps).
- `skcomplex.core.viz`: `_visuals.py` (`ImageVisual`, `PointsVisual`, `ScalarsVisual`, and `BodiesVisual`).
- `skcomplex.core.analysis`: `_lyapunov.py` and `_bifurcation.py`.
- `skcomplex.core.experiments`: `_runner.py`.
- `skcomplex.core.viz`: also `TraitsVisual`, `PhaseRingVisual`, and the default visuals of agents and network fields.
- `skcomplex.core.compile` and `skcomplex.core.solvers`: hidden events of state conditions, and the spectral Laplacian.
- `skcomplex.models`: `pendulum`, `double_pendulum`, `circular_billiard`, `chemical_langevin_equation`,
  `hodgepodge`, `complex_ginzburg_landau`, `social_impact`, the chimera and spring chain models, and the models of the
  model checks, with their gallery examples where the visual or the defaults change. Examples whose comments mark a
  workaround of these defects lose the workaround, and scaffolds that the fixes unblock are curated.
- The changelog entry for the angle range is left to the release, since the parallel apply runs forbid edits of the
  changelog.
- No new dependency. NumPy, Numba, Rerun, and the standard library's `concurrent.futures` cover the fixes.
