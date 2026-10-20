# Design

## Context

The items come from the issue log of the agents that curated the gallery examples. Each one was reproduced on the
current code with `.venv/bin/python`, after `fix-core-defects-2` was applied:
- A rotor `phi' = -1` from `phi = 0.1` with `domain=Domain.ANGLE` holds values from 0 to 6.27 after one time unit,
  jumping from 0 to 2 pi when it crosses zero. `_wrapping_angles` wraps with `np.mod(x, 2 pi)`.
- `xy(16)` run until 50 holds angles from -5.7 to 11.2. `LatticeStepper.advance`, which `MonteCarlo` inherits, never
  calls the boundary function, so `_wrapping_angles` never runs on lattice and Monte Carlo problems.
- `ImageVisual._log_image` colors real states with `_palette_colors` over the min-max range, and only complex phases
  with the cyclic `_phase_colors`.
- `PointsVisual` logs 2D positions as `Points2D` unchanged, and Rerun's 2D views draw the second coordinate downward.
  The curated `double_pendulum`, `pendulum`, `circular_billiard`, and `projectile` examples subclass `RerunObserver` to
  log `LineStrips2D` rods, paths, and walls, and `Points2D` bobs, negating `y` by hand.
- `ScalarsVisual` logs `rr.Scalars` with no `SeriesLines` colors, so Rerun picks colors from the entity paths.
- `lyapunov_spectrum(logistic_map(r=r), until=2000)` takes 0.8 to 1.2 s per call, after a first call. The profile
  shows all of it in Numba compilation: every call builds new `njit` functions with the parameter values substituted,
  so `_map_spectrum` also recompiles for the new function types.
- A script without a main guard that calls `orbit_diagram(..., executor='process')` makes the spawned workers import
  the script again. They print `RuntimeError` tracebacks, and the parent raises `BrokenProcessPool`.
- `sandpile((64, 64))` takes 3.6 s per 1000 grains in its steady state, and `manna_sandpile((64, 64))` 5.2 s. Every
  relaxation iteration evaluates the toppling rule on all 4096 sites, while an avalanche touches a few.
- The Schlögl chemical Langevin equation with the defaults of Gillespie 2000 has its states near 85 and 563 molecules.
  In 64 Euler-Maruyama paths of 200 time units from the unstable point 250, none switches.
- `hodgepodge` with `n = 100`, `k1 = 2`, `k2 = 3` forms no spirals for `g` in 1, 3, 5, 10, 20, or 28 on a 100 × 100
  lattice within 600 steps. With `g = 34` it forms spirals, and keeps about 18 % of the cells ill. The rule matches
  Gerhardt and Schuster's, checked against an independent NumPy step.

## Goals / Non-Goals

**Goals:**

- Fix each confirmed defect at its source, with a regression test that fails before the fix.
- Give examples the visuals they need, so they draw bodies and paths without subclassing the observer.
- Run sweeps over initial conditions and parameters of small systems as one vectorised simulation.

**Non-Goals:**

- Flipping the vertical axis of images and agents. Image rows and lattice indices count downward, and agents on grids
  share their views with lattice images, so both keep Rerun's orientation.
- A rigid-body or collision engine. `BodiesVisual` only draws what the states say.
- Items that `fix-core-defects-2` covers: `catalyst_pellet` under `LSODA` and the gallery runtime of
  `network_structure_analysis`.

## Decisions

### Angles wrap into one turn centred on zero

`_wrapping_angles` and `Domain.ANGLE.encode` both map a value `a` to `pi - mod(pi - a, 2 pi)`, which lies in
`(-pi, pi]`, and keeps `pi` at `pi`. One helper in `skcomplex.core.base`, `wrap_angle`, holds the formula and both call
it. Phases of complex states already use `(-pi, pi]`, so the two conventions now agree.

Models that draw initial angles from `[0, 2 pi)` keep their draws, since the first boundary step wraps them.

### Lattice steps apply the boundary laws

`LatticeStepper.advance` applies `_bounded` after every whole step, as the ODE and map steppers do after every internal
step. `MonteCarlo` inherits it. A lattice problem without boundary laws or angles has no boundary function, so the cost
is one `None` check per step.

### Cyclic colors for angles

`ImageVisual.log` reads the domain of each state from the frame's problem, and logs an `ANGLE` state with
`_phase_colors`, which wraps any real value. Its space-time image in one dimension keeps the depth image of the values.

### Vertical axis up in 2D

`PointsVisual` negates the second coordinate of 2D positions and trails when it logs them, and `BodiesVisual` does the
same. A view transform was rejected: Rerun applies the transforms of the entities below the origin of a view, not the
transform of its origin, so the points would have to move below a new child path, and every saved blueprint and
example would change its entity paths. Negating the coordinate keeps the paths. Hovering a point shows the negated
value, which the docstrings state.

### A visual for bodies, rods, trails, and outlines

`BodiesVisual(positions, pivot=None, rods=False, trail=TRAIL_LENGTH, traced=None, outline=(), radius=None)`:
- `positions` is a function of the element's states by keyword that returns an array of shape `(k, 2)`, one row per
  body, or a pair of state names for a single body at `(x, y)`.
- `rods=True` draws one line strip from `pivot`, or from the first body without a pivot, through the bodies in order.
- `traced` names the indices of the bodies whose trails are drawn, all bodies by default, and the trail fades: it is
  logged as segments whose colors grow more transparent with age.
- `outline` holds static polylines in the element's coordinates, logged once as `LineStrips2D` with `static=True`.

It logs `<path>/bodies`, `<path>/rods`, `<path>/trail`, and `<path>/outline`, and every state as a time series below
`<path>/<state>`. Its views are a 2D spatial view of the path whose bounds cover the outline, the pivot, and the rods
at full length, and a time series view of the states. It is registered as `'bodies'`, so element types declare it with
`VisualSpec('bodies', {...})`.

`pendulum` and `double_pendulum` declare bodies at `(l sin theta, -l cos theta)` from a pivot at the origin with rods,
and `circular_billiard` declares one body at `(x, y)` with the unit circle as outline. Their examples drop their
observer subclasses. `projectile` keeps its points visual, which now draws the path upward.

The survey for this decision looked for a Rerun component that draws linkages or fading trails. Rerun 0.39 has none,
so the visual composes `Points2D` and `LineStrips2D`, which is the adapter the constitution asks for.

### Distinct series colors

`ScalarsVisual` logs one static `SeriesLines` per state of an element, the first time it logs the element, with colors
from a fixed categorical palette of eight colors indexed by the state's position. The palette is the Okabe-Ito palette
without black, which stays distinct for common color vision deficiencies. `daisyworld` then shows its white and black
daisies in different colors with no change to the model.

### Replicated systems

`replicate(system, size, **values)` in `skcomplex.core.model` returns a system whose elements are groups of `size`
members of the same element types, with the same labels, so a dotted path such as `map.x` names the same state.
Keyword arguments map dotted paths, as `values={'map.r': array}`, to per-member values. Each group keeps its element's
other values. The system's observables are kept. A system with interactions, subsystems, groups, fields, or agents
raises the named `ValidationError`, since a replica of those needs role bindings per member that the system does not
declare.

`orbit_diagram(..., vectorized=True)` replicates the system once over the swept values, runs one `Simulation`, and
reads the member columns of the observable after the transient. `basin_stability(..., vectorized=True)` replicates the
system over the drawn initial conditions, runs once, and calls the classifier with the trajectory of one member,
`Trajectory.member(index)`, which returns a trajectory of the original single system. The vectorised paths reuse the
group machinery of laws and the existing NumPy and Numba kernels, and run without an executor.

Two alternatives were weighed:
- **An executor that batches tasks of an `Experiment`.** It would need the experiment to know which tasks share a
  structure, and it still compiles one problem per task, which is the cost the replicate avoids.
- **A new ensemble form of problems.** It would duplicate the group machinery that already vectorises laws.

### Lyapunov spectra compile once per structure

`lyapunov_spectrum` prints the field and the Jacobian with the parameters as a second argument `p`, so the code depends
only on the model's structure. A module cache maps the printed source to its `njit` function, so a second call with the
same structure reuses the functions, and `_map_spectrum` and `_flow_spectrum` reuse their specialisations.

### Dead worker processes raise a named error

`Experiment.run` catches `BrokenProcessPool` from an executor it owns and raises `ExperimentError` from it. The message
says that worker processes import the main script again on macOS and Windows, that a script must guard its experiment
with `if __name__ == '__main__':`, and that `executor='thread'` or `'serial'` avoids processes. Falling back to serial
runs silently was rejected, since a user who asked for processes would not learn why the run is slow, and the workers'
own tracebacks would still print.

### Relaxations update their active neighbourhoods

`CompiledRelaxation.apply` evaluates the first iteration on every site. When the relaxation declares `activity`, each
later iteration applies its synchronous local rules only at the active sites and at the sites that read them, found
from the reverse of the neighbour table. The documented contract of `activity` becomes: a site that is not active and
reads no active site keeps its value in an iteration. Drives and rules on other schedules keep their full updates. The
size, duration, and area of the record keep their meaning.

### Model defaults

`chemical_langevin_equation` scales the Schlögl constants to a volume five times smaller: `c1 = 1.5e-6`,
`c2 = 2.5e-3`, `c3 = 2e-4`, with `A` and `B` and `c4` kept, and starts from `x = 50`. Its states lie near 17 and 113
molecules, and 98 % of the paths of a NumPy reference switch within 200 time units. Its registered `until` becomes 200.
`hodgepodge` and `HodgepodgeCell` take `g = 34`.

## Risks / Trade-offs

- [Angle trajectories change by multiples of 2 pi] → Tests and examples that read angles in `[0, 2 pi)` move to
  `(-pi, pi]`, and the changelog lists the change as breaking.
- [Negated y in hovered values] → Docstrings of the 2D visuals state it. Positions in trajectories are unchanged.
- [An `activity` function that marks too few sites] → It now skips sites that it would have updated before. The
  contract is documented, and both sandpiles mark every site above threshold, which are the only sites that change on
  their own.
- [The CLE defaults now describe fewer molecules] → The docstring states the volume, and the original constants remain
  arguments.
