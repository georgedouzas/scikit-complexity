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
- `rerun-visualization`: cyclic colors for angles, 2D positions drawn with the vertical axis up, a bodies visual, and
  distinct series colors.
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
- `skcomplex.models`: `pendulum`, `double_pendulum`, `circular_billiard`, `chemical_langevin_equation`, and
  `hodgepodge`, with their gallery examples where the visual or the defaults change.
- The changelog entry for the angle range is left to the release, since the parallel apply runs forbid edits of the
  changelog.
- No new dependency. NumPy, Numba, Rerun, and the standard library's `concurrent.futures` cover the fixes.
