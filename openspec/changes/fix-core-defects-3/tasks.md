# Tasks

Each task writes its regression test first, checks that the test fails on the current code, and then fixes the
defect. Every test follows the naming and public-API rules of the constitution.

## 1. Angles

- [ ] 1.1 Add `wrap_angle` to `skcomplex.core.base`, wrap `ANGLE` states into `(-pi, pi]` in `Domain.ANGLE.encode`
  and `_wrapping_angles`, and verify the small-swing and encoding scenarios, and the existing tests of angle models
- [ ] 1.2 Apply the boundary laws after every step of `LatticeStepper`, and verify the XY scenario and the existing
  lattice and Monte Carlo tests
- [ ] 1.3 Color real `ANGLE` states of `ImageVisual` with the cyclic palette, and verify the angles scenario

## 2. Visuals

- [ ] 2.1 Negate the second coordinate of 2D points and trails in `PointsVisual`, document it, and verify the
  point-above-the-origin scenario and the existing visual tests
- [ ] 2.2 Give the series of `ScalarsVisual` static distinct colors from a categorical palette, and verify the
  daisyworld scenario
- [ ] 2.3 Add `BodiesVisual`, register it as `'bodies'`, export it from `skcomplex.core.viz`, and verify the double
  pendulum and billiard scenarios
- [ ] 2.4 Declare the bodies visual on `pendulum`, `double_pendulum`, and `circular_billiard`, drop the observer
  subclasses of their examples, and check their gallery thumbnails

## 3. Analysis and experiments

- [ ] 3.1 Print the tangent dynamics of `lyapunov_spectrum` with the parameters as an argument and cache the compiled
  functions by source, and verify the sweep scenario and the existing Lyapunov tests
- [ ] 3.2 Add `ExperimentError`, raise it from `Experiment.run` when an owned process pool breaks, and verify the
  script-without-a-guard scenario
- [ ] 3.3 Add `replicate` to `skcomplex.core.model`, and verify the many-initial-conditions and interactions scenarios
- [ ] 3.4 Add `Trajectory.member`, add `vectorized` to `orbit_diagram` and `basin_stability`, and verify the
  vectorised period doubling and basins scenarios

## 4. Lattices

- [ ] 4.1 Update only the active sites and the sites that read them in relaxation iterations after the first,
  document the contract of `activity`, and verify the same-avalanches and steady-state speed scenarios and the
  existing relaxation tests

## 5. Models and tooling

- [ ] 5.1 Scale the defaults of `chemical_langevin_equation` to a volume five times smaller with `until = 200`, and
  verify the switching scenario and its catalog tests
- [ ] 5.2 Set `g = 34` in `hodgepodge` and `HodgepodgeCell`, and verify the waves scenario and its catalog tests
- [ ] 5.3 Print the thumbnail path from `gallery_tools run`

## 6. Gate

- [ ] 6.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
