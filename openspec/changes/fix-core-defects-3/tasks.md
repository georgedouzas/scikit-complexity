# Tasks

Each task writes its regression test first, checks that the test fails on the current code, and then fixes the
defect. Every test follows the naming and public-API rules of the constitution. When a fix unblocks a scaffolded
gallery example, the task curates it, and when a curated example carries a commented workaround of the defect, the
task removes the workaround.

## 1. Angles

- [x] 1.1 Add `wrap_angle` to `skcomplex.core.base`, wrap `ANGLE` states into `(-pi, pi]` in `Domain.ANGLE.encode`
  and `_wrapping_angles`, and verify the small-swing and encoding scenarios, and the existing tests of angle models
- [x] 1.2 Apply the boundary laws after every step of `LatticeStepper`, and verify the XY scenario and the existing
  lattice and Monte Carlo tests
- [x] 1.3 Color real `ANGLE` states of `ImageVisual` with the cyclic palette, verify the angles scenario, and remove
  the angle workarounds of the mechanics examples

## 2. Bodies and the vertical axis

- [x] 2.1 Negate the second coordinate of 2D points and trails in `PointsVisual`, document it, and verify the
  point-above-the-origin scenario and the existing visual tests
- [x] 2.2 Add `BodiesVisual`, register it as `'bodies'`, export it from `skcomplex.core.viz`, and verify the double
  pendulum and billiard scenarios
- [x] 2.3 Declare the bodies visual on the pendulum, billiard, and orbit models whose examples subclass
  `RerunObserver`, drop those subclasses, and check their gallery thumbnails

## 3. Agents, networks, and oscillators

- [x] 3.1 Widen color ranges that are not given in the points, agents, and sites visuals, and verify the
  agents-that-start-equal scenario
- [x] 3.2 Add `TraitsVisual`, make it the default of agent groups without a space, verify the Moran process scenario,
  and curate the unblocked scaffolds
- [x] 3.3 Draw agents of one coordinate on a line, verify the traffic scenario, and curate `charbonneau_traffic`
- [x] 3.4 Give fields on networks the sites visual, verify the network diffusion scenario, and curate the unblocked
  scaffolds
- [x] 3.5 Add `PhaseRingVisual` and declare it on the chimera models, draw 3D paths for small groups of points, and
  size the masses of `spring_chain` by the whole drawing, and verify the chimera scenario

## 4. Fields

- [x] 4.1 Give the Nyquist mode of the spectral Laplacian its eigenvalue, seed `v` of `complex_ginzburg_landau` from
  `random_state`, and verify the checkerboard and seeded-fields scenarios

## 5. Model checks

- [x] 5.1 Flip opinions of `social_impact` under net pressure, and verify the no-pressure scenario
- [x] 5.2 Reproduce and fix the reported behaviour of `gimballed_gyroscope`, `huygens_clocks`,
  `light_seeking_robots`, `meadow_capacity`, and `stigmergic_predator_prey`, each with a regression test
- [x] 5.3 Correct the docstrings of `circadian_clock`, `hypercycles`, `bouchaud_mezard_wealth`, and
  `living_crystals`
- [x] 5.4 Scale the defaults of `chemical_langevin_equation` to a volume five times smaller with `until = 200`, and
  verify the switching scenario and its catalog tests
- [x] 5.5 Set `g = 34` in `hodgepodge` and `HodgepodgeCell`, and verify the waves scenario and its catalog tests

## 6. Visual defaults

- [x] 6.1 Give the series of `ScalarsVisual` static distinct colors from a categorical palette, and verify the
  daisyworld scenario

## 7. Laws, analysis, and experiments

- [x] 7.1 Compile state conditions of `where` and `step` in rate laws into hidden events, and verify the bathtub
  scenario
- [x] 7.2 Print the tangent dynamics of `lyapunov_spectrum` with the parameters as an argument, cache the compiled
  functions by source, differentiate `abs` as `sign`, and verify the sweep and Sprott scenarios
- [x] 7.3 Add `ExperimentError`, raise it from `Experiment.run` when an owned process pool breaks, and verify the
  script-without-a-guard scenario
- [x] 7.4 Add `replicate` to `skcomplex.core.model`, and verify the many-initial-conditions and interactions scenarios
- [x] 7.5 Add `Trajectory.member`, add `vectorized` to `orbit_diagram` and `basin_stability`, and verify the
  vectorised period doubling and basins scenarios

## 8. Lattices

- [x] 8.1 Update only the active sites and the sites that read them in relaxation iterations after the first,
  document the contract of `activity`, and verify the same-avalanches and steady-state speed scenarios and the
  existing relaxation tests

## 9. Remaining items of the second log

Each task reproduces its items first, records any that do not reproduce in the proposal's dropped items, and adds a
regression test for each fix.

- [x] 9.1 Hide lag states and cap series in default blueprints, add a blueprint option for components of vector
  states, and record grouping the series of one view by magnitude for a later change
- [x] 9.2 Scale default point radii to the visual bounds and brighten default point colors
- [x] 9.3 Choose or warn about the default solver of large diffusive fields, profile single scalar SDEs, raise the
  named blow-up error for `LSODA` failures of divergent runs, catch states sliding along switches, and check
  `rayleigh_benard` and `decaying_turbulence`
- [ ] 9.4 Sanitise observer entity paths, give user paths their own namespace, add `Trajectory.names`, add a
  vectorised accessor of agent positions over frames, log the auxiliaries and flows of stocks and flows as series, and
  log the waiting times of `priority_queues`
- [ ] 9.5 Check `restricted_three_body`, `diffusion_in_a_speckle_field`, `legion`, `faraday_disk`, `obesity`,
  `priority_queues`, and `burridge_knopoff`, and document the dealiasing of `kuramoto_sivashinsky`
- [ ] 9.6 Print the thumbnail path from `gallery_tools run`, enforce the whole-script budget, accept single-channel
  images, warn when a thumbnail lacks the series its caption names, silence the warnings of `fit_power_law`, include
  the core code in the gallery cache key, write every run's recording to a unique temporary path the scraper is told
  about, and remove the md5 of every example whose capture failed

## 10. Gate

- [ ] 10.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
