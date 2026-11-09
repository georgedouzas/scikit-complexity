## 1. Preconditions

- [x] 1.1 Confirm `add-system-package` and `slim-core-elements-and-spaces` are merged on `development`, and that
  `NonnegativeElement`, `BallNeighborhood`, and `skcomplex.core.components.systems` import. Verify with
  `pdm run python -c` importing the three names.

## 2. Periodic guard

- [x] 2.1 Add `periodic_guard(t, period, start=0.0, count=None)` to `core/components/laws/utils/_signals.py` with the
  body of `ImpulseInteraction._get_periodic_guard`, export it from `laws.utils`, and add a docstring example. Verify
  with a test in `tests/core/components/laws/utils/test_signals.py` that its zeros fall at `start + j period` for
  `j < count` and nowhere else.
- [x] 2.2 Add the scenarios "Periodic impulses by a guard" and "Impulses at given times" as tests of
  `CrossingInteraction` in `tests/core/components/interactions/test_crossing.py`, including a guard that reads no
  member over a group. Verify they pass.

## 3. Relaxation into rates

- [x] 3.1 Add physics `RelaxationTimeInteraction` in `physics/common/particles/_relaxation_time.py` on
  `RateInteraction`, with `goal`, `time`, `noise`, the checks, and the hooks `_get_goal` and `_get_time` of
  `RelaxationInteraction`. Export it from the family and move the relaxation tests that fit physics, such as the
  Ornstein-Uhlenbeck variance, to `tests/systems/physics/common/particles/test_relaxation_time.py`. Verify they pass.
- [x] 3.2 Rebase `LangevinThermostatInteraction` and `BhatnagarGrossKrookInteraction` on it, and migrate heat cooling,
  coloured noise, and climate variability. Verify with the tests of `tests/systems/physics` for those modules and the
  LAMMPS thermostat tests.
- [x] 3.3 Add engineering `SpeedAdaptationInteraction` in `engineering/common/traffic/_speed_adaptation.py` with its tests,
  rebase `OptimalVelocityInteraction` and `LangevinDrivingInteraction`, and migrate the continuum traffic systems.
  Verify with `tests/systems/engineering`.
- [x] 3.4 Add chemistry `MassExchangeInteraction` in `chemistry/kinetics/_mass_exchange.py` with its tests, and migrate
  the reactors and the oscillators. Verify with `tests/systems/chemistry/kinetics`.
- [x] 3.5 Add biology `LeakyIntegrationInteraction` in `biology/neuroscience/_leaky_integration.py` with its tests, and
  migrate the populations and networks. Write the trails and adaptivity relaxations as `RateInteraction`. Verify with
  `tests/systems/biology`.
- [x] 3.6 Add economics `PartialAdjustmentInteraction` in `economics/resources/_food.py`, rebase social
  `MotivationInteraction` on `RateInteraction` with its own rate, and write the mathematics driven-flow control as
  `RateInteraction`. Verify with the tests of those modules.
- [x] 3.7 Make the module writer of the system-dynamics translator emit a `RateInteraction` whose rate is
  `(goal - stock) / time` of the goal and time functions, and update the docstrings of `_system_dynamics.py` and
  `_strang_splitting.py`. Verify with `tests/core/engine/translators/test_system_dynamics.py` against PySD.
- [x] 3.8 Delete `interactions/_relaxation.py`, its export, and `tests/core/components/interactions/test_relaxation.py`
  once every test it held lives in a realization test. Verify `grep -rw RelaxationInteraction src tests docs` finds
  nothing, and the core interaction tests pass.

## 4. Diffusion into rates

- [ ] 4.1 Add physics `DiffusivityInteraction` in `physics/common/fields/_diffusivity.py` on `RateInteraction`, with the
  coefficient check and the heat-kernel test moved from `test_diffusion.py` to
  `tests/systems/physics/common/fields/test_diffusivity.py`. Verify it passes.
- [ ] 4.2 Migrate heat, fluid flows, gyres, and gels to it, and write the chemistry, biology, and mathematics
  diffusions as `RateInteraction` with the rate `D laplacian(u)`. Verify with the tests of every migrated module and
  their catalog tag tests.
- [ ] 4.3 Delete `interactions/_diffusion.py`, its export, and its core test. Verify `grep -rw DiffusionInteraction src
  tests docs` finds only the `NeighborInteraction` docstring example class, which is renamed to a neutral name.

## 5. Advection into conservation laws

- [ ] 5.1 Add physics `AdvectionInteraction` in `physics/common/fields/_advection.py` on `ConservationInteraction`,
  with the drift check and the tests moved from `test_advection.py`. Verify they pass, with the test that its
  trajectory equals a `ConservationInteraction` of the flux `a u`.
- [ ] 5.2 Migrate gyres, gels, kinetic flocks, kinetic equations, and fluid flows to it. Add social
  `SelectionDriftInteraction` and biology `SizeGrowthInteraction` in their modules. Verify with the tests of those
  modules and the phase-space tests.
- [ ] 5.3 Rewrite the docstring examples of `_semi_lagrangian.py` and `_strang_splitting.py` with
  `ConservationInteraction` and `RateInteraction`, delete `interactions/_advection.py`, its export, and its core test,
  and move the phase-space advection tests in `tests/core/engine/solvers` to `ConservationInteraction`. Verify with the
  doctests and `tests/core/engine/solvers`.

## 6. Impulses into crossings

- [ ] 6.1 Add engineering `PulseInteraction` in `engineering/common/signals/_pulse.py` on `CrossingInteraction`, with `reset`,
  `times`, `period`, `start`, `count`, and the checks of `ImpulseInteraction`, and move `test_impulse.py` to
  `tests/systems/engineering/common/signals/test_pulse.py`. Verify it passes.
- [ ] 6.2 Migrate circuits, magnetic levitation, and minimum-time control to it, add physics `KickInteraction` in
  `physics/nonlinear_oscillators/_oscillators.py`, and rebase `IrradiationInteraction` on `CrossingInteraction` with
  `periodic_guard`. Verify that every event falls at the time it fell before, with the tests of those modules.
- [ ] 6.3 Delete `interactions/_impulse.py`, its export, and its core test. Verify `grep -rw ImpulseInteraction src
  tests docs` finds nothing.

## 7. Maps of economics

- [ ] 7.1 Add `PriceAdjustmentInteraction` in `economics/markets/_shops.py` on `MapInteraction`, with the modes, the
  bounds, the forms, and the schedules of `SignAdjustmentInteraction`, and move `test_sign_adjustment.py` to
  `tests/systems/economics/markets/test_price_adjustment.py`. Verify the shops tests and the moved tests pass.
- [ ] 7.2 Add `PayoffLearningInteraction` in `economics/common/learning/_payoff_learning.py` on `MapInteraction`, with the
  scores, the rules, `chosen`, and both forms of `ReinforcementInteraction`, and move `test_reinforcement.py` to
  `tests/systems/economics/common/learning/test_payoff_learning.py`. Migrate oligopoly, Brock-Hommes, perishables, El Farol,
  and minority. Verify with `tests/systems/economics`.
- [ ] 7.3 Delete `_sign_adjustment.py` and `_reinforcement.py`, their exports, and their core tests. Verify with grep
  and the core interaction tests.

## 8. Physics-only interactions

- [ ] 8.1 Move `MobilityInteraction` with its private helpers to `physics/common/particles/_stokesian_mobility.py` as
  `StokesianMobilityInteraction`, rebase `SquirmerSlipInteraction` and `BlinkingTrapsInteraction`, and move
  `test_mobility.py` to `tests/systems/physics/common/particles/test_stokesian_mobility.py`. Verify they pass with
  the squirmers and coupled traps tests.
- [ ] 8.2 Move `EinsteinInteraction` to `physics/cosmology/_metric_cosmologies.py` as
  `EinsteinFieldEquationsInteraction`, move `test_einstein.py` to the cosmology tests, and reword the docstring of
  `PseudoRiemannianSpace` so it names no system class. Verify with `tests/systems/physics/cosmology` and the
  curved-spacetime tests.
- [ ] 8.3 Update the squirmers gallery example and the user guide sections on Stokes mobility and curved spacetimes to
  import the physics classes. Verify with the docs doctests.

## 9. Renames

- [ ] 9.1 Rename `LifeCycleInteraction` to `BirthDeathInteraction` in `interactions/_birth_death.py` and
  `test_birth_death.py`, and update every user. Verify `grep -rw LifeCycleInteraction` finds nothing and the agent
  tests pass.
- [ ] 9.2 Rename the `MatchingInteraction` arguments `limit` to `reserve` and `allowance` to `cap`, the rule-spec field
  `clearing` to `value`, and the `QueueSpec` order `'limit'` to `'reserve'`, with the docstrings of the matching specs.
  Update the economics and engineering users. Verify with the matching, labour, double auction, order book, and
  ridesharing tests.
- [ ] 9.3 Document `EncounterInteraction` as the binary kinetic collision operator over class axes, and reword the
  contact text of `RegionScoreSpec` as a boundary cost between labelled regions. Verify with the doctests.

## 10. Documents and inventory

- [ ] 10.1 Rewrite the user guide pages `general_interactions.md`, `phase_space_fields.md`, `stocks_and_flows.md`, and
  `agent_based_systems.md` with the core forms and the field realizations, with no hard-coded counts. Verify with the
  docs build.
- [ ] 10.2 Drop the primitive `hydrodynamic-interactions` from `docs/overview/inventory.toml` and from the squirmers and
  coupled traps entries, set the description of `curved-spacetimes`, and take the priorities the inventory check
  reports. Verify with `docs/inventory_checks.py`.

## 11. Integration

- [ ] 11.1 Run the field-term and import tests of `tests/core/components/test_base.py`, and grep the core for the old
  class names. Verify no match and no failure.
- [ ] 11.2 Run the nox sessions for tests, types, lint, and docs with the thread caps set to 1. Verify all pass.
