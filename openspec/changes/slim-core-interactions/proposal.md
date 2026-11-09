## Why

The core should name one mathematical operation per class. Several core interactions are only a parametrised version
of another core interaction, or name a quantity of one field. `RelaxationInteraction` is a `RateInteraction` with the
rate `(goal - x) / time`. `DiffusionInteraction` and `AdvectionInteraction` are a rate or a conservation law with one
fixed flux. `ImpulseInteraction` is a `CrossingInteraction` whose guard is a clock. `SignAdjustmentInteraction` and
`ReinforcementInteraction` are maps with one fixed function. `MobilityInteraction` and `EinsteinInteraction` belong to
physics alone. These copies make two ways to write one law, and they hide what the core really offers. This change
keeps only the distinct operations in `skcomplex.core.components.interactions` and moves each specific form to the field
that names it.

## What Changes

- **BREAKING** `RelaxationInteraction` leaves the core. Systems use `RateInteraction` with the rate `(goal - x) / time`.
  Fields with their own term get a named realization: physics `RelaxationTimeInteraction`, engineering
  `SpeedAdaptationInteraction`, chemistry `MassExchangeInteraction`, biology `LeakyIntegrationInteraction` in
  neuroscience, economics `PartialAdjustmentInteraction`. Social `MotivationInteraction` is rebased on
  `RateInteraction`. The system-dynamics translator writes a `RateInteraction`.
- **BREAKING** `DiffusionInteraction` leaves the core. Systems use `RateInteraction` with the rate
  `D * laplacian(u)`. Physics gets `DiffusivityInteraction`.
- **BREAKING** `AdvectionInteraction` leaves the core. Systems use `ConservationInteraction` with the flux `a u`.
  Physics gets its own `AdvectionInteraction`, social `SelectionDriftInteraction`, and biology
  `SizeGrowthInteraction`.
- **BREAKING** `ImpulseInteraction` leaves the core. Systems use `CrossingInteraction` with a time guard. A new law
  utility `periodic_guard(t, period, start, count)` gives the guard that crosses zero at periodic times. Engineering
  gets `PulseInteraction`, physics `KickInteraction`, and biology `IrradiationInteraction` is rebased on
  `CrossingInteraction`.
- **BREAKING** `SignAdjustmentInteraction` leaves the core. Economics gets `PriceAdjustmentInteraction` on
  `MapInteraction`.
- **BREAKING** `ReinforcementInteraction` leaves the core. Economics gets `PayoffLearningInteraction` on
  `MapInteraction`.
- **BREAKING** `MobilityInteraction` moves to physics as `StokesianMobilityInteraction`.
- **BREAKING** `EinsteinInteraction` moves to physics as `EinsteinFieldEquationsInteraction`.
  `PseudoRiemannianSpace` and `GeodesicInteraction` stay in the core.
- **BREAKING** `LifeCycleInteraction` is renamed `BirthDeathInteraction`.
- **BREAKING** `MatchingInteraction` takes neutral argument names: `limit` becomes `reserve` and `allowance` becomes
  `cap`. The rule specs `PartnerSpec`, `RandomSearchSpec`, and `StableMatchingSpec` rename `clearing` to `value`, and
  the `QueueSpec` order `'limit'` becomes `'reserve'`.
- `EncounterInteraction` is documented as a binary kinetic collision operator over class axes.
- `HolonomyScoreSpec` and `RegionScoreSpec` keep their names. Both name mathematics, the holonomy of a connection and
  a score of labelled regions.
- Nothing is deleted. The code of every leaving class moves into its realizations, and its tests move with it.
- No alias and no shim module remains, as the project makes breaking changes in 0.x.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `general-mechanisms`: the core lists rates, conservation laws, crossings, maps, and birth-death without the
  parametrised copies, and crossings gain a periodic guard.
- `system-modeling`: core interactions are distinct operations, never a parametrised copy of another core interaction.
- `model-library`: the Langevin thermostat, the kinetic equations, and the fictitious play name their new bases, and
  the merged interactions have named field realizations.
- `phase-space-fields`: phase-space transport uses a conservation law with a drift flux, and BGK relaxation a rate.
- `system-dynamics-import`: goal-seeking stocks become rate interactions.
- `curved-spacetimes`: the field equations of an ansatz are a physics realization.
- `hydrodynamic-interactions`: the mobility interaction is a physics realization.

## Impact

- Code: `src/skcomplex/core/components/interactions/` loses nine modules and renames one. The law utilities gain
  `periodic_guard`. The matching specs in `src/skcomplex/core/specs/` rename one field. The system-dynamics module
  writer and the docstrings of the Strang splitting, semi-Lagrangian, and pseudo-Riemannian modules change. Every
  system module that used a leaving class changes, in physics, engineering, chemistry, biology, economics, social, and
  mathematics.
- Tests: the core tests of the leaving classes move under `tests/systems/`, mirroring their realizations.
- Documents: the user guide pages on general interactions, phase-space fields, stocks and flows, and agent-based
  systems, and the squirmers gallery example.
- Inventory: the primitive `hydrodynamic-interactions` leaves the core vocabulary, and `curved-spacetimes` drops the
  field equations from its description.
- Dependencies: lands after `add-system-package` and `slim-core-elements-and-spaces`. Its examples use
  `skcomplex.core.components.systems`, `NonnegativeElement` with the state `x`, and `BallNeighborhood`.
- Users: code that imports a leaving class imports its realization or writes the core class with the function. Code
  that names `limit`, `allowance`, or `clearing` on a matching uses the new names.
