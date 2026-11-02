# Proposal

## Why

An audit of `skcomplex` outside the models on 2026-11-02 found the core mostly names operations, but some field
vocabulary and duplicate blocks remain, against the rule that a core class names a mathematical operation that is a
fundamental paradigm usable across domains. Later changes will build on these names, so they change before wave 3.

## What Changes

- **BREAKING** `SamplingInteraction(energy=, temperature=)` becomes `objective=` and `intensity=`, `EnergySpec` becomes
  `ObjectiveSpec`, and `SamplerSpec.temperature` becomes `intensity`. Physics models keep `temperature` in thin subclasses.
- **BREAKING** `ContinuousClearingSpec`, `BatchClearingSpec`, and `IterativeClearingSpec` become `ContinuousMatchingSpec`,
  `BatchMatchingSpec`, and `IterativeMatchingSpec`.
- **BREAKING** `GreedyMatchingSpec` merges into `PartnerSpec`.
- **BREAKING** `DestructionLaw` and `ProliferationLaw` merge into one `BilinearLaw`.
- **BREAKING** `CollisionInteraction` and `CollisionLaw` fold into `EncounterInteraction(cross_section=)`.
- **BREAKING** `SpreadingInteraction` leaves the core for a thin `TransitionInteraction` subclass in the models that use it.
- **BREAKING** `corotational` and `upper_convected` move to `models/physics/common`.
- **BREAKING** Quantum laws, solvers, kernels, lowering, compiler, visuals, and analysis move to
  `models/physics/common/quantum`, compiled through the existing `Simulation(compiler=...)`.
- **BREAKING** `models/earth/{climate,oceanography,seismology}` move to `models/physics/`, and the earth field goes.
- `LammpsFixSpec` moves from the root specs next to `LammpsSolver`; `HeadingSpec` becomes `DirectionSpec`; particle groups
  read headings and velocities from `StateSpec(role=...)`, and `HEADING_NAMES` and `VELOCITY_STATES` go; `_Offer` leaves
  `core/interactions`.
- `KineticSolver` and the other solver names are checked against the same rule.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `general-mechanisms`: neutral vocabulary, one partner spec, one bilinear law, collision integrals on encounters, spreading
  on transitions, and quantum dynamics out of the core.
- `model-library`: the earth field moves to physics, matching spec names, and kinetic equations on encounters.
- `model-inventory`: the coverage page path of climate.
- `phase-space-fields`: collisions through `EncounterInteraction`.

## Impact

- **Code:** `core/interactions`, `core/laws`, `core/specs`, `specs`, `engine/{compile,problems,solvers,backends,viz,runtime}`,
  `analysis`, and every model using these names. Import-linter layers and contracts lose the earth field.
- **API:** breaking and pre-1.0, with no aliases, as the project prefers one way.
- **Trajectories:** unchanged, except rounding where `1/(1/T)` replaces `T`, compared over 20 seeds.
- **Order:** after `add-static-models` and `add-tensor-fields` merge, before wave 3. The open changes naming moved paths are
  updated with it.
