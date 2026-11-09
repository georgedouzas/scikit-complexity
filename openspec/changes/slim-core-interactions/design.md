## Context

`skcomplex.core.components.interactions` holds one module per class. An audit of the package found nine classes that
are a parametrised version of another core class or belong to one field. The user decided the rules (see proposal.md).
A core class that only fixes a function of another core class leaves the core. A field with its own term for that form
keeps a named subclass. A pure math form gets no copy, and systems write the core class with its function. Nothing is
deleted: the code of a leaving class moves into its realizations or into the surviving core class.

The audit was checked against the code. Its claims hold, with these corrections.

- `DiffusionInteraction` subclasses `ConservationInteraction`, but its only law is `RateLaw(u, D * laplacian(u))`. A
  `RateInteraction` on a field state emits the same law, and its `conserved` noise already reads the field lattice. So
  the merge into `RateInteraction` needs no new core feature.
- `RelaxationInteraction` has more users than the audit listed. Physics also uses it in heat cooling, coloured noise,
  and climate variability. Engineering uses it twice in the continuum traffic systems. Chemistry uses it in the reactor
  and oscillator systems. Biology uses it in the neuroscience populations and networks, the trails, and the
  physiology adaptivity. The table below lists every user.
- `ImpulseInteraction` takes `times`, but no system uses them. Only its tests do, so `times` moves with the engineering
  `PulseInteraction`, which keeps every argument of the core class.
- `budget`, the audit's name for `allowance` of `MatchingInteraction`, is in the list of field terms that
  `tests/core/components/test_base.py` forbids. This design uses `cap` instead.
- The audit placed realizations in new family packages. The user approved new `common` packages after the draft, so
  the realizations shared by several packages of engineering and economics go into new families of their `common`
  packages.

This change lands after `add-system-package` and `slim-core-elements-and-spaces`. Paths and examples use
`skcomplex.core.components.systems`, `NonnegativeElement(label, visual=None, *, x=0.0)`, and `BallNeighborhood`.

## Goals / Non-Goals

**Goals:**

- One class per mathematical operation in `skcomplex.core.components.interactions`.
- Every trajectory of every catalog system stays the same within the tolerance of its tests.
- Each realization keeps the arguments, checks, hooks, docstring, and tests its core form had.

**Non-Goals:**

- The law classes and law utilities other than the new `periodic_guard`. The sibling change on laws owns them,
  including `collision_integral` and the private engine helpers.
- `_SubsystemSpec` in `skcomplex.core.components.specs`. It is not an interaction, so it stays with the sibling changes.
- `OPTIMIZERS` of `TrainingInteraction`. The engine reads it, so it stays public.
- Elements, spaces, and neighbourhoods. `slim-core-elements-and-spaces` replaces `VonNeumannNeighborhood` in
  `CascadeInteraction` and `SamplingInteraction`, and the docstring examples that use `CompartmentElement` or
  `DensityElement`.

## Decisions

### What leaves, merges, or changes

| From (core) | To (core) | Realizations by field, with paths under `skcomplex/systems/` | Core dependents |
| --- | --- | --- | --- |
| `RelaxationInteraction` | `RateInteraction` with the rate `(goal - x) / time` | physics `RelaxationTimeInteraction` in `physics/common/particles/_relaxation_time.py`, base of `LangevinThermostatInteraction` and `BhatnagarGrossKrookInteraction`, used by heat cooling, coloured noise, and climate variability. engineering `SpeedAdaptationInteraction` in `engineering/common/traffic/_speed_adaptation.py`, base of `OptimalVelocityInteraction` and `LangevinDrivingInteraction`, used by the continuum traffic systems. chemistry `MassExchangeInteraction` in `chemistry/kinetics/_mass_exchange.py`, used by the reactors and the oscillators. biology `LeakyIntegrationInteraction` in `biology/neuroscience/_leaky_integration.py`, used by the populations and networks. economics `PartialAdjustmentInteraction` in `economics/resources/_food.py`. social `MotivationInteraction` in `social/crowds/_pedestrians.py`, rebased on `RateInteraction`. No copy: the trails and the adaptivity of biology and the driven flows of mathematics write `RateInteraction` | `_module_writer.py` template, docstrings of `_system_dynamics.py` and `_strang_splitting.py` |
| `DiffusionInteraction` | `RateInteraction` with the rate `D laplacian(u)` | physics `DiffusivityInteraction` in `physics/common/fields/_diffusivity.py`, used by heat, fluid flows, gyres, and gels. No copy: chemistry reaction-diffusion, biology Min proteins, foraging, trails, cardiac, physiology, angiogenesis, and colonies, and mathematics network processes write `RateInteraction` | none |
| `AdvectionInteraction` | `ConservationInteraction` with the flux `a u` | physics `AdvectionInteraction` in `physics/common/fields/_advection.py`, used by gyres, gels, kinetic flocks, kinetic equations, and fluid flows. social `SelectionDriftInteraction` in `social/cooperation/_multilevel_selection.py`. biology `SizeGrowthInteraction` in `biology/ecology/_forest_size_structure.py` | docstrings of `_semi_lagrangian.py` and `_strang_splitting.py` |
| `ImpulseInteraction` | `CrossingInteraction` with the guard `t - t_k` or `periodic_guard(t, period, start, count)` | engineering `PulseInteraction` in `engineering/common/signals/_pulse.py`, used by circuits, magnetic levitation, and minimum-time control. physics `KickInteraction` in `physics/nonlinear_oscillators/_oscillators.py`. biology `IrradiationInteraction` in `biology/oncology/_radiotherapy.py`, rebased on `CrossingInteraction` | new `periodic_guard` in `core/components/laws/utils/_signals.py` |
| `SignAdjustmentInteraction` | `MapInteraction` | economics `PriceAdjustmentInteraction` in `economics/markets/_shops.py` | none |
| `ReinforcementInteraction` | `MapInteraction` | economics `PayoffLearningInteraction` in `economics/common/learning/_payoff_learning.py`, used by oligopoly, Brock-Hommes, perishables, El Farol, and minority | none |
| `MobilityInteraction` | moves | physics `StokesianMobilityInteraction` in `physics/common/particles/_stokesian_mobility.py`, base of `SquirmerSlipInteraction` and `BlinkingTrapsInteraction` | none |
| `EinsteinInteraction` | moves | physics `EinsteinFieldEquationsInteraction` in `physics/cosmology/_metric_cosmologies.py` | docstring of `spaces/_pseudo_riemannian.py` |
| `LifeCycleInteraction` | renamed `BirthDeathInteraction` in `_birth_death.py` | none, users in economics, mathematics, social, engineering, and biology change their import | none |
| `MatchingInteraction(limit, allowance)` | `MatchingInteraction(reserve, cap)` | economics markets, finance, macroeconomics, resources, and engineering ridesharing rename their arguments | `PartnerSpec`, `RandomSearchSpec`, `StableMatchingSpec` rename `clearing` to `value`, `QueueSpec` order `'limit'` becomes `'reserve'`, docstrings of `BatchMatchingSpec`, `IterativeMatchingSpec`, and `MatchSpec` |
| `EncounterInteraction` | docstring | none | none |
| `HolonomyScoreSpec`, `RegionScoreSpec` | keep | none | none |

### A realization keeps the code of its form

Each realization is a subclass of the surviving core class. It takes the arguments of the leaving class, keeps its
checks as private methods, and keeps its hooks, such as `_get_goal` and `_get_time`, so the subclasses that override
them keep working. `RelaxationTimeInteraction` and `SpeedAdaptationInteraction` hold the same few lines in two fields.
That copy is the rule of the user: realizations are never shared across fields. The core keeps the one general form, a
rate. Alternative considered: keep `RelaxationInteraction` as a shared core convenience. Rejected, since it is a
parametrised `RateInteraction` and gives two ways to write one law.

### Pure math forms get no class

The driven flows of mathematics write the Ornstein-Uhlenbeck control as `RateInteraction` with the rate
`-eta / tau` and its noise. Network processes write diffusion as `RateInteraction` with the rate
`D laplacian(u)`. The audit proposed `ControlRelaxationInteraction`, `GraphLaplacianInteraction`,
`MolecularDiffusionInteraction`, and `SignalDiffusionInteraction`. Mathematics has no field term, and in chemistry and
biology diffusion is the math term, so these get no class. Trails and adaptivity of biology are one-off relaxations
with no field term either. The neuroscience leak is the leaky integrator, a term of the field, so it keeps a class.

### The periodic guard is shared vocabulary

Periodic impulses need a guard that crosses zero once at every `start + j period`, stays away from zero between them,
and stops after `count` impulses. Engineering, physics, and biology all need it, so it is a law utility beside
`pulse_train` in `laws/utils/_signals.py`, with the body of `ImpulseInteraction._get_periodic_guard`. Alternative
considered: a private method in each realization. Rejected, since three fields would copy one function.
`CrossingInteraction` keeps one guard per interaction. `PulseInteraction` keeps the given `times` by returning one
crossing law per time and one for the period, as `ImpulseInteraction` did.

### Placement follows existing packages

A realization of one system module lives in that module. A realization of several modules of one package lives in that
package. A realization of several packages of one field lives in a family package of its `common` package. Physics uses the
existing families `particles` and `fields`. Engineering and economics had no fitting family, so, with the approval of
the user, this change adds `engineering/common/traffic` for `SpeedAdaptationInteraction`,
`engineering/common/signals` for `PulseInteraction`, and `economics/common/learning` for
`PayoffLearningInteraction`, as the `model-library` spec asks for families rather than loose modules. The loose
modules that exist today stay where they are.

### The inventory follows the core

The `primitives` of the inventory name core primitives. The core no longer holds a mobility interaction, so the
primitive `hydrodynamic-interactions` leaves the vocabulary, and the squirmers and the coupled traps drop it. The
inventory check then recomputes their priority, and the entries take the priority it reports. `curved-spacetimes`
keeps pseudo-Riemannian spaces and geodesics in the core, so it stays, with the description "Pseudo-Riemannian spaces
with symbolic metrics, curvature, and geodesics".

## Risks / Trade-offs

- [A crossing whose guard reads no member may fire once per group, not once per member.] → `PulseInteraction`,
  `KickInteraction`, and `IrradiationInteraction` keep the test that an impulse sets every member of a group at once.
  If the crossing gives one event per member, the realization builds one law for the group as `ImpulseInteraction`
  did.
- [Merging a subclass into `RateInteraction` changes its capability tags.] → The tags of `DiffusionInteraction` were
  `RateLaw` alone, the same as a `RateInteraction` without noise. The catalog tag tests of every migrated system run
  in the same task as its migration.
- [Realization names collide with the old core names.] → The physics `AdvectionInteraction` keeps the old name in a
  new module. No alias remains in the core, so an old import fails loudly.
- [Copies of the relaxation code in two fields drift apart.] → Each copy is a few lines with its own tests in
  `tests/systems/`.
- [The sibling change on laws edits `laws/utils/_signals.py` too.] → `periodic_guard` is one new public function, and
  this change lands after the siblings, so it rebases on their version.

## Migration Plan

Users replace imports as the table says. A relaxation becomes
`RateInteraction(..., rate=lambda s: (goal - s.target.x) / time)` or a field realization. A diffusion becomes
`RateInteraction(..., rate=lambda s: D * laplacian(s.target.u))`. An advection becomes
`ConservationInteraction(..., flux=lambda s: drift * s.target.u)`. An impulse becomes
`CrossingInteraction(..., guard=lambda s: periodic_guard(s.t, period, start, count), direction=0)`. Matchings rename
`limit` to `reserve`, `allowance` to `cap`, and `clearing` to `value`. There is no rollback beyond reverting the
commits.
