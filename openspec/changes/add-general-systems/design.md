## Context

See proposal.md for the motivation. The core offers `BaseSystem`, the composite `CompositeSystem`, and one general
system, `ReactionNetworkSystem`. Its pattern is the model for this change: it takes its data in `__init__`, stores it
unmodified, and builds a `CompositeSystem` from hooks (`_get_species`, `_get_reactions`, `_get_mode`,
`_get_interactions`) that a field system overrides with its own parameters.

The names in this document are those after `slim-core-elements-and-spaces`, `rename-core-systems`,
`slim-core-interactions`, and `slim-core-laws`, which land first. Diffusion is then a `RateInteraction` with the rate
`D laplacian(u)`, `FieldElement` is a `GroupElement` on sites, and `DeclaredElement` declares states and parameters
from data, also as the member of a group through `values={'variables': ...}`.

### The audit

Snapshot of 2026-11-09 on `development`. The catalog under `src/skcomplex/systems` held 734 system classes: physics
199, biology 198, mathematics 87, economics 79, social 66, engineering 49, computer science 42, chemistry 14. Of these,
728 subclassed `BaseSystem` directly and 6 subclassed the reaction network system.

Method. A script in the scratchpad, outside the repository, parsed every system module with `ast`. For each system
class it collected the core names used by the class and, transitively, by the classes and functions of the systems
package it references, its topics, and the parts its `_build_system` creates. Shapes were first matched by these
names, then the build methods of every kept system and of samples of every rejected shape were read by hand. Counts of
rejected shapes are approximate, as a class may match several shapes. The rebase lists below were checked one by one.

| Shape | Fields and counts | Example systems | Verdict |
| --- | --- | --- | --- |
| Rate equations of declared states, no space (ODE, SDE) | biology 15, chemistry 1, economics 13, engineering 16, mathematics 21, physics 30, social 6 | `ChuaSystem`, `WilsonCowanSystem`, `CapitalStockSystem`, `DuffingSystem`, `ThomasSystem` | Keep as `DynamicalSystem` |
| Iterated maps of declared states, no space | biology 5, computer science 1, economics 4, engineering 1, mathematics 16, physics 5 | `HenonMapSystem`, `FibonacciRabbitsSystem`, `MultiplierAcceleratorSystem`, `BeamBeamMapSystem` | Keep as `DynamicalSystem` in discrete time |
| Node dynamics with coupling over a graph | of 36 continuous network systems (biology 8, social 12, mathematics 7, physics 6, economics 2, computer science 1), 6 have this shape: biology 3, physics 2, mathematics 1 | `NanotubeGapJunctionExchangeSystem`, `TippingCascadeNetworkSystem`, `NetworkDiffusionSystem` | Keep as `NetworkDynamicalSystem` |
| Local kinetics with diffusion of fields on a lattice | of 46 field systems with a Laplacian (physics 28, biology 10, chemistry 3, engineering 2, five more), 10 have this shape: biology 5, chemistry 3, physics 2 | `GrayScottSystem`, `FisherKPPSystem`, `BarkleySystem`, `DirectedPercolationLangevinSystem` | Keep as `ReactionDiffusionSystem` |
| Stocks and named flows | 23 build named transfers with no space: economics 9, engineering 7, social 6, physics 1. 16 change their stocks only through flows: economics 7, engineering 3, social 6 | `BathtubSystem`, `BassDiffusionSystem`, `EscalationSystem` | Keep as `StockFlowSystem` |
| Replicator or imitation dynamics from a payoff | the replicator equation of a payoff matrix: biology 2, through `ReplicatorInteraction` of `biology/common/populations`. Physics 1 uses one scalar replicator term inside a climate model. The games of social 11 and economics 6 are agents that imitate on graphs or lattices, or PDEs over group compositions | `ReplicatorDynamicsSystem`, `HypercyclesSystem`, `SpatialPrisonersDilemmaSystem` | Reject. The replicator equation serves one field and already has its field interaction. Imitation is the core `ImitationInteraction` |
| Markov chains | `TransitionInteraction` 9 (biology 5, economics 2, mathematics 2), `JumpInteraction` or jump laws 21 (physics 8, mathematics 6, biology 5, social 2) | `MarkovSourceSystem`, `BirthDeathQueueSystem`, `PolyaUrnSystem` | Reject. A chain is one core interaction with its matrix or its rates, and chains of counts are `ReactionNetworkSystem` in the mode `'jump'`. A class would add no feature |
| Cellular automata | `LocalRuleInteraction` 42: computer science 29, physics 7, chemistry 2, social 2, biology 1, engineering 1 | `GameOfLifeSystem`, `HodgepodgeSystem`, `MajorityRuleSystem`, `NagelSchreckenbergSystem` | Reject. It is a group on sites and one core interaction, and `group-cellular-automata` forbids a class that duplicates it |
| Hamiltonian or Lagrangian systems | `HamiltonianInteraction` or `EulerLagrangeInteraction` 9: physics 8, engineering 1 | `HenonHeilesSystem`, `TwoBodyOrbitSystem`, `DcMotorSystem` | Reject. One core interaction each, and nearly one field |
| Agent-based systems with rules and a schedule | 51: economics 17, biology 14, social 11, mathematics 6, computer science 2, engineering 1 | `SugarscapeSystem`, `WolfSheepSystem`, `ElFarolSystem` | Reject. No shared structure beyond `CompositeSystem` with groups and phases. The rules differ per model |
| Particles in a continuous space | 55: physics 46, biology 4, engineering 3, mathematics 1, social 1 | `VicsekSystem`, `BoidsSystem`, `ActiveBrownianParticlesSystem` | Reject. A group on a continuous space with pair, neighbour, or movement interactions. The forces differ, so a class would not shorten the signature |
| Linear programmes of flux balance | biology 23, economics 0 on inspection | `FluxBalanceSystem`, `DynamicFBASystem` | Reject. One field, held by `biology/common/metabolism` |
| Boolean networks | biology 8 | `RandomBooleanNetworkSystem`, `YeastCellCycleSystem` | Reject. One field |
| Monte Carlo sampling of spins or labels | `SamplingInteraction` 12: physics 8, biology 3, computer science 1 | `IsingSystem`, `PottsSystem`, `CellularPottsSystem` | Reject. One core interaction |
| Network growth and rewiring in discrete time | 26: mathematics 14, biology 6, social 4, computer science 1, physics 1 | `BarabasiAlbertSystem`, `DuplicationDeletionSystem` | Reject. `TopologyInteraction` with rules that differ per model |
| Species and reactions | biology 50, chemistry 8 | `SIRSystem`, `HahnfeldtSystem` | Already `ReactionNetworkSystem`. More rebases onto it are outside this change |

Systems of a kept shape whose element class other systems or users compose stay on their base. The audit found 41
such systems among the declared-state shapes, such as `LorenzSystem`, `RosslerSystem`, `VanDerPolSystem`, and
`MackeyGlassSystem`, whose elements live in `mathematics/common/flows` or are composed by coupled systems.

## Goals / Non-Goals

**Goals:**

- Four general systems that each remove the element class and the threefold parameter list of their field systems.
- Rebases that keep every public name, parameter, state name, tag, and trajectory of a system.
- Examples of each general system from two fields, in its docstring and in the core components guide.

**Non-Goals:**

- No class for a rejected shape.
- No rebase of a system whose element class is shared.
- No change to the system-dynamics translator. Whether it writes a `StockFlowSystem` subclass is an open question.
- No further rebase onto `ReactionNetworkSystem`.

## Decisions

### One `DynamicalSystem` for flows and maps

`DynamicalSystem` is the textbook object: a state space and an evolution rule, in continuous or discrete time. The
time is a parameter, as the form of the reactions is a parameter of `ReactionNetworkSystem`, because the base
interfaces forbid a class chosen by mathematical form. Noise amplitudes are allowed only in continuous time.

```python
class DynamicalSystem(BaseSystem[CompositeSystem]):
    def __init__(
        self,
        states: Mapping[str, object] | None = None,
        equations: Mapping[str, Callable[[SimpleNamespace], object]] | None = None,
        *,
        parameters: Mapping[str, object] | None = None,
        noises: Mapping[str, Callable[[SimpleNamespace], object] | float] | None = None,
        time: str = 'continuous',
        convention: str = 'ito',
        element: str = 'state',
        label: str = 'system',
    ) -> None: ...
```

Hooks: `_get_states`, `_get_equations`, `_get_parameters`, `_get_noises`, `_get_time`, `_get_element_label`,
`_get_variables`, which returns a `DeclaredVariableSpec` per name, plain `StateSpec()` and `ParameterSpec()` by default,
so a subclass declares domains, quantities, or a fractional order there, and `_get_equation_interactions`. It builds a
`CompositeSystem` of one `DeclaredElement` and, per state, a `RateInteraction` with the rate and the noise, or a
`MapInteraction` in the mode `'replace'`. Tags follow from these parts: `'rate'` with `'noise'` when a noise is given,
or `'update'` with a discrete time.

Examples of two fields: the Lorenz flow of mathematics and the Wilson-Cowan populations of neuroscience in
continuous time, and the Hénon map of mathematics and the multiplier-accelerator of economics in discrete time.

Alternatives considered: two classes, `DifferentialEquationSystem` and `IteratedMapSystem`. They split one structure
by form. Or a name such as `EquationSystem`. "Dynamical system" is the established name of the structure.

### Maps of one step read the start of the step

The Hénon map needs `y' = b x` to read the old `x`. The discrete-time maps must all read the states of the start of
the step. The first task checks it with the Hénon map in both orders of states. If `MapInteraction` does not give it,
`DynamicalSystem` builds one `MapInteraction` on a vector state in place of one per state, and the spec stays as is.

### `NetworkDynamicalSystem` and `ReactionDiffusionSystem` subclass `DynamicalSystem`

Both are dynamical systems whose states sit on the sites of a space. They inherit the hooks and add their own.

```python
class NetworkDynamicalSystem(DynamicalSystem):
    def __init__(
        self,
        graph: nx.Graph | None = None,
        states: Mapping[str, object] | None = None,
        equations: Mapping[str, Callable[[SimpleNamespace], object]] | None = None,
        coupling: Mapping[str, Callable[[SimpleNamespace], object]] | None = None,
        *,
        parameters: Mapping[str, object] | None = None,
        noises: Mapping[str, Callable[[SimpleNamespace], object] | float] | None = None,
        convention: str = 'ito',
        element: str = 'nodes',
        label: str = 'system',
    ) -> None: ...


class ReactionDiffusionSystem(DynamicalSystem):
    def __init__(
        self,
        states: Mapping[str, object] | None = None,
        equations: Mapping[str, Callable[[SimpleNamespace], object]] | None = None,
        diffusivities: Mapping[str, float] | None = None,
        *,
        shape: Sequence[int] | int = 64,
        spacing: float = 1.0,
        boundary: str = 'periodic',
        discretization: str = 'finite-difference',
        parameters: Mapping[str, object] | None = None,
        noises: Mapping[str, Callable[[SimpleNamespace], object] | float] | None = None,
        convention: str = 'ito',
        element: str = 'medium',
        label: str = 'system',
    ) -> None: ...
```

`NetworkDynamicalSystem` builds `NetworkSpace(graph)`, a `FieldElement` of `DeclaredElement` members, the local
equations as rates, and per state of `coupling` a `CouplingInteraction` with `read` and `write` that state and the
function `g`. The function reads the node as `s.member` and its neighbour as `s.other`, as `CouplingInteraction`
documents. Its hooks add `_get_graph` and `_get_coupling_interactions`. A missing graph raises a `ValidationError`.
Examples: diffusion on a graph in mathematics and gap-junction exchange between cells in physiology.

`ReactionDiffusionSystem` builds a `LatticeSpace(shape, spacing=spacing, boundary=boundary)`, a `FieldElement` of
`DeclaredElement` members with `discretization`, the kinetics as rates, and per state of `diffusivities` a
`RateInteraction` with the rate `D laplacian(u)`. Its hooks add `_get_diffusivities`, `_get_space`, `_get_boundaries`,
which returns the boundary of each field for the `FieldElement`, and `_get_diffusion_interactions`. A chemistry
system overrides `_get_equation_interactions` to give its reactions, as the Gray-Scott system does today. Examples:
the Gray-Scott model of chemistry and the Fisher-Kolmogorov front of ecology.

Both run in continuous time only, so neither takes `time`. Coupled maps on networks, such as PageRank, keep their
current base, because a coupling map and a local map would both set the next value.

Alternatives considered: `space` as a parameter of `DynamicalSystem`. It would mix graph coupling, lattice diffusion,
and boundaries into one signature, which is no longer simple.

### `StockFlowSystem` stands apart

Stocks change only through named flows, which the trajectory reports. Equations of states would allow changes
outside flows, so `StockFlowSystem` is not a `DynamicalSystem`.

```python
class StockFlowSystem(BaseSystem[CompositeSystem]):
    def __init__(
        self,
        stocks: Mapping[str, float] | None = None,
        flows: Mapping[str, tuple[str | None, str | None, Callable[[SimpleNamespace], object]]] | None = None,
        *,
        parameters: Mapping[str, object] | None = None,
        element: str = 'stocks',
        label: str = 'system',
    ) -> None: ...
```

Each flow names the stock it leaves, or `None` for a source, the stock it enters, or `None` for a sink, and its rate.
It builds one `DeclaredElement` and per flow a `TransferInteraction` with `source` and `target` the element, `state`
the stock or `Source()`, `into` the stock or `Sink()`, `amount` the rate, and `flow` the name. The rate reads the
stocks and parameters as `s.source`, as the catalog systems do today. Hooks: `_get_stocks`, `_get_flows`,
`_get_parameters`, `_get_variables`, `_get_element_label`, `_get_flow_interactions`. Examples: the bathtub of
engineering and the arms race of social organizations.

Alternatives considered: the name `CompartmentalSystem`, the mathematical term of compartmental analysis. Biology uses
"compartment" for its own element, so the change keeps the name the user gave. A `rates` argument for stocks that
change outside flows, as six catalog systems have. It would be a second way to change a stock.

### Rebases and the element classes that leave

A system rebases when its shape matches, its build has no other interactions, and its element class serves it alone.
The element class leaves the module and its `__init__.py`, and its laws move into the system's hooks. The audit found
101 such element classes exported today, so this is breaking. The tasks name them per module.

The rebase lists of the snapshot, by field:

`DynamicalSystem` in continuous time:

- biology: `ecology/_cell_quota.py` `DroopCellQuotaSystem`, `ecology/_factories.py` `PenaeidShrimpSystem`,
  `epidemiology/_concurrency.py` `WattsMayConcurrencySystem`, `epidemiology/_vaccination.py` `SIRVaccinationSystem`,
  `neuroscience/_populations.py` `PolyhomeostaticNeuronSystem` and `WilsonCowanSystem`,
  `oncology/_antiangiogenic.py` `VascularCorePeripherySystem`, `oncology/_chemotherapy.py`
  `DrugResistanceSwitchingSystem`, `oncology/_combination_therapy.py` `HER2TargetedTherapySystem`,
  `oncology/_hormone_therapy.py` `PiecewiseLinearIASSystem`, `oncology/_optimal_therapy.py`
  `OptimalChemotherapySystem`, `physiology/_regulation.py` `SteerableFeedbackLoopSystem`,
  `systems_biology/_networks.py` `GlycolysisOscillationsSystem`
- chemistry: `kinetics/_reactors.py` `PulsedElectrochemicalReactorSystem`
- economics: `macroeconomics/_capital.py` `CapitalStockSystem`, `macroeconomics/_optimal_growth.py`
  `OptimalConsumptionSystem`, `macroeconomics/_purchasing_power.py` `CategorizedPurchasingPowerSystem`,
  `markets/_prices.py` `RiccatiPricesSystem`, `resources/_optimal_harvesting.py` `OptimalHarvestingSystem`
- engineering: `circuits/_circuits.py` `ChuaSystem`, `MemristorOscillatorSystem`, `MixedModeChuaSystem`, and
  `TunnelDiodeOscillatorSystem`, `circuits/_memristive_mixed_mode_oscillator.py`
  `MemristiveMixedModeOscillatorSystem`, `control/_control.py` `PendulumSwingUpSystem`, `PidLoopSystem`,
  `SatelliteTransferSystem`, and `ThermostatSystem`, `mechatronics/_vibrating_table.py` `VibratingTableSystem`,
  `power_systems/_electromechanics.py` `FaradayDiskSystem` and `RegenerativeBrakingSystem`
- mathematics: `dynamical_systems/_driven_flows.py` `OUDrivenRosslerSystem`, `dynamical_systems/_flows.py`
  `FiniteTimeBlowUpSystem`, `HyperchaoticRosslerSystem`, `IkedaSystem`, `LangfordSystem`, `NoseHooverSystem`,
  `SprottJerkSystem`, `SprottMixedFlowSystem`, and `ThomasSystem`, `dynamical_systems/_koopman.py`
  `KoopmanSlowManifoldSystem`, `dynamical_systems/_normal_forms.py` `CuspFormSystem`, `HopfFormSystem`,
  `SaddleNodeFormSystem`, and `TakensBogdanovFormSystem`, `dynamical_systems/_subcritical_forms.py`
  `SubcriticalHopfFormSystem`, `fractional_dynamics/_fractional.py` `FractionalLorenzSystem`,
  `FractionalOscillatorSystem`, `FractionalRelaxationSystem`, and `FractionalVanDerPolSystem`
- physics: `celestial_mechanics/_spin_orbit.py` `RelativisticOrbitSystem`, `climate/_energy_balance.py`
  `DaisyworldSystem` and `EnergyBalanceSystem`, `climate/_ice_ages.py` `CarbonCycleSystem`,
  `SaltzmanGlaciationSystem`, and `VanDerPolIceAgesSystem`, `climate/_low_order.py` `GeodynamoSystem` and
  `Lorenz84System`, `climate/_pollution.py` `PopulationPollutionSystem`, `climate/_tipping_types.py`
  `SuteraFraedrichTippingSystem`, `climate/_variability.py` `ENSOSystem` and `HasselmannSystem`, `heat/_cooling.py`
  `CoolingCoffeeSystem`, `mechanics/_accelerators.py` `FrictionOscillatorSystem`, `mechanics/_rotations.py`
  `RotatingBallsOnRodsSystem`, `nonlinear_oscillators/_oscillators.py` `AdlerEquationSystem`, `DuffingSystem`, and
  `ForcedTodaSystem`, `oceanography/_thermohaline.py` `StommelSystem`, `statistical_mechanics/_master_equations.py`
  `LandauRelaxationSystem`

`DynamicalSystem` in discrete time:

- biology: `ecology/_factories.py` `FibonacciRabbitsSystem`, `ecology/_instability.py` `IntrinsicInstabilitySystem`,
  `epidemiology/_recurrent.py` `DelayedSIRSSystem`, `evolution/_prebiotic.py` `EigenQuasispeciesSystem`,
  `neuroscience/_synapses.py` `AdaptrodeSystem`
- computer science: `neural_networks/_next_generation.py` `NextGenerationReservoirSystem`
- economics: `finance/_switching_exchange.py` `KirmanExchangeRateSystem`, `innovation/_diversity.py`
  `CreativePhaseTransitionSystem`, `macroeconomics/_growth.py` `MultiplierAcceleratorSystem`,
  `resources/_crop_livestock.py` `CropLivestockHouseholdSystem`
- engineering: `control/_koopman.py` `KoopmanModelPredictiveControlSystem`
- mathematics: `dynamical_systems/_maps.py` `BakerMapSystem`, `CDNMapSystem`, `CatMapSystem`, `CircleMapSystem`,
  `ConservativeHenonMapSystem`, `FermiUlamMapSystem`, `HenonMapSystem`, `IntermittencyMapSystem`, `LoziMapSystem`,
  `NewtonRaphsonMapSystem`, `StandardMapSystem`, `TentMapSystem`, and `TowelMapSystem`, `fractals/_quadratic.py`
  `JuliaSystem` and `MandelbrotSystem`
- physics: `active_matter/_flocks.py` `GregoireChateSystem`, `climate/_soil.py` `SoilTemperatureSystem`,
  `criticality/_sandpiles.py` `SelfOrganizedBranchingSystem`, `mechanics/_accelerators.py` `BeamBeamMapSystem`,
  `statistical_mechanics/_urns.py` `EhrenfestUrnSystem`

`NetworkDynamicalSystem`:

- biology: `physiology/_exchange.py` `NanotubeGapJunctionExchangeSystem`, `epidemiology/_networks.py`
  `MetapopulationSystem`, `epidemiology/_adaptive.py` `QuenchedMeanFieldSISSystem`
- physics: `climate/_tipping.py` `TippingCascadeNetworkSystem`, `synchronization/_networks.py`
  `KuramotoOnNetworksSystem`, if its edge group holds no state the trajectory reports
- mathematics: `networks/_processes.py` `NetworkDiffusionSystem`

`ReactionDiffusionSystem`:

- biology: `ecology/_fronts.py` `FisherKPPSystem`, `physiology/_cardiac.py` `AlievPanfilovSystem` and
  `BarkleySystem`, `physiology/_physiology.py` `FitzhughNagumoSystem`, `systems_biology/_min_proteins.py`
  `MinProteinSkeletonSystem`
- chemistry: `pattern_formation/_reaction_diffusion.py` `GrayScottSystem`, `BrusselatorSystem`, and
  `SchnakenbergSystem`
- physics: `heat/_heat.py` `HeatDiffusionSystem`, `criticality/_absorbing.py` `DirectedPercolationLangevinSystem`

`StockFlowSystem`:

- economics: `innovation/_bass.py` `BassDiffusionSystem`, `markets/_commodities.py` `CommodityCycleSystem`,
  `markets/_inventory.py` `CarDealerSystem`, `resources/_extraction.py` `OilEconomySystem` and
  `FisheryCapitalSystem`, `resources/_fisheries.py` `HilsaFisherySystem`, `resources/_food_balance.py`
  `RiceFoodBalanceSystem`
- engineering: `processes/_grain_storage.py` `GrainStorageSystem`, `processes/_processes.py` `BathtubSystem`,
  `processes/_systems.py` `FoodReliefSystem`
- social: `opinion_dynamics/_undecided.py` `UndecidedFlowsSystem`, `organizations/_traps.py`
  `SuccessToTheSuccessfulSystem`, `ShiftingTheBurdenSystem`, `EscalationSystem`, `PolicyResistanceSystem`, and
  `DriftToLowPerformanceSystem`

In the snapshot this is 70 and 31 systems for `DynamicalSystem`, 6 for `NetworkDynamicalSystem`, 10 for
`ReactionDiffusionSystem`, and 16 for `StockFlowSystem`. The four changes that land first may move a system, so each
rebase task first checks its list again.

### Conformance is exact

Before a module is rebased, the trajectory of every system in it is recorded with its defaults, `random_state=0`
where it takes one, and its tuned run settings, into the scratchpad, not the repository. After the rebase the names
must match and the values must be equal with `numpy.testing.assert_array_equal`. A tolerance would hide a changed
sum order, which chaotic systems such as the Chua circuit amplify within their run. A system that fails stays on its
base, and the task records why. The mirrored tests of the module, with their doctests, must pass unchanged in their
values.

## Risks / Trade-offs

- [A rate moved from an element law into an interaction compiles to a different sum order] → The exact comparison
  catches it, and the system keeps its base. The core test of each general system compares its trajectory with the
  same equations written as an element law.
- [Noise drawn by an interaction in place of an element law changes the order of the random stream] → Same exact
  check for the stochastic rebases, such as `HasselmannSystem`.
- [101 public element classes leave] → They are reachable through the system that uses them, and their equations
  stay in its hooks. The changelog lists them. Users who composed one write it with `DeclaredElement`.
- [The lists drift while the four earlier changes land] → Each rebase task checks its module again before it starts.
- [`rewrite-docstrings-and-docs` writes the systems page of the guide at the same time] → This change adds sections
  named after the classes to `core_components/systems.md` and creates the page only if it does not exist yet.

## Migration Plan

Each general system lands with its tests and docstring first, then its rebases field by field, each verified by the
exact comparison. The guide page, the inventory, and the full checks follow. Rollback of a rebase is the revert of its
module, since no other module depends on a leaving element class.

## Open Questions

- Should the system-dynamics translator write a subclass of `StockFlowSystem` for files whose stocks change only
  through flows? It would shorten translated modules, and it can follow in its own change.
- Should the six stock-flow systems with rate laws outside flows, such as `CascadeClassifierSystem` and
  `IrrigationSystem`, write those rates as named flows? Their trajectories would gain flow names, so it is a change of
  results that the user decides.
- The helper `_random_cells` exists in six modules of computer science, social, and biology. It may become a draw
  utility of the core. That is a separate decision on `laws.utils`.
