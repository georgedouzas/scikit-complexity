# Inventory of model classes per mechanism

Appendix of design.md. Paths are relative to `src/skcomplex/models/`.
- **Pure:** the class implements the mechanism alone and is rewritten to the core class, or deleted when nothing of
  its own is left.
- **Composite:** the class holds the mechanism among others and binds the core class for that part.
- **State holders:** elements that only declare the states a core element provides.
- **Excluded:** the class differs in substance and stays as it is.

## 1. Random walk: `RandomWalkInteraction`

Pure:
- mathematics/stochastic_processes/_walks.py:RandomWalkerElement
- mathematics/stochastic_processes/_walks.py:PersistentWalkerElement
- mathematics/stochastic_processes/_walks.py:MemoryWalkerElement
- mathematics/stochastic_processes/_walks.py:AgeingWalkerElement
- mathematics/networks/_processes.py:NetworkWalkerElement
- physics/soft_matter/_diffusion.py:BrownianParticleElement
- physics/soft_matter/_diffusion.py:ScaledWalkerElement
- physics/soft_matter/_anomalous.py:ContinuousTimeWalkerElement
- physics/soft_matter/_anomalous.py:LevyWalkerElement
- physics/soft_matter/_anomalous.py:LevyFlierElement
- physics/soft_matter/_anomalous.py:TransientWalkerElement
- physics/soft_matter/_anomalous.py:FractionalWalkerElement
- physics/statistical_mechanics/_traps.py:TrapWalkerElement
- physics/active_matter/_particles.py:TumblerElement
- physics/statistical_mechanics/_driven_particles.py:ZeroRangeHopInteraction
- physics/statistical_mechanics/_driven_particles.py:ZeroRangeHopsInteraction
- physics/statistical_mechanics/_driven_particles.py:ExclusionHopsInteraction
- physics/statistical_mechanics/_processes.py:ExclusionSiteElement
- biology/ecology/_clustering.py:HoppingInteraction
- social/organizations/_sabotage.py:SupervisorElement
- economics/markets/_goods_traders.py:GoodsTraderElement

Composite, including the copied `_wander` helpers:
- physics/statistical_mechanics/_processes.py:WalkerElement
- biology/epidemiology/_agents.py:WalkerElement
- biology/epidemiology/_agents.py:PersonElement
- biology/ecology/_wolf_sheep.py:SheepElement
- biology/ecology/_wolf_sheep.py:WolfElement
- biology/ecology/_agent_predation.py:PreyElement
- biology/ecology/_agent_predation.py:PredatorElement
- social/organizations/_hierarchies.py:ContenderElement
- social/organizations/_sabotage.py (its `_wander`)
- economics/macroeconomics/_metaphoria.py:SettlerElement

Excluded:
- physics/soft_matter/_traps.py:TrapSiteElement (a trap automaton)
- biology/collective_behaviour/_trails.py:WalkerElement (walks to a goal)
- biology/epidemiology/_agents.py:TravellerElement (moves between cities by a rule)
- social/segregation/_schelling.py:ResidentElement (moves when unhappy)
- social/organizations/_obesity.py:EaterElement (commutes)
- mathematics/stochastic_processes/_walks.py:WeakeningParticleElement (nonlinear drift)

## 2. Reaction and compartment flow: `ReactionInteraction`, `CompartmentElement`

Pure:
- biology/epidemiology/_compartments.py:InfectionInteraction
- biology/epidemiology/_compartments.py:RecoveryInteraction
- biology/epidemiology/_compartments.py:ImmunityLossInteraction
- chemistry/kinetics/_reactions.py:BindingInteraction
- chemistry/kinetics/_reactions.py:DissociationInteraction
- chemistry/kinetics/_reactions.py:SchloglElement
- biology/collective_behaviour/_decisions.py:NervousnessInteraction
- biology/ecology/_populations.py:PredationInteraction
- biology/ecology/_predation.py:PredatorPreyEventsInteraction
- biology/systems_biology/_networks.py:SynthesisInteraction
- biology/systems_biology/_networks.py:GeneralizedMassActionInteraction
- biology/systems_biology/_cascades.py:ActivationInteraction
- social/spreading/_rumours.py:RumourContactsInteraction
- physics/statistical_mechanics/_absorbing_processes.py:ReactingParticlesElement
- physics/statistical_mechanics/_driven_particles.py:ReservoirInteraction
- engineering/processes/_processes.py:PassageInteraction
- economics/innovation/_bass.py:BassMarketElement
- biology/ecology/_neutral.py:NeutralSpeciesElement
- mathematics/stochastic_processes/_queues.py:QueueElement
- mathematics/stochastic_processes/_counting.py:PoissonCounterElement
- biology/ecology/_clustering.py:HabitatPatchElement

State holders, replaced by `CompartmentElement`:
- biology/epidemiology/_compartments.py:CompartmentElement
- chemistry/kinetics/_reactions.py:SpeciesElement
- social/spreading/_rumours.py:ListenersElement
- biology/collective_behaviour/_decisions.py:RatsElement

Excluded:
- biology/physiology/_circulation.py:TransferInteraction
- biology/systems_biology/_networks.py:SSystemInteraction
- biology/systems_biology/_networks.py:HillRepressionInteraction
- biology/systems_biology/_genes.py:RepressionInteraction
- biology/evolution/_prebiotic.py:CatalysisInteraction
- chemistry/kinetics/_oscillators.py:BrusselatorReactionElement
- chemistry/kinetics/_oscillators.py:OregonatorElement
- engineering/processes/_processes.py:TubElement

## 3. Spreading: `SpreadingInteraction`

Pure:
- biology/epidemiology/_adaptive.py:HostElement
- biology/epidemiology/_adaptive.py:ActiveHostElement
- biology/epidemiology/_adaptive.py:CarrierElement
- biology/epidemiology/_networks.py:DegreeClassElement
- biology/epidemiology/_outbreaks.py:ImmunizingDegreeClassElement
- biology/epidemiology/_networks.py:SubpopulationElement
- biology/epidemiology/_vaccination.py:VaccinatedPopulationElement
- biology/epidemiology/_recurrent.py:VisitsInteraction
- biology/epidemiology/_recurrent.py:EpidemicCenterElement
- physics/statistical_mechanics/_processes.py:ContactSiteElement
- physics/statistical_mechanics/_processes.py:PercolationSiteElement
- physics/criticality/_criticality.py:ForestElement
- physics/criticality/_criticality.py:SpreadingForestElement
- social/opinion_dynamics/_diffusion.py:AdoptionInteraction

Composite:
- biology/epidemiology/_agents.py:PersonElement
- biology/epidemiology/_agents.py:WalkerElement
- biology/epidemiology/_agents.py:TravellerElement
- economics/innovation/_adoption.py:HouseholdElement

Excluded:
- chemistry/kinetics/_hodgepodge.py:HodgepodgeCellElement
- biology/oncology/_gene_viral_therapy.py:OncolyticVirusElement
- biology/epidemiology/_adaptive.py:RewiringInteraction

## 4. Coupling: `CouplingInteraction`

Pure:
- physics/synchronization/_kuramoto.py:PhaseCouplingInteraction
- physics/synchronization/_kuramoto.py:MeanFieldCouplingInteraction
- physics/synchronization/_kuramoto.py:DelayedCouplingInteraction
- physics/synchronization/_kuramoto.py:PhaseLagInteraction
- physics/synchronization/_networks.py:EdgeCouplingInteraction
- physics/synchronization/_chaos.py:DiffusiveCouplingInteraction
- physics/synchronization/_chaos.py:AggregateAveragingInteraction
- physics/synchronization/_networks.py:LogisticNodeElement
- physics/synchronization/_amplitude_death.py:GlobalCouplingInteraction
- physics/synchronization/_pulses.py:WinfreeMeanFieldInteraction
- physics/active_matter/_alignment.py:PolarAlignmentInteraction
- physics/active_matter/_collective.py:CuckerSmaleAlignmentInteraction
- engineering/power_systems/_grids.py:TransmissionLineInteraction
- social/opinion_dynamics/_bounded_confidence.py:ConfidentElement
- social/opinion_dynamics/_bounded_confidence.py:EncountersInteraction
- biology/neuroscience/_populations.py:PulseCouplingInteraction
- biology/neuroscience/_populations.py:MutualInhibitionInteraction
- biology/neuroscience/_networks.py:DelayedSynapseInteraction

Composite:
- biology/physiology/_exchange.py:ExchangingCellElement
- economics/finance/_wealth.py:SaverElement
- economics/finance/_wealth.py:MeanFieldSaverElement

Excluded:
- physics/synchronization/_swarmalators.py:SwarmalatorCouplingInteraction
- biology/neuroscience/_networks.py:GlobalInhibitionInteraction
- biology/neuroscience/_networks.py:TermanWangElement
- physics/mechanics/_pendulums.py:PendulumSpringInteraction
- biology/physiology/_circulation.py:TransferInteraction

## 5. Threshold and cascade: `ThresholdInteraction`

Pure:
- physics/criticality/_criticality.py:SandpileCellElement
- physics/criticality/_criticality.py:MannaCellElement
- earth/seismology/_faults.py:FaultElement
- social/spreading/_cascades.py:ThresholdNodeElement
- economics/finance/_contagion.py:DebtRankCascadeInteraction
- biology/systems_biology/_boolean.py:ThresholdFunctionInteraction
- computer_science/neural_networks/_hopfield.py:NeuronElement
- biology/neuroscience/_neurons.py:LeakyIntegrateAndFireElement
- biology/neuroscience/_neurons.py:IzhikevichElement
- biology/neuroscience/_networks.py:SpikingNeuronElement
- physics/synchronization/_pulses.py:IntegrateAndFireOscillatorElement
- biology/neuroscience/_excitable.py:ExcitableElementElement
- chemistry/kinetics/_cyclic.py:CyclicCellElement

Excluded:
- biology/neuroscience/_conductances.py:HodgkinHuxleyElement
- biology/evolution/_networks.py:TaxonElement
- biology/ecology/_sequential.py:AbundanceElement
- biology/oncology/_hormone_therapy.py:IntermittentSuppressionElement
- chemistry/kinetics/_hodgepodge.py:HodgepodgeCellElement

## 6. Choice by a rule: `ChoiceInteraction`

Pure:
- physics/statistical_mechanics/_fully_connected_spins.py:FullyConnectedSpinElement
- physics/criticality/_learning.py:LearningAgentElement
- physics/criticality/_evolutionary_learning.py:LearnerElement
- computer_science/neural_networks/_diluted.py:DilutedNeuronElement
- social/opinion_dynamics/_coevolution.py:ClubMemberElement
- social/opinion_dynamics/_standardization.py:StandardizationInteraction
- social/segregation/_blocks.py:BlockMovesInteraction
- social/cooperation/_spatial_games.py:PrisonerElement
- social/cooperation/_spatial_games.py:ContributorElement
- biology/evolution/_finite_populations.py:TangledNatureStepInteraction
- economics/markets/_brock_hommes.py:SpeculatorElement
- economics/markets/_fair_food.py:ConsumerElement

Excluded:
- physics/statistical_mechanics/_spins.py:IsingSpinElement (uses the core `SampleLaw`)
- physics/statistical_mechanics/_spins.py:PottsSpinElement (uses the core `SampleLaw`)
- physics/statistical_mechanics/_spins.py:XYSpinElement (uses the core `SampleLaw`)
- physics/statistical_mechanics/_graph_spins.py:GraphIsingSpinElement (uses the core `SampleLaw`)
- physics/statistical_mechanics/_spin_glasses.py:BondSpinElement (uses the core `SampleLaw`)
- social/organizations/_hierarchies.py:ContenderElement (a fight outcome)

## 7. Foraging and regrowth: `ForagingInteraction`, `RegrowthInteraction`

Foragers:
- biology/ecology/_wolf_sheep.py:SheepElement
- biology/ecology/_wolf_sheep.py:WolfElement
- biology/ecology/_agent_predation.py:PreyElement
- biology/ecology/_agent_predation.py:PredatorElement
- biology/ecology/_stigmergy.py:ForagerElement
- biology/ecology/_meadow.py:CowElement
- biology/ecology/_meadow.py:GrazingInteraction
- biology/ecology/_meadow.py:HerdingInteraction
- economics/resources/_commons.py:CowElement
- economics/resources/_commons.py:GrazingInteraction
- economics/markets/_sugarscape.py:ForagerElement
- economics/macroeconomics/_metaphoria.py:SettlerElement

Patches:
- biology/ecology/_wolf_sheep.py:GrassElement
- biology/ecology/_meadow.py:PastureElement
- economics/resources/_commons.py:PastureElement
- economics/markets/_sugarscape.py:SugarPatchElement
- economics/macroeconomics/_metaphoria.py:PlotElement

Excluded:
- biology/ecology/_stigmergy.py:StigmergicPreyElement (only sets parameters of `ForagerElement`)
- biology/ecology/_stigmergy.py:StigmergicPredatorElement (only sets parameters of `ForagerElement`)
- biology/collective_behaviour/_foraging.py:AntElement (pheromone-guided)
- economics/resources/_commons.py:HerderElement (herd-size decisions)

## 8. Evolution: `EvolutionInteraction`

Pure:
- computer_science/evolutionary_computation/_genetic_algorithms.py:EvolutionInteraction
- computer_science/evolutionary_computation/_landscapes.py:ReproductionInteraction
- biology/evolution/_finite_populations.py:MoranStepInteraction
- biology/ecology/_speciation.py:NeutralReplacementInteraction
- social/cooperation/_tournaments.py:TournamentInteraction
- economics/econophysics/_minority.py:MinorityGameInteraction

Excluded:
- biology/evolution/_finite_populations.py:AdaptiveClimbingInteraction
- biology/evolution/_finite_populations.py:ClimberElement
- economics/econophysics/_el_farol.py:PatronElement
- economics/econophysics/_minority.py:SpeculatorElement
- economics/econophysics/_kolkata.py:CrowdAvoidingDinerElement
- economics/markets/_perishables.py:CustomerElement
- economics/markets/_duopoly.py:CournotAdjustmentInteraction
- economics/markets/_shops.py:StoreElement
- biology/neuroscience/_synapses.py:AdaptrodeElement

## 9. Ornstein-Uhlenbeck and geometric noise: `OrnsteinUhlenbeckInteraction`

Pure:
- earth/climate/_variability.py:ClimateAnomalyElement
- physics/common/particles/_particles.py:LangevinThermostatInteraction
- economics/finance/_assets.py:GeometricBrownianMotionInteraction
- economics/finance/_correlated_assets.py:CorrelatedGeometricBrownianMotionInteraction
- engineering/traffic/_langevin.py:LangevinDrivingInteraction

Composite:
- physics/active_matter/_particles.py:OrnsteinUhlenbeckParticleElement
- physics/active_matter/_particles.py:ActiveParticleElement
- physics/soft_matter/_colored_noise.py:ColoredNoiseParticleElement
- mathematics/dynamical_systems/_driven_flows.py:DrivenRosslerElement
- economics/finance/_wealth.py:SaverElement
- economics/finance/_wealth.py:MeanFieldSaverElement

Excluded, each with a nonlinear drift or noise:
- physics/soft_matter/_brownian.py:DoubleWellElement
- physics/soft_matter/_brownian.py:WallParticleElement
- physics/soft_matter/_diffusion.py:ModulatedParticleElement
- earth/climate/_tipping_types.py:RampedFoldElement
- earth/climate/_tipping_types.py:StochasticEnergyBalanceElement
- mathematics/stochastic_processes/_intermittency.py:IntermittentVariableElement
- mathematics/stochastic_processes/_walks.py:WeakeningParticleElement
- physics/active_matter/_particles.py:DepotParticleElement
- physics/pattern_formation/_ordering.py:ThermalCahnHilliardElement
- physics/criticality/_absorbing.py:ActiveFieldElement
- physics/active_matter/_collective.py:PolarFluidElement
- engineering/robotics/_robots.py:LightSensitiveRobotElement

## 10. Imitation: `ImitationInteraction`

Pure:
- social/opinion_dynamics/_voters.py:VoterElement
- social/opinion_dynamics/_voters.py:VariantVoterElement
- social/opinion_dynamics/_voters.py:MajorityVoterElement
- social/opinion_dynamics/_persuasion.py:PersuaderElement
- social/opinion_dynamics/_persuasion.py:DiscussionsInteraction
- social/opinion_dynamics/_persuasion.py:SocialImpactInteraction
- social/opinion_dynamics/_coevolution.py:CoevolvingVoterElement
- social/opinion_dynamics/_axelrod.py:VillageElement
- social/organizations/_employment.py:WorkplaceElement
- biology/evolution/_games.py:CyclicSpeciesElement
- social/cooperation/_spatial_games.py:PrisonerElement
- social/cooperation/_spatial_games.py:ContributorElement

Composite:
- economics/finance/_order_book.py:StockTraderElement

Excluded:
- social/opinion_dynamics/_conventions.py:NamingGamesInteraction
- social/opinion_dynamics/_diffusion.py:BelieverElement
- social/opinion_dynamics/_cognition.py:ConversationsInteraction
- social/opinion_dynamics/_coevolution.py:SeparationInteraction

## 11. Forces and potentials: `PairForceInteraction`, `PotentialInteraction`, `DampingInteraction`, `ConstantForceInteraction`

Pair forces:
- biology/collective_behaviour/_flocks.py:MorsePotentialInteraction
- social/crowds/_pedestrians.py:SocialRepulsionInteraction
- physics/active_matter/_collective.py:SoftRepulsionInteraction
- physics/active_matter/_collective.py:PhoreticAttractionInteraction
- physics/common/particles/_particles.py:LennardJonesInteraction (stays as a realization)
- physics/common/particles/_particles.py:SoftSphereInteraction (stays as a realization)
- physics/common/particles/_particles.py:SpringInteraction (stays as a realization)
- physics/common/particles/_particles.py:GravityInteraction (stays as a realization)

External potentials and forces:
- social/crowds/_pedestrians.py:WallsInteraction
- social/crowds/_pedestrians.py:MotivationInteraction
- physics/common/particles/_particles.py:HarmonicPotentialInteraction (stays as a realization)
- physics/common/particles/_particles.py:ConstantForceInteraction (moves to the core)
- physics/common/particles/_particles.py:LinearDragInteraction (becomes the core `DampingInteraction`)
- physics/soft_matter/_diffusion.py:OpticalTrapInteraction
- physics/soft_matter/_diffusion.py:SpeckleFieldInteraction
- physics/soft_matter/_ratchets.py:TiltedWashboardInteraction
- physics/soft_matter/_ratchets.py:FlashingSawtoothInteraction
- physics/mechanics/_pendulums.py:MagnetInteraction
- physics/mechanics/_pendulums.py:SupportSpringInteraction
- physics/mechanics/_wells.py:ExponentialWellInteraction
- physics/waves/_strings.py:DrivenAnchorInteraction
- biology/collective_behaviour/_flocks.py:SelfPropulsionInteraction

Excluded:
- physics/synchronization/_swarmalators.py:SwarmalatorCouplingInteraction
- physics/mechanics/_springs.py:HarmonicForceInteraction
- physics/mechanics/_mechanics.py:ElasticCollisionInteraction

## 12. Class densities: `DensityElement`

Pure:
- biology/oncology/_immune_competition.py:CellsElement
- economics/econophysics/_welfare.py:WealthClassElement
- social/opinion_dynamics/_kinetic_opinion.py:OpinionLevelElement
- social/crowds/_criminality.py:CitizensElement
- social/crowds/_criminality.py:SecurityAgentsElement
- social/crowds/_evacuation.py:PedestriansElement

Excluded, each a specific encounter or rate realization that stays in its model:
- biology/oncology/_immune_competition.py:TumourGrowthInteraction
- biology/oncology/_immune_competition.py:ImmuneCompetitionInteraction
- economics/econophysics/_welfare.py:WealthExchangeInteraction
- economics/econophysics/_welfare.py:WelfarePolicyInteraction
- social/opinion_dynamics/_kinetic_opinion.py:OpinionEncountersInteraction
- social/crowds/_criminality.py:CriminalContagionInteraction
- social/crowds/_criminality.py:DeterrenceInteraction
- social/crowds/_criminality.py:RecruitmentInteraction
- social/crowds/_evacuation.py:WalkingInteraction
- social/crowds/_evacuation.py:CrowdingInteraction

## 13. Growth and decay: `GrowthInteraction`

Pure:
- biology/oncology/_growth.py:ExponentialGrowthInteraction
- biology/oncology/_growth.py:ExponentialLinearGrowthInteraction
- biology/oncology/_growth.py:PowerLawGrowthInteraction
- biology/oncology/_growth.py:GeneralizedLogisticGrowthInteraction
- biology/oncology/_growth.py:GompertzGrowthInteraction
- biology/oncology/_growth.py:LossInteraction
- biology/oncology/_growth.py:LogKillInteraction
- biology/ecology/_populations.py:ExponentialGrowthInteraction
- biology/ecology/_populations.py:LogisticGrowthInteraction
- biology/ecology/_populations.py:SeasonalLogisticGrowthInteraction
- biology/ecology/_populations.py:DelayedLogisticGrowthInteraction
- biology/ecology/_populations.py:MortalityInteraction
- biology/systems_biology/_networks.py:GoodwinStageElement
- biology/systems_biology/_cascades.py:CascadeStageElement
- economics/resources/_harvesting.py:StockElement
- economics/resources/_harvesting.py:HarvestInteraction
- economics/resources/_optimal_harvesting.py:HarvestedResourceElement
- economics/macroeconomics/_growth.py:SolowGrowthInteraction
- economics/macroeconomics/_capital.py:CapitalStockElement

State holders:
- biology/ecology/_populations.py:PopulationElement
- biology/oncology/_growth.py:TumourElement
- economics/macroeconomics/_growth.py:EconomyElement

Excluded:
- biology/oncology/_growth.py:VascularSupportInteraction
- biology/ecology/_succession.py:PioneerGrowthInteraction
- biology/ecology/_succession.py:SuccessionalGrowthInteraction
- biology/ecology/_communities.py:CompetitionInteraction
- biology/ecology/_communities.py:MutualismInteraction
- biology/ecology/_microbial_communities.py:GeneralizedLotkaVolterraInteraction
- biology/ecology/_predation.py:RosenzweigMacArthurInteraction
- biology/ecology/_factories.py:RabbitsElement

## 14. Alignment: `AlignmentInteraction`

Pure:
- physics/active_matter/_flocks.py:VicsekParticleElement
- physics/active_matter/_flocks.py:CohesiveParticleElement
- physics/active_matter/_flocks.py:NematicParticleElement
- physics/active_matter/_nonreciprocal.py:NonreciprocalMoverElement
- physics/active_matter/_collisions.py:BinaryCollisionsInteraction
- biology/collective_behaviour/_flocks.py:BoidElement
- biology/collective_behaviour/_leadership.py:ShoalMemberElement
- engineering/robotics/_robots.py:SwarmRobotElement

Excluded, listed under coupling:
- physics/active_matter/_collective.py:CuckerSmaleAlignmentInteraction
- physics/active_matter/_alignment.py:PolarAlignmentInteraction

## 15. Matching, feedback, and moved learning

- economics/common/markets: `MarketInteraction`, the seven protocols, `ClearingSpec`, `Order`, `Trade`. Their users:
  - economics/markets/_double_auction.py
  - economics/markets/_labour.py
  - economics/finance/_order_book.py
  - economics/macroeconomics/_credit.py
  - engineering/traffic/_ridesharing.py:DispatchInteraction, which uses `MatchLaw` today
- economics/common/institutions: `InstitutionInteraction` and its 7 users.
- economics/common/learning: `logit_choice`, `logit_probabilities`, `evolve`, and their users.

## D. Duplicated equations

- `StuartLandauElement` (core):
  - physics/synchronization/_amplitude_death.py:StuartLandauElement
  - physics/nonlinear_oscillators/_oscillators.py:StuartLandauElement
  - mathematics/dynamical_systems/_normal_forms.py:HopfElement
  - physics/synchronization/_kuramoto.py:MeanFieldAmplitudeElement
- `VanDerPolElement` (core):
  - physics/nonlinear_oscillators/_oscillators.py:VanDerPolElement
  - physics/nonlinear_oscillators/_oscillators.py:ForcedVanDerPolElement
  - earth/climate/_ice_ages.py:VanDerPolIceAgesElement
- FitzHugh-Nagumo (`biology/common/excitable`):
  - biology/physiology/_physiology.py:FitzHughNagumoElement
  - biology/neuroscience/_conductances.py:FitzHughNagumoNeuronElement
- `LogisticMapElement` (core), removing the copied `_logistic` helper:
  - mathematics/dynamical_systems/_maps.py:LogisticMapElement
  - physics/synchronization/_chaos.py:LogisticUnitElement
  - physics/synchronization/_networks.py:LogisticNodeElement
- Meadow and commons (core foraging and regrowth): the cow, pasture, and grazing classes of biology/ecology/_meadow.py
  and economics/resources/_commons.py.
- `GrowthInteraction`: both `ExponentialGrowthInteraction` classes.

## H. Group mechanisms on hyperedges: `HyperedgeInteraction`

- social/opinion_dynamics/_persuasion.py:DiscussionsInteraction
- social/cooperation/_spatial_games.py:ContributorElement
- economics/econophysics/_kolkata.py:RestaurantsInteraction
- economics/econophysics/_kolkata.py:DinerElement
- economics/econophysics/_minority.py:MinorityGameInteraction
- economics/econophysics/_el_farol.py:BarInteraction
