## MODIFIED Requirements

### Requirement: Mean-field systems read group reductions
The library SHALL build Kuramoto, Kuramoto-Sakaguchi chimeras, Winfree, theta neurons, LEGION, winners take all,
replicator dynamics, hypercycles, heterogeneous mean-field SIS, and Bouchaud-Mézard wealth with reductions over groups,
with no placeholder positions, no stand-in synapse, no vector state used as members, and no pair interaction covering
every pair.

#### Scenario: Kuramoto in linear time
- **WHEN** `kuramoto` is built with 50 oscillators
- **THEN** its system has no space and no pair interaction, and its drift equals
  `omega_i + (K / N) sum_j sin(theta_j - theta_i)` within 1e-12 at a random state

#### Scenario: Kuramoto transition
- **WHEN** `kuramoto` runs with 2000 oscillators at `K` equal to half and twice `K_c = 2 sigma sqrt(2 / pi)`
- **THEN** the time-averaged order parameter after the transient stays below 0.1 at half and exceeds 0.6 at twice

#### Scenario: Chimera populations
- **WHEN** `KuramotoSakaguchiChimeraSystem` runs with 32 oscillators per population
- **THEN** each population reads the means of `cos theta` and `sin theta` of both populations, and after the
  transient one population's order parameter exceeds 0.99 while the other's stays below 0.95

#### Scenario: Winfree mean influence
- **WHEN** `winfree` is built with 50 oscillators
- **THEN** its drift equals `omega_i - sin(theta_i) (K / N) sum_j (1 + cos theta_j)` within 1e-12, with no pair
  interaction and no self-influence correction

#### Scenario: Theta neurons with an instantaneous mean
- **WHEN** `ThetaNeuronsSystem` runs with 2000 neurons
- **THEN** its state holds only the phases, every neuron reads the mean pulse of all neurons at the current state, and
  the complex order parameter after the transient agrees with the Ott-Antonsen reduction of Luke, Barreto, and So
  (2013) within 0.05

#### Scenario: LEGION inhibitor reads the maximum
- **WHEN** `legion` runs on its default stimulus of two objects
- **THEN** the inhibitor is driven by the maximum oscillator activity, and after the transient no frame has both
  objects active at once while each object is active in some frame

#### Scenario: Winners take all with a group
- **WHEN** `WinnersTakeAllSystem` runs with its default inputs and `beta = 2`
- **THEN** its system has one inhibition interaction that reads the group total, whatever the number of units, and
  only the unit with input 1.1 stays active, with activity within 1e-3 of 1.1

#### Scenario: Replicator strategies as members
- **WHEN** `ReplicatorDynamicsSystem` runs rock, paper, scissors with 3 strategies as 3 members of a group
- **THEN** the shares sum to 1 within 1e-10 over `t` in `[0, 100]`, and the product of the shares stays within 1e-6
  of its initial value

#### Scenario: Hawks and doves
- **WHEN** `ReplicatorDynamicsSystem` runs hawks and doves with `V = 2` and `C = 3`
- **THEN** the hawk share approaches `2 / 3` within 1e-4

#### Scenario: Hypercycle on the simplex
- **WHEN** `hypercycles` runs with 4 molecules as members of a group on the cycle topology
- **THEN** the concentrations sum to 1 within 1e-10 and converge to a fixed point where every concentration exceeds
  0.01

#### Scenario: Degree classes as members
- **WHEN** `HeterogeneousMeanFieldSISSystem` runs with `lam` at half and twice `<k> / <k^2>`
- **THEN** its degree classes are members of a group with their degree and weight as parameters, and the infected
  density falls below 1e-6 at half and stays above 1e-3 at twice

#### Scenario: Wealth exchange with the mean
- **WHEN** `BouchaudMezardWealthSystem` is built with no graph and 50 agents
- **THEN** each agent exchanges with the group mean, the exchange part of the drift equals that of the
  complete-graph Laplacian form within 1e-12, and the exchange conserves the total wealth within 1e-12

### Requirement: Systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL have an entry in the system
inventory, and a system the inventory records as pending this change SHALL register under the name of its entry. A
system that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are rebuilt on group reductions under their names and field paths, and marked implemented:
  - `kuramoto` in `physics/synchronization`
  - `KuramotoSakaguchiChimeraSystem` in `physics/synchronization`
  - `winfree` in `physics/synchronization`
  - `ThetaNeuronsSystem` in `biology/neuroscience`
  - `legion` in `biology/neuroscience`
  - `WinnersTakeAllSystem` in `biology/neuroscience`
  - `ReplicatorDynamicsSystem` in `biology/evolution`
  - `hypercycles` in `biology/evolution`
  - `HeterogeneousMeanFieldSISSystem` in `biology/epidemiology`
  - `BouchaudMezardWealthSystem` in `economics/finance`

#### Scenario: Neighbourhood systems stay blocked
- **WHEN** the change is applied and the inventory check runs
- **THEN** `SzaboCellsSystem` and `ActivePassiveFlockingSystem` in `physics/active_matter` stay blocked on
  `add-adaptive-networks`, which specifies the neighbourhood reductions they need

#### Scenario: New system recorded
- **WHEN** the change registers a system with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Pulse-coupled oscillators are members of a group
The library SHALL build `MirolloStrogatzSystem` in `physics/synchronization` with the oscillators as members of a group,
whose firing pulses reach the other members through a reduction, rather than as one vector state in a Euclidean space.

#### Scenario: Mirollo-Strogatz synchronizes as a group
- **WHEN** `MirolloStrogatzSystem` runs with its default excitatory coupling from random phases
- **THEN** its oscillators are members of one group, and they fire in synchrony within the run

### Requirement: Boolean networks take any number of inputs
`RandomBooleanNetworkSystem` and `YeastCellCycleSystem` SHALL bind the inputs of every node through one variadic role, so a node
reads any number of inputs up to the number of nodes and no input role is bound to the target as padding.

#### Scenario: Eight inputs per node
- **WHEN** `RandomBooleanNetworkSystem(n_nodes=12, k=8)` is built and simulated for 50 steps
- **THEN** every node's update reads 8 distinct nodes through a table of 256 entries, and the run completes

#### Scenario: Too many inputs
- **WHEN** `RandomBooleanNetworkSystem(n_nodes=4, k=5)` is called
- **THEN** a named validation error is raised that names `k` and the number of nodes

#### Scenario: Yeast cycle unchanged
- **WHEN** `YeastCellCycleSystem()` runs from its default start for 15 steps
- **THEN** its trajectory equals the one recorded from the five-role version before the rebuild, and it ends in the
  stationary G1 state with only Cdh1 and Sic1 on

### Requirement: Ring roads are one group of cars
`OptimalVelocitySystem` and `IntelligentDriverSystem` SHALL hold their cars as one group and one interaction that reads the
car ahead by a periodic member offset, for any number of cars.

#### Scenario: One group whatever the number of cars
- **WHEN** `OptimalVelocitySystem(n_cars=50)` is built
- **THEN** the system holds one group of 50 cars and one interaction, and its trajectory has the arrays `cars.x` and
  `cars.v` with 50 columns

#### Scenario: Same dynamics as one interaction per car
- **WHEN** `OptimalVelocitySystem(n_cars=20)` and `IntelligentDriverSystem(n_cars=20)` run with their default settings
- **THEN** positions and speeds match those of the per-car versions recorded before the rebuild within a relative
  tolerance of 1e-8

#### Scenario: Jam on the rebuilt ring
- **WHEN** the rebuilt optimal velocity system runs above its critical density from near-uniform flow
- **THEN** the spread of the car speeds grows and a stop-and-go wave forms

### Requirement: Agent systems read groups through roles and the space through the system
`trafficking` and `MeadowCapacitySystem` SHALL bind every group their rules read to a role, `trafficking` and
`RingRoadDriversSystem` SHALL read the space's extent from the system view, no rule of an interaction in the library SHALL
read a group other than through its roles, and no rule SHALL read its own parameters by its label. Rules of agent
types, which have no roles, SHALL read the groups of their own system by relative label.

#### Scenario: Trafficking with two roles
- **WHEN** `TraffickingSystem(random_state=0)` is built and run for 90 days
- **THEN** its daily interaction binds the roles `women` and `premises`, has no `width` or `height` argument, and the
  bills and trips equal those recorded before the rebuild

#### Scenario: Nested meadow
- **WHEN** the meadow system built with random state 0 is placed as the subsystem `farm` of another system and run
- **THEN** it runs, its grazing and herding interactions bind the role `pasture`, and its cow counts equal those of the
  system run at the root

#### Scenario: Ring road without a length parameter
- **WHEN** `RingRoadDriversSystem(10, length=500.0)` runs
- **THEN** the drivers have no `road_length` parameter and every position stays in `[0, 500)`

#### Scenario: No rule reads its own label
- **WHEN** the library's layout test scans the source of every system
- **THEN** no rule reads a parameter by its own label, and no interaction names a group in `system.agents`,
  `system.count`, or `system.create` by a string or an attribute that is not one of its roles

### Requirement: Tumour growth laws
The library SHALL provide `ExponentialLinearGrowthSystem`, `PowerLawGrowthSystem`, `VonBertalanffySystem`, `GeneralizedLogisticSystem`,
`gompertz`, and `DynamicCarryingCapacitySystem` as system classes under `biology/oncology`, as the system inventory
lists them with their sources, each with a gallery example.

#### Scenario: Gompertz growth
- **WHEN** `gompertz` runs from a small tumour
- **THEN** its volume matches the closed-form Gompertz solution within 1e-6 relative and approaches the carrying
  capacity

### Requirement: Cell kill and toxicity
The library SHALL provide `SkipperLogKillSystem`, `NortonSimonSystem`, and `DrugToxicitySystem` as system classes under
`biology/oncology` and `biology/physiology`, as the system inventory lists them with their sources, each with a gallery
example.

#### Scenario: Log kill
- **WHEN** `SkipperLogKillSystem` receives equal impulsive doses on a non-growing tumour
- **THEN** each dose lowers the logarithm of the tumour size by the same amount within 1e-6

### Requirement: Chemotherapy
The library SHALL provide `DePillisRadunskayaSystem`, `MultidrugResistanceSystem`, `DrugResistanceSwitchingSystem`,
`DelayedRestingHuntingChemotherapySystem`, `DelayedMetastasisSystem`, `CellCycleChemotherapySystem`, and
`ErythropoiesisMercaptopurineSystem` as system classes under `biology/oncology`, as the system inventory lists them with
their sources, each with a gallery example.

#### Scenario: Chemotherapy clears a tumour
- **WHEN** `DePillisRadunskayaSystem` runs untreated and then with a constant chemotherapy infusion of 1
- **THEN** the untreated tumour settles at its high equilibrium, and the treated tumour falls below 1% of its initial
  size while the normal cells stay above 0.7 of their carrying capacity

### Requirement: Immunotherapy
The library SHALL provide `KirschnerPanettaSystem`, `DendriticCellVaccineSystem`, `AntiPD1CheckpointSystem`, `DelayedDCVaccineSystem`,
and `TumourImmuneSuppressionSystem` as system classes under `biology/oncology`, as the system inventory lists them with
their sources, each with a gallery example.

#### Scenario: Tumour-immune oscillations and clearance
- **WHEN** `KirschnerPanettaSystem` runs with an antigenicity in the oscillatory range and then with adoptive therapy and
  IL-2 above the book's treatment factors `a1 = 500` and `a2 = 7e7`
- **THEN** the untreated tumour and effector cells oscillate, and the treated tumour is cleared

### Requirement: Anti-angiogenic therapy
The library SHALL provide `hahnfeldt` and `VascularCorePeripherySystem` as system classes under `biology/oncology`, as
the system inventory lists them with their sources, each with a gallery example.

#### Scenario: Hahnfeldt equilibrium
- **WHEN** `hahnfeldt` runs untreated
- **THEN** the tumour and carrying capacity approach the equilibrium volume `(b/d)^(3/2)` within 1%

### Requirement: Radiotherapy
The library SHALL provide `RadiotherapyCompetitionSystem` and `LQRadiosensitizerSystem` as system classes under
`biology/oncology`, as the system inventory lists them with their sources, each with a gallery example.

#### Scenario: Linear-quadratic fractions
- **WHEN** `LQRadiosensitizerSystem` delivers a fraction of dose `d` without radiosensitizer
- **THEN** the surviving tumour fraction of that fraction is `exp(-alpha d - beta d^2)` within 1e-6

### Requirement: Hormone therapy
The library SHALL provide `ContinuousAndrogenSuppressionSystem`, `IdetaIntermittentAndrogenSystem`, `ProstateCellQuotaSystem`,
`DroopCellQuotaSystem`, `ProstateAndrogenReceptorSystem`, `PiecewiseLinearIASSystem`, and `TamoxifenCellCycleSystem` as system
classes under `biology/ecology` and `biology/oncology`, as the system inventory lists them with their sources, each
with a gallery example.

#### Scenario: Intermittent suppression
- **WHEN** `IdetaIntermittentAndrogenSystem` runs with its PSA thresholds
- **THEN** the therapy switches on as PSA rises through the upper threshold and off as it falls through the lower one,
  and PSA cycles around the thresholds, overshooting them by the androgen delay, while treatment is effective

### Requirement: Gene, viral, nanoparticle, and stem-cell therapies
The library SHALL provide `TCRGeneTherapySystem`, `SirnaTGFBetaSystem`, `OncolyticVirusSystem`, `OncolyticVirusDendriticSystem`,
`NanoparticleCoronaSystem`, `TwoCompartmentPharmacokineticsSystem`, `StemCellDifferentiationSystem`, `BoneMarrowBloodLeukaemiaSystem`,
and `LeukaemiaStemCellsSystem` as system classes under `biology/oncology`, `biology/physiology`, and
`chemistry/kinetics`, as the system inventory lists them with their sources, each with a gallery example.

#### Scenario: Two-compartment pharmacokinetics
- **WHEN** `TwoCompartmentPharmacokineticsSystem` receives one bolus dose
- **THEN** the central concentration matches the closed-form bi-exponential solution within 1e-6 relative, and the total
  drug decays by elimination only

### Requirement: Combination therapies
The library SHALL provide `chemoimmunotherapy`, `ClonalChemoimmunotherapySystem`, `chemovirotherapy`,
`GliomaChemovirotherapySystem`, `HER2TargetedTherapySystem`, and `AngiogenesisImmunotherapySystem` as system classes under
`biology/oncology`, as the system inventory lists them with their sources, each with a gallery example.

#### Scenario: Combination beats each therapy alone
- **WHEN** `chemovirotherapy` runs with the book's chemotherapy alone, the virus alone, and both
- **THEN** the tumour at the end of the default run of 64 days, before the virus bursts through the tumour, is
  smallest under the combination

### Requirement: Cellular network systems
The library SHALL provide `PhosphorylationOscillatorSystem`, `BiochemicalSystemsTheorySystem`, `ThreeGeneBooleanNetworkSystem`,
`MammalianCellCycleSystem`, `BooleanRepressilatorSystem`, `GoodwinOscillatorSystem`, and `GeneralizedLotkaVolterraSystem` as system
classes under `biology/ecology` and `biology/systems_biology`, as the system inventory lists them with their sources,
each with a gallery example.

#### Scenario: Goodwin oscillations
- **WHEN** `GoodwinOscillatorSystem` runs with three variables and Hill coefficients 6 and 12
- **THEN** it settles on its stable fixed point at 6 and on a limit cycle at 12, the threshold for three equal
  degradation rates being 8

### Requirement: Surveyed systems are recorded in the inventory
The inventory SHALL record every system of Padmanabhan et al. 2021, Konieczny et al. 2023, and Raman 2021 under the
source keys `Pa`, `Ko`, and `Rm`: the systems above as pending this change, systems pending other changes, systems blocked
on `add-constraint-based-models`, out-of-scope entries, and the new sources of systems the library already implements.

#### Scenario: Source of an implemented system
- **WHEN** a contributor reads the inventory entry of `MichaelisMentenSystem`
- **THEN** its sources include `Ko 1.1.4` and `Rm 5.3.1`

### Requirement: Mean-field and disordered spin systems
The library SHALL provide `CurieWeissSystem`, `MattisSpinGlassSystem`, `EdwardsAndersonSystem`, and `SherringtonKirkpatrickSystem` as
system classes under `physics/statistical_mechanics`, as the system inventory lists them with their sources, each
with a gallery example.

#### Scenario: Curie-Weiss magnetization
- **WHEN** `CurieWeissSystem` with 2000 spins runs to stationarity below and above the critical temperature `T_c = J`
- **THEN** the mean absolute magnetization at `T = 0.5 J` matches the positive root of `m = tanh(m J / T)` within 0.02,
  and at `T = 2 J` it is below 0.05

### Requirement: Driven and noisy particles
The library SHALL provide `ColoredNoiseLangevinSystem`, `OnOffIntermittencySystem`, `BiasedTrapWalkSystem`,
`TiltedPeriodicPotentialSystem`, `FlashingRatchetSystem`, and `ShakenSpringBlockSystem` as system classes under
`mathematics/stochastic_processes` and `physics/soft_matter`, as the system inventory lists them with their sources, each
with a gallery example.

#### Scenario: Ratchet current
- **WHEN** `FlashingRatchetSystem` runs with an asymmetric and with a symmetric sawtooth potential at the same switching rate
- **THEN** the asymmetric potential gives a mean current whose sign follows the asymmetry and exceeds three standard
  errors, and the symmetric one gives a current within three standard errors of zero

### Requirement: Birth-death and interacting particle processes
The library SHALL provide `BirthDeathQueueSystem`, `AbsorbingBirthDeathSystem`, `ZeroRangeProcessSystem`, `DissipativeZeroRangeSystem`,
and `asep` as system classes under `mathematics/stochastic_processes` and `physics/statistical_mechanics`, as the
system inventory lists them with their sources, each with a gallery example.

#### Scenario: Stationary queue
- **WHEN** `BirthDeathQueueSystem` runs as an M/M/1 queue with load `rho = 0.5`
- **THEN** the time-averaged occupation distribution matches `(1 - rho) rho^n` within 0.01 for `n` up to 5

### Requirement: Active matter and traffic
The library SHALL provide `AligningActiveBrownianParticlesSystem`, `BinaryCollisionFlockingSystem`, `LangevinTrafficSystem`, and
`TrafficDensityEquationSystem` as system classes under `engineering/traffic` and `physics/active_matter`, as the system
inventory lists them with their sources, each with a gallery example.

#### Scenario: Alignment transition
- **WHEN** `AligningActiveBrownianParticlesSystem` runs at a low and a high alignment strength at fixed density and noise
- **THEN** the stationary polarization is below 0.2 at the low strength and above 0.6 at the high strength

### Requirement: Agents, evolution, and populations
The library SHALL provide `PerishableGoodsDuopolySystem`, `StandardizedAgentsSystem`, `OriginFixationWalkSystem`,
`NeutralBiodiversitySystem`, `NeutralClusteringSystem`, and `HeterogeneousMeanFieldSIRSystem` as system classes under
`biology/ecology`, `biology/epidemiology`, `biology/evolution`, `economics/markets`, and `social/opinion_dynamics`, as
the system inventory lists them with their sources, each with a gallery example.

#### Scenario: Epidemic threshold on networks
- **WHEN** `HeterogeneousMeanFieldSIRSystem` runs on a degree distribution with transmission rates below and above the
  threshold `<k>/(<k^2> - <k>)` of the recovery rate, close to `<k>/<k^2>`
- **THEN** the final epidemic size is below 1% below the threshold and above 5% above it

### Requirement: Maps and coupled oscillators
The library SHALL provide `CubicChaoticMapSystem`, `SkewTentMapSystem`, `DoublingMapSystem`, `SubcriticalHopfFormSystem`,
`OscillatorDeathSystem`, and `HeterogeneousGlobalCouplingSystem` as system classes under `mathematics/dynamical_systems`
and `physics/synchronization`, as the system inventory lists them with their sources, each with a gallery example.

#### Scenario: Invariant density of the tent map
- **WHEN** `SkewTentMapSystem` iterates an ensemble of 10000 points for 200 steps
- **THEN** the histogram of the final points is uniform on [0, 1] within the sampling error of each of 20 bins

### Requirement: Bertin systems are recorded in the inventory
The inventory SHALL record every system of Bertin 2021 under the source key `Bt`: the systems above as pending this
change, `LevyFlightSystem` and `trap` as pending `add-anomalous-processes`, `RandomWalkWithMemorySystem` and
`DilutedHopfieldSystem` as pending `add-stochastic-maps`, `RandomGraphEnsemblesSystem` as pending `add-adaptive-networks`,
`RandomEnergySystem` and `perceptron_satisfiability` as blocked on `add-static-models`, and the new sources of systems
the library already has.

#### Scenario: Source of an existing system
- **WHEN** a contributor reads the inventory entry of `kuramoto`
- **THEN** its sources include `Bt 7.3.1`

### Requirement: Probabilistic Boolean networks
The library SHALL provide `ProbabilisticBooleanNetworkSystem` in `biology/systems_biology`, a Boolean network that draws
each node's transfer function from its candidates with given probabilities at every step, as Raman 2021 section 7.3.1
poses it.

#### Scenario: Selection probabilities
- **WHEN** `ProbabilisticBooleanNetworkSystem` runs for many steps with one node choosing between two functions with
  probabilities 0.7 and 0.3
- **THEN** the share of steps that used the first function is 0.7 within three standard errors

### Requirement: Bertin systems on stochastic maps
The library SHALL provide `RandomWalkWithMemorySystem` in `mathematics/stochastic_processes` and `DilutedHopfieldSystem` in `computer_science/neural_networks`, as Bertin 2021 poses them and the system inventory records them, built on this
change's capability, each with a gallery example.

#### Scenario: Registered under the inventory names
- **WHEN** the change is applied and the inventory check runs
- **THEN** both systems are registered under their field paths and marked implemented

### Requirement: Intrinsic-instability populations
The library SHALL provide `IntrinsicInstabilitySystem` in `biology/ecology`, the resident and invading populations of
Jensen 2023 section 13.2, Eqs. (13.3) and (13.4), whose resident population follows an Ornstein-Uhlenbeck step with
noise scaled by the square root of its size and is suppressed by an invader that a rare mutation seeds and that then
grows exponentially, built on this change's capability, with a gallery example.

#### Scenario: Leaving the fixed point from within
- **WHEN** `IntrinsicInstabilitySystem` runs for 4000 steps with its default parameters and random state 0
- **THEN** the invader is seeded before step 3000, the resident population stays within 0.15 of 1 before the
  invader appears, and at the end the resident population is below 0.5 and the invader above 1

### Requirement: Rewriting systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL have an entry in the system
inventory, and a system the inventory records as pending this change SHALL register under the name of its entry. A
system that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `KochCurveSystem` in `mathematics/fractals` (Th 3.5.2, Tr 3.5)
  - `CantorSetSystem` in `mathematics/fractals` (Th 3.5.2, Tr 3.5)
  - `SierpinskiSubstitutionSystem` in `mathematics/fractals` (Th 3.5.2, Tr 3.5)

#### Scenario: New system recorded
- **WHEN** the change registers a system with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Lévy flights and trap systems
The library SHALL provide `LevyFlightSystem` in `physics/soft_matter` and `trap` in `physics/statistical_mechanics`, as Bertin 2021 poses them and the system inventory records them, built on this
change's capability, each with a gallery example.

#### Scenario: Registered under the inventory names
- **WHEN** the change is applied and the inventory check runs
- **THEN** both systems are registered under their field paths and marked implemented

### Requirement: Network structure analysis system
The library SHALL provide `NetworkStructureAnalysisSystem` under `mathematics/networks` (Rm 3.1, 3.3, 3.4, Bt 6.1.1,
6.1.2), a system class built on the public core that takes a NetworkX graph, records it as one agent per node,
and reports its centralities, mixing, communities, and triangle z-score as observables. The inventory SHALL mark it as
implemented.

#### Scenario: Measures of a generated network
- **WHEN** `NetworkStructureAnalysisSystem` runs on a stochastic block system of 4 blocks of 100 nodes
- **THEN** its trajectory holds the network and observables of clustering, mean path length, assortativity, modularity,
  and the agreement of the detected communities with the blocks above 0.9

#### Scenario: Given graph
- **WHEN** `NetworkStructureAnalysisSystem` runs on Zachary's karate club graph from NetworkX
- **THEN** its reported clustering and mean path length equal those of NetworkX within 1e-12, and the community
  detection splits the club into at least two communities with modularity above 0.35

#### Scenario: Gallery example
- **WHEN** a user opens the gallery example of `NetworkStructureAnalysisSystem`
- **THEN** it shows the Rerun screenshot of the graph colored by community, and below it the degree distribution and the
  motif z-scores as analysis figures

### Requirement: Quantum spin chain systems
The library SHALL provide the quantum Heisenberg system (Mihailović et al. 2023 §7.3) under
`physics/statistical_mechanics`: a chain of spins one half with the XXZ Hamiltonian
`J Σ (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j) - h Σ S^z_i` over neighbouring pairs, optional dephasing at a rate
`gamma`, and an initial Néel state, a single flipped spin, a domain wall, or the ground state.

#### Scenario: Ground state of the ring
- **WHEN** the spectrum of `QuantumHeisenbergSystem` with 4 periodic sites, `J = 1`, `Δ = 1`, and `h = 0` is computed
- **THEN** its lowest eigenvalue is -2 within 1e-12

#### Scenario: Single magnon spreads as a Bessel light cone
- **WHEN** `QuantumHeisenbergSystem` runs with 12 periodic sites, `J = 1`, `Δ = 0`, and one flipped spin at site 6, to
  time 2
- **THEN** at every frame the probability `1/2 - <S^z_{6+d}>` of finding the flip at distance `d` equals
  `J_d(J t)²` within 1e-5, and the total magnetisation stays constant within 1e-8

#### Scenario: Dephasing destroys coherence
- **WHEN** `QuantumHeisenbergSystem` runs with 6 periodic sites from a Néel state with `gamma = 0.1`
- **THEN** the solver is the Lindblad solver, the purity never increases from frame to frame and ends below 1, and
  the total magnetisation stays constant within 1e-8

### Requirement: Quantum systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL have an entry in the system
inventory, and a system the inventory records as out of scope or blocked by this change SHALL register under the name of
its entry. A system that the library already implements SHALL gain the new formulation under the same name, not be
registered twice.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `QuantumHeisenbergSystem` in `physics/statistical_mechanics` (Mi 7.3)
  - `schrodinger` in `physics/waves` (Sc 4), with the operator formulation

#### Scenario: Gallery example of a quantum system
- **WHEN** the gallery runs the example `docs/examples/physics/statistical_mechanics/plot_quantum_heisenberg.py`
- **THEN** it creates `QuantumHeisenbergSystem` by name, streams the run to Rerun within the recording budget, its
  thumbnail is the space-time image of the local magnetisation with a light cone, and its analysis figure compares the
  ground-state energy per site with `1/4 - ln 2`

### Requirement: Agent economy systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL have an entry in the system
inventory, and a system the inventory records as pending this change SHALL register under the name of its entry. A
system that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `CreditNetworksSystem` in `economics/macroeconomics`
  - `FractionalReserveBankingSystem` in `economics/macroeconomics`
  - `DoubleAuctionSystem` in `economics/markets`
  - `LabourMarketSystem` in `economics/markets`
  - `FairFoodSystem` in `economics/markets`
  - `sugarscape` in `economics/markets`
  - `BrockHommesSystem` in `economics/markets`
  - `KineticWealthExchangeSystem` in `economics/econophysics`
  - `MinorityGameSystem` in `economics/econophysics`
  - `ElFarolSystem` in `economics/econophysics`
  - `KolkataRestaurantSystem` in `economics/econophysics`
  - `axelrod` in `social/opinion_dynamics`

#### Scenario: Example entries left pending
- **WHEN** the change is applied
- **THEN** these entries, which repeat patterns the implemented systems already exercise, stay pending this change:
  - `GoodsTradersSystem` in `economics/markets`
  - `shops` in `economics/markets`
  - `barter` in `economics/markets`
  - `order_book` in `economics/finance`
  - `TragedyOfTheCommonsSystem` in `economics/resources`
  - `ridesharing` in `engineering/traffic`

#### Scenario: New system recorded
- **WHEN** the change registers a system with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: System-dynamics and stock-flow consistent systems
The library SHALL provide Meadows' systems zoo (bathtub, coffee cooling, thermostat, inventory with delays, renewable
and nonrenewable resources), the system traps (success to the successful, shifting the burden, escalation), Bass
diffusion, the beer game, and a stock-flow consistent monetary economy as system classes, with `bathtub` and
`CarDealerSystem` rebuilt on flows under their names. The tragedy of the commons belongs to `add-agent-economies`.

#### Scenario: Inventory oscillation
- **WHEN** the car-dealership inventory system receives a 10% step in demand
- **THEN** the inventory oscillates before settling, and shortening the perception delay reduces the oscillation's
  amplitude

### Requirement: Stock-flow systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL have an entry in the system
inventory, and a system the inventory records as pending this change SHALL register under the name of its entry. A
system that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries, which probe flows, sources and sinks, auxiliaries, lookups, delays, smoothing, loops, and
  discrete flows in distinct ways, are registered under their field paths and marked implemented:
  - `StockFlowConsistentEconomySystem` in `economics/macroeconomics`
  - `CommodityCycleSystem` in `economics/markets`
  - `BeerGameSystem` in `economics/markets`
  - `BassDiffusionSystem` in `economics/innovation`
  - `SuccessToTheSuccessfulSystem` in `social/organizations`
  - `ShiftingTheBurdenSystem` in `social/organizations`
  - `escalation` in `social/organizations`
  - `RiceMillingSystem` in `engineering/processes`
  - `SolidWasteSystem` in `engineering/processes`
  - `FoodReliefSystem` in `engineering/processes`
  - `irrigation` in `engineering/processes`

#### Scenario: Repeated patterns registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries, which repeat patterns of stocks and flows that the systems above probe, are registered under
  their field paths and marked implemented: `EnergyReturnSystem` in `physics/climate`, and `cocoa`, `HilsaFisherySystem`, and
  `food_security` in `economics/resources`

#### Scenario: Meadows' resource systems rebuilt
- **WHEN** the change is applied
- **THEN** Meadows' nonrenewable and renewable resource systems stay registered as `OilEconomySystem` and
  `FisheryCapitalSystem`, rebuilt on flows, and the duplicate survey entries `renewable_resource` and
  `nonrenewable_resource` are removed from the inventory

#### Scenario: New system recorded
- **WHEN** the change registers a system with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: The chemical Langevin equation switches by noise
`ChemicalLangevinEquationSystem` with its defaults SHALL switch between its two stable states within 200 time units.

#### Scenario: Noise-driven switching
- **WHEN** the default chemical Langevin equation runs until 200 with Euler-Maruyama steps of 1e-3 from seed 0
- **THEN** after time 20 the number of molecules is below 30 at some frame and above 90 at another

### Requirement: Social impact flips under net pressure
`SocialImpactSystem` SHALL flip an opinion when the net pressure to change plus the random field is positive.

#### Scenario: No pressure keeps opinions
- **WHEN** a social impact system without noise starts with every agent of opinion -1
- **THEN** no opinion changes in 5 steps

### Requirement: System defaults show the described behaviour
`RestrictedThreeBodySystem`, `DiffusionInASpeckleFieldSystem`, and `legion` with their defaults SHALL show chaos, anomalous
diffusion, and alternating objects within short runs, and `BurridgeKnopoffSystem` SHALL register a solver that does not
stall on its stick-slip switches.

#### Scenario: Chaotic restricted three-body orbit
- **WHEN** the default restricted three-body problem and one started 1e-8 away run until 20
- **THEN** the body passes within 0.2 of the Moon and the two paths end more than 1e-6 apart

#### Scenario: Anomalous diffusion in a speckle field
- **WHEN** the default speckle field runs until 60 in frames of 0.5
- **THEN** the particles move less than 5 units on average in the first frame, and from time 10 their mean squared
  displacement grows as `t^a` with `0.1 < a < 0.8`

#### Scenario: Both LEGION objects fire
- **WHEN** the default LEGION network runs until 100
- **THEN** each of its two objects starts an active phase

#### Scenario: Stick-slip without a stall
- **WHEN** the default Burridge-Knopoff fault runs until 400 with its registered solver
- **THEN** no warning is raised, no block moves backwards, and some frame holds a slip above 0.3

### Requirement: Echo state network system
The library SHALL provide `EchoStateNetworkSystem` in `computer_science/neural_networks`, a reservoir of leaky tanh units driven by an input, with a curated gallery example.

#### Scenario: Echo state property
- **WHEN** `EchoStateNetworkSystem` runs from two initial states with spectral radius 0.9
- **THEN** the two reservoir trajectories converge, which is the echo state property

### Requirement: Chemical Langevin equations of reaction networks
The chemistry module SHALL describe binding and dissociation reactions as chemical Langevin equations, in which one
Wiener process per reaction drives every species the reaction changes, and SHALL provide the chemical Langevin
equation of the enzyme network `E + S <-> C -> E + P` as a system class.

#### Scenario: Binding conserves a difference
- **WHEN** `A + B -> C` runs as a chemical Langevin equation with `EulerMaruyamaSolver`
- **THEN** `A - B` stays at its initial value within 1e-9 relative error at every frame, while `A` fluctuates

#### Scenario: Enzyme totals under noise
- **WHEN** the chemical Langevin network runs for its default time with a fixed random state
- **THEN** `E + C` and `S + C + P` stay at their initial values within 1e-9 relative error at every frame

#### Scenario: Two descriptions at once
- **WHEN** a binding reaction is both stochastic and in Langevin mode
- **THEN** compiling raises a named error that names the interaction

### Requirement: Correlated noise systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL have an entry in the system
inventory, and a system the inventory records as pending this change SHALL register under the name of its entry. A
system that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `ChemicalLangevinNetworkSystem` in `chemistry/kinetics`
  - `CorrelatedGeometricBrownianMotionSystem` in `economics/finance`

#### Scenario: New system recorded
- **WHEN** the change registers a system with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Fractional dynamics systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL have an entry in the system
inventory, and a system the inventory records as pending this change SHALL register under the name of its entry. A
system that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `FractionalRelaxationSystem` in `mathematics/fractional_dynamics`
  - `FractionalOscillatorSystem` in `mathematics/fractional_dynamics`
  - `FractionalVanDerPolSystem` in `mathematics/fractional_dynamics`
  - `FractionalLorenzSystem` in `mathematics/fractional_dynamics`

#### Scenario: New system recorded
- **WHEN** the change registers a system with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Repeated agent economy systems
The library SHALL provide `GoodsTradersSystem`, `shops`, `barter`, `order_book`, `TragedyOfTheCommonsSystem`, and
`ridesharing` as system classes, each with a curated gallery example. Each SHALL be a `System` under discrete
time with a phases schedule, with agents as elements with rule laws in groups, built on the core's matching, transfer,
and invariant laws and on the trading realizations of `skcomplex.systems.economics.realizations`, and none SHALL add a
core primitive.

#### Scenario: Barter reaches the contract curve
- **WHEN** `barter` runs Edgeworth exchange between two traders with Cobb-Douglas preferences until no trade improves
  both
- **THEN** the final allocation lies on the contract curve within 2% of the total endowment

#### Scenario: Commons collapse
- **WHEN** `TragedyOfTheCommonsSystem` runs with more herders than the meadow carries and no regulating adaptive parameter
- **THEN** the grass falls below a tenth of its capacity and the herd shrinks after it

#### Scenario: Trades conserve goods
- **WHEN** `GoodsTradersSystem` runs for 50 steps with an invariant on the total of each good
- **THEN** no invariant is violated, since every trade is a transfer between two agents

### Requirement: Evolutionary Sherrington-Kirkpatrick game
The library SHALL provide `EvolutionarySKGameSystem` in `physics/criticality`, after Lesmana et al. (2026, arXiv
2604.15669): learning agents on a Sherrington-Kirkpatrick landscape with Bak-Sneppen replacement of the least fit,
built from the core's group, reduction, draw, and update primitives under discrete time, with the extremal replacement
an update switched by the group minimum, and with its softmax choice defined in its system package rather than imported
from another field's realizations.

#### Scenario: Rules of the game
- **WHEN** a user reads the system class of `EvolutionarySKGameSystem`
- **THEN** couplings are symmetric Gaussian with variance `1/N`, each agent's incentive is updated from its local field
  with a memory weight, actions are softmax draws at the agent's temperature, and the least fit agent is replaced with a
  temperature redrawn uniformly on [0, 2]

#### Scenario: Self-organisation to the edge of ergodicity breaking
- **WHEN** `EvolutionarySKGameSystem` runs with 256 agents and memory 0.75 past its transient
- **THEN** the replacement cascades follow a power law with exponent within 0.15 of 3/2, and the mean reward exceeds the
  reward of the same game run at every fixed temperature tried in a sweep

#### Scenario: Example of the evolutionary game
- **WHEN** a user opens `docs/examples/physics/criticality/plot_evolutionary_sk_game.py`
- **THEN** its thumbnail shows the evolved temperature distribution and its analysis compares the cascade sizes and the
  ergodicity-breaking diagnostic with the paper's results

### Requirement: Emergence systems are recorded in the system inventory
The inventory SHALL record `EvolutionarySKGameSystem` with its arXiv source, its primitives, and its priority, pending this
change until it is implemented.

#### Scenario: Entry of the evolutionary game
- **WHEN** a contributor reads the inventory entry of `EvolutionarySKGameSystem`
- **THEN** it cites arXiv 2604.15669, is tagged with discrete time, no space, a group, no binding, and the discrete
  law class, and has priority `core`

### Requirement: Kinetic systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL have an entry in the system
inventory, and a system the inventory records as pending this change SHALL register under the name of its entry. A
system that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `ImmuneCompetitionSystem` in `biology/oncology`
  - `KineticWealthWelfareSystem` in `economics/econophysics`
  - `KineticOpinionSystem` in `social/opinion_dynamics`
  - `evacuation` in `social/crowds`
  - `criminality` in `social/crowds`

#### Scenario: New system recorded
- **WHEN** the change registers a system with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Optimal control systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL have an entry in the system
inventory, and a system the inventory records as pending this change SHALL register under the name of its entry. A
system that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `SIRVaccinationSystem` in `biology/epidemiology`
  - `OptimalConsumptionSystem` in `economics/macroeconomics`
  - `OptimalHarvestingSystem` in `economics/resources`
  - `PidInvertedPendulumSystem` in `engineering/control`
  - `LqrOscillatorSystem` in `engineering/control`
  - `OptimalChemotherapySystem` in `biology/oncology`
  - `OptimalImmunotherapySystem` in `biology/oncology`

#### Scenario: New system recorded
- **WHEN** the change registers a system with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Group automata systems
The library SHALL provide the majority automaton on the free group, the lamplighter automaton, the discrete Laplacian
as a linear automaton, the reversible Q2R Ising automaton, the golden mean shift, and the majority automaton on the
Grigorchuk group as system classes, each with a gallery example whose thumbnail is its Rerun screenshot and
which shows an analysis plot from its run.

#### Scenario: Q2R conserves energy
- **WHEN** the Q2R automaton runs on a periodic lattice
- **THEN** the Ising energy of the configuration stays constant at every step

#### Scenario: Grigorchuk automaton
- **WHEN** a user creates `GrigorchukAutomatonSystem` by name and runs it with its default parameters
- **THEN** it runs the majority rule with the memory set `{1, a, b, c, d}` on a ball of radius 8 with 271 sites, every
  interior cell takes the majority value of itself and its four neighbours, and the space reports the ball sizes 1, 5,
  11, 23, 40, 68, 108, 176, and 271 for radii 0 to 8

#### Scenario: Grigorchuk gallery plot
- **WHEN** a user opens the gallery example of `GrigorchukAutomatonSystem`
- **THEN** below the Rerun screenshot of the Cayley ball it plots the ball sizes for radii 0 to 10, and the text states
  that the local log-log slope rises while `log |B(n)| / n` falls, so the growth is faster than polynomial and slower
  than exponential

### Requirement: Group cellular automaton systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL have an entry in the system
inventory, and a system the inventory records as pending this change SHALL register under the name of its entry. A
system that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `FreeGroupAutomatonSystem` in `computer_science/automata`
  - `LamplighterAutomatonSystem` in `computer_science/automata`
  - `HeisenbergAutomatonSystem` in `computer_science/automata`
  - `BaumslagSolitarAutomatonSystem` in `computer_science/automata`
  - `LinearRingAutomatonSystem` in `computer_science/automata`
  - `subshifts` in `computer_science/automata`
  - `GrigorchukAutomatonSystem` in `computer_science/automata` (CC 6.9)

#### Scenario: New system recorded
- **WHEN** the change registers a system with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Surveyed physics systems
The library SHALL provide `swarmalators`, `EhrenfestUrnSystem`, `DirectedPercolationLangevinSystem`, `SKGameSystem`,
`KellerSegelSystem`, `BenardMarangoniSystem`, `ChargedSchrodingerSystem`, `CahnHilliardSystem`, and `AllenCahnSystem` as
system classes built from core primitives, each with a curated gallery example.

#### Scenario: Ehrenfest urn relaxes to the binomial
- **WHEN** `EhrenfestUrnSystem` runs from all balls in one urn
- **THEN** the count in the first urn relaxes to a binomial distribution with mean half the balls

#### Scenario: Directed percolation threshold
- **WHEN** `DirectedPercolationLangevinSystem` sweeps its control parameter across the absorbing transition
- **THEN** the stationary density vanishes below a threshold and grows above it

### Requirement: Surveyed biology systems
The library SHALL provide `HormoneCascadeAmplifierSystem`, `MinProteinSkeletonSystem`, `NanotubeGapJunctionExchangeSystem`,
`ClosedContourCellExchangeSystem`, `AdaptiveResponseCostsSystem`, `GlucoseDualHormoneControlSystem`,
`CoagulationPositiveFeedbackSystem`, and `CouzinLeadershipSystem` as system classes built from core primitives, each with
a curated gallery example.

#### Scenario: Informed minority leads the group
- **WHEN** `CouzinLeadershipSystem` runs with a small informed minority sharing a preferred direction
- **THEN** the group's mean heading turns towards that direction, and the accuracy grows with the minority's size

### Requirement: Surveyed earth, economics, and mathematics systems
The library SHALL provide `SuteraFraedrichTippingSystem`, `RampedStochasticFoldSystem`, `metaphoria`,
`WilhiteTradeNetworksSystem`, `KoopmanSlowManifoldSystem`, and `OUDrivenRosslerSystem` as system classes built from core
primitives, each with a curated gallery example.

#### Scenario: Tipping lags a ramp
- **WHEN** `RampedStochasticFoldSystem` ramps its control parameter through the fold faster than the system relaxes
- **THEN** the state tips after the control passes the static fold point, later for faster ramps

### Requirement: Adaptive network systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL have an entry in the system
inventory, and a system the inventory records as pending this change SHALL register under the name of its entry. A
system that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `KuramotoOnNetworksSystem` in `physics/synchronization`
  - `CoupledMapLatticesOnNetworksSystem` in `physics/synchronization`
  - `vicsek` in `physics/active_matter`
  - `GregoireChateSystem` in `physics/active_matter`
  - `GinelliNematicsSystem` in `physics/active_matter`
  - `SequentiallyLinearEcosystemSystem` in `biology/ecology`
  - `QuenchedMeanFieldSISSystem` in `biology/epidemiology`
  - `AdaptiveSISSystem` in `biology/epidemiology`
  - `ActivityDrivenEpidemicsSystem` in `biology/epidemiology`
  - `JainKrishnaSystem` in `biology/evolution`
  - `SoleManrubiaSystem` in `biology/evolution`
  - `HegselmannKrauseSystem` in `social/opinion_dynamics`
  - `CoevolvingVoterSystem` in `social/opinion_dynamics`
  - `JoinAClubSystem` in `social/opinion_dynamics`
  - `barabasi_albert` in `mathematics/networks`
  - `HolmeKimSystem` in `mathematics/networks`
  - `VertexCopyingSystem` in `mathematics/networks`
  - `NetworkAttackSystem` in `mathematics/networks`
  - `RandomGraphEnsemblesSystem` in `mathematics/networks`

#### Scenario: New system recorded
- **WHEN** the change registers a system with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Dynamics analysis probe systems
The library SHALL provide `KinouchiCopelliSystem`, `NonreciprocalCahnHilliardSystem`, `NonreciprocalVicsekSystem`, and `TippingCascadeNetworkSystem`, each with a curated gallery example.

#### Scenario: Maximal dynamic range
- **WHEN** `KinouchiCopelliSystem` sweeps the branching ratio
- **THEN** the dynamic range peaks at a branching ratio of 1, as Kinouchi and Copelli (2006) found

#### Scenario: Traveling patterns
- **WHEN** `NonreciprocalCahnHilliardSystem` runs with nonreciprocal coupling above threshold
- **THEN** its demixed domains travel, as Brauns and Marchetti (2023) predict

### Requirement: Higher-order probe systems
The library SHALL provide the probe systems of higher-order interactions that the inventory records as pending or
blocked by this change, each with a curated gallery example, built on the classes and reductions this change adds.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their names and field paths and marked implemented:
  - `HypergraphKuramotoSystem` in `physics/synchronization`
  - `HypergraphConsensusSystem` in `social/opinion_dynamics`
  - `SimplicialSocialDilemmasSystem` and `HypergraphPublicGoodsSystem` in `social/cooperation`
  - `SimplicialContagionSystem` and `HypergraphExposureSisSystem` in `social/spreading`
  - `TriadicPercolationSystem`, `HypergraphRandomWalkSystem`, and `GrowingSimplicialComplexSystem` in `mathematics/networks`

#### Scenario: Explosive synchronization
- **WHEN** `HypergraphKuramotoSystem` sweeps the three-body coupling up and down
- **THEN** the order parameter shows hysteresis, as Majhi et al. (2022) review

#### Scenario: Bistable contagion
- **WHEN** `SimplicialContagionSystem` runs with a triangle infection rate above the critical value
- **THEN** the epidemic has two stable prevalences depending on the initial seed

### Requirement: Group mechanisms run as symbolic groupings
`GalamMajoritySystem` and `PublicGoodsGameSystem` SHALL be built on `GroupingInteraction(form='symbolic')`, and SHALL agree with
their previous trajectories over 20 seeds. Kolkata and the minority game SHALL keep their agent forms.

#### Scenario: Galam on the symbolic grouping
- **WHEN** `GalamMajoritySystem` runs from a share of 0.6 for 20 steps over 20 seeds
- **THEN** the mean final share agrees with the previous agent form within the standard error of the 20 seeds

### Requirement: Hamiltonian mechanics systems
The library SHALL build the Kepler problem `TwoBodyOrbitSystem`, the restricted three-body problem, and Hénon-Heiles on
`HamiltonianInteraction`, and their catalog entries SHALL pass the Hamiltonian as an observable with an invariant
tolerance, so the `InvariantObserver` checks its drift.

#### Scenario: Kepler orbit
- **WHEN** `TwoBodyOrbitSystem` runs for ten periods of a bound orbit with `Yoshida4Solver`
- **THEN** the orbit closes on itself within 1e-6 relative error and the energy drift stays below 1e-8

#### Scenario: Hénon-Heiles below the escape energy
- **WHEN** `HenonHeilesSystem` runs from an energy of 1/12 to time 1000
- **THEN** the orbit stays bounded and the energy stays within its invariant tolerance

### Requirement: Variational mechanics systems are recorded in the system inventory
Every system this change rebuilds SHALL keep its name and field path in the system inventory, and SHALL list the
`variational-mechanics` primitive.

#### Scenario: Inventory systems rebuilt
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are rebuilt on `HamiltonianInteraction` under their names and field paths, list the primitive
  `variational-mechanics`, and are marked implemented:
  - `TwoBodyOrbitSystem` in `physics/celestial_mechanics`
  - `RestrictedThreeBodySystem` in `physics/celestial_mechanics`
  - `HenonHeilesSystem` in `physics/nonlinear_oscillators`

### Requirement: Elman simple recurrent network
The library SHALL provide Elman's simple recurrent network (Gros 2015 §10.5.1, Eq. 10.18, and Elman 1990) as a
system class under discrete time in epochs. It SHALL nest the discrete map
`s(t) = σ(A s(t-1) + D x(t) + b)`, `y(t) = σ(W s(t) + c)` with adaptive shaped parameters, and train it with a
`TrainingInteraction` on the squared error of predicting the next input, by default on Elman's temporal XOR stream.

#### Scenario: Temporal XOR
- **WHEN** `simulate(ElmanNetworkSystem(), until=200, random_state=0)` trains 4 hidden units with Adam on a
  stream of 3000 bits
- **THEN** after 200 epochs the metric `xor_error`, the mean squared error of the predictions of every third bit, is
  below 0.05, the metric `random_error` of the predictions of the other bits stays above 0.2, and
  `training/loss` at the last frame is below its value at frame 0

#### Scenario: Elman's original rule
- **WHEN** `ElmanNetworkSystem` is created with `truncation=1`
- **THEN** it trains with the previous hidden states as constant context inputs, and `xor_error` still falls below
  `random_error` within 500 epochs

#### Scenario: Gallery example of the Elman network
- **WHEN** the gallery runs `docs/examples/computer_science/neural_networks/plot_elman_network.py`
- **THEN** it records at most 300 frames, its thumbnail is the time series view of the loss and the two metrics, in
  which `xor_error` falls toward 0 while `random_error` stays near 0.25, and below the screenshot a Matplotlib figure
  shows the mean squared prediction error at each position of the bit triple before and after training, with the
  third position alone near 0 after training

### Requirement: The generalized Lotka-Volterra matrix is a parameter
`GeneralizedLotkaVolterraSystem` SHALL hold its interaction matrix as a shaped parameter of a `CouplingInteraction`, so a
schedule or an experiment changes it, and SHALL keep its trajectory.

#### Scenario: Rebuilt community keeps its trajectory
- **WHEN** `GeneralizedLotkaVolterraSystem` runs with its catalog defaults
- **THEN** its trajectory equals the trajectory of the previous constant matrix within 1e-10

#### Scenario: Matrix swept by an experiment
- **WHEN** a grid experiment sets the matrix parameter to two matrices that differ off the diagonal
- **THEN** the two members of the experiment differ in their final abundances

### Requirement: Random Lotka-Volterra community with k-body interactions
The library SHALL provide `HigherOrderRandomLotkaVolterraSystem` in `biology/ecology` (Sidhom and Galla 2024, arXiv
2409.10990): a group of species with the growth `x_i (k - x_i)` and the rate `x_i sum_(j<k) a_ijk x_j x_k` of the
paper's Eq. 2, written `x_i contraction(a, x, x) / 2`, from a random coupling tensor `a` of order 3 with the mean, variance,
and correlation of its Eqs. 4 and 5, a shaped parameter drawn by the system class from its `random_state`, so a schedule or an
experiment changes it, with a curated gallery example.

#### Scenario: No interactions
- **WHEN** the system runs with a tensor of zeros
- **THEN** every species converges to its carrying capacity within 1e-6

#### Scenario: Random community
- **WHEN** the system runs with its defaults, the equations, scaling, and parameters of Sidhom and Galla (2024, arXiv
  2409.10990), over 20 seeds
- **THEN** every abundance stays non-negative and finite, and the mean fraction of surviving species matches the
  paper's result at those parameters within its stated precision: 200 species, `mu = -4`, `sigma = 1`, and `gamma = 0`
  give 0.928 from its Eq. 17, and the paper states no number, so the check allows 0.02, three standard errors of 20 seeds

### Requirement: Learning systems are recorded in the system inventory
Every system this change implements SHALL register under the field path of its subfield and SHALL have an entry in the
system inventory, and a system the inventory records as blocked by this change SHALL register under the name of its
entry.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `ElmanNetworkSystem` in `computer_science/neural_networks`, with the sources Gr 10.5.1 and Elman 1990
  - `HigherOrderRandomLotkaVolterraSystem` in `biology/ecology`

### Requirement: Nonlocal field systems
The library SHALL provide, as system classes built on the general interactions and the nonlocal operators:
- `AmariNeuralFieldSystem` under `biology/neuroscience` (2112.12147 §1.3.1);
- `DoubleGyreSystem` under `physics/oceanography` (1910.00583 §III.6.1);
- `GktTrafficSystem` under `engineering/traffic` (cond-mat/0012229 §III.E.4);
- `ActiveViscoelasticGelSystem` under `physics/active_matter` (2306.08868 VI).

#### Scenario: Amari bump
- **WHEN** the Amari neural field `tau u_t = -u + convolve(H(u), w) + h` runs on a line with a lateral-inhibition
  kernel `w`, a Heaviside firing rate, and a resting level `h < 0`, from a localized excitation
- **THEN** a stationary bump forms, and its width is within 5% of the root `a` of `W(a) + h = 0` with `w(a) < 0`,
  where `W(a)` is the integral of `w` from 0 to `a`

#### Scenario: Double gyre circulation
- **WHEN** the quasi-geostrophic double gyre runs in a closed square basin from rest under the symmetric double-gyre
  wind stress `-tau0 cos(2 pi y / L)` in its viscous, steady regime
- **THEN** the streamfunction forms two gyres of opposite sign, one on each side of the mid-basin latitude, with each
  extremum in the western fifth of the basin, and along the central latitude of each gyre the interior streamfunction
  is within 10% of the Sverdrup balance `beta psi_x = curl(tau)`

#### Scenario: Stop-and-go waves
- **WHEN** the gas-kinetic-based traffic system runs on a ring road with a small local density perturbation, once at a
  density of 15 vehicles per kilometre and once at a density in its unstable range
- **THEN** the perturbation decays at the low density, and at the unstable density it grows into congested clusters
  whose downstream fronts move upstream at between 10 and 25 kilometres per hour

#### Scenario: Contractile gel
- **WHEN** the active viscoelastic gel runs from a small perturbation of uniform cell density above its contractile
  threshold
- **THEN** the density concentrates into peaks, and the velocity at every step solves the force balance within 1e-10

### Requirement: Field couplings are interactions
`legion` and `NanotubeGapJunctionExchangeSystem` SHALL write the coupling of their fields with `CouplingInteraction`, so
`TermanWangElement` and `ExchangingCellElement` hold only the dynamics of one site, and SHALL keep their trajectories.

#### Scenario: Rebuilt couplings keep the trajectories
- **WHEN** `legion` and `NanotubeGapJunctionExchangeSystem` run with their catalog defaults
- **THEN** the trajectory of `NanotubeGapJunctionExchangeSystem` equals that of the previous elements within 1e-10, and the
  rates of `legion` equal those of the previous elements within 1e-12 at many random states, with its trajectories
  equal within 1e-10 over a short window before rounding grows

### Requirement: Static systems
The library SHALL provide `LQCellSurvivalSystem` in `biology/oncology`, `RandomEnergySystem` and `ContinuumPercolationSystem`
in `physics/statistical_mechanics`, `OligopolyEquilibriaSystem` in `economics/markets`, `VonThunenSystem` in
`economics/resources`, `GravityUrbanScalingSystem` in `social/organizations`, and `MandelbrotLeastEffortSystem` in
`social/language`, as system classes under `StaticTime` that run through `simulate` and sweep
their parameters through `sweep`.

#### Scenario: Fractionation
- **WHEN** `LQCellSurvivalSystem` evaluates 30 fractions of 2 Gy with `alpha/beta = 10`
- **THEN** the biologically effective dose is 72 Gy (Padmanabhan et al. 2021 Eq. 6.8), and the dose-protraction factor
  of a constant dose rate tends to 1 as the delivery time tends to 0

#### Scenario: Glass transition
- **WHEN** a `sweep` sweeps the temperature of `RandomEnergySystem` with `N = 20` over 10 replicates
- **THEN** the entropy follows `ln 2 - 1/(4 T^2)` above `T_g = 0.6006` and stays below 0.1 at `T = 0.3`

#### Scenario: Mixed price equilibrium
- **WHEN** `OligopolyEquilibriaSystem` solves the pricing game of `m` firms with loyal share `alpha` and switchers
  `beta = 1 - m alpha`
- **THEN** the symmetric equilibrium price distribution is `F(P) = 1 - (alpha (1 - P)/(beta P))^(1/(m - 1))` on
  `[alpha/(alpha + beta), 1]`, and each firm's expected profit is `alpha` (Forrest 2018 Lemma 20.1)

#### Scenario: Von Thünen rings
- **WHEN** `VonThunenSystem` assigns the cells of a plain around one market to the crop of highest bid rent
  `Y_i (p_i - c_i) - Y_i f_i d` at distance `d`, after Giordano (2024, arXiv 2407.02181 §7)
- **THEN** the land use forms concentric rings whose boundaries lie where the bid rents of neighbouring crops are equal,
  within one cell, and raising a crop's transport rate `f_i` shrinks its ring

#### Scenario: Percolation threshold
- **WHEN** a `sweep` sweeps the filling factor of `ContinuumPercolationSystem` over 0.8 to 1.5 for two sizes
- **THEN** the spanning probabilities that `clusters` gives cross between filling factors 1.0 and 1.25

#### Scenario: Urban scaling exponent
- **WHEN** a `sweep` runs `GravityUrbanScalingSystem` at five populations from 100 to 1600 with area exponent `alpha`
  and interaction exponent `gamma < 2`, and `scaling(..., kind='power')` fits the total interaction against the
  population
- **THEN** the fitted exponent is within 0.05 of `2 - alpha gamma / 2`, for one seed up to `gamma = 1` and for totals
  averaged over seeds at `gamma = 1.5`, where close pairs make single cities heavy-tailed

#### Scenario: Least-effort frequencies
- **WHEN** `MandelbrotLeastEffortSystem` solves for word costs `c_i = log(i + 1)`
- **THEN** the probabilities follow `(i + 1)^-lambda` with the solved `lambda` within 1e-9, and the ratio of cost to
  entropy at the solved `lambda` is lower than at `lambda ± 0.1`

### Requirement: Dynamic equilibrium, satisfiability, and search systems
The library SHALL provide the fictitious play of `OligopolyEquilibriaSystem` with `dynamics='fictitious-play'`, built on
`ChoiceInteraction` and `ReinforcementInteraction`, `perceptron_satisfiability` in `computer_science/neural_networks`,
built on `SatisfiabilityInteraction`, and `EvolutionaryOptimisersSystem` in `computer_science/evolutionary_computation`,
built on `OptimizationInteraction` and on `SamplingInteraction` with `cooling`, all under discrete time.

#### Scenario: Fictitious play of a discretised duopoly
- **WHEN** `OligopolyEquilibriaSystem` runs with `dynamics='fictitious-play'` for 2000 rounds
- **THEN** the empirical distribution of prices is within 0.2 of `F(P)` in the Kolmogorov distance

#### Scenario: Satisfiable fraction
- **WHEN** a `sweep` runs `perceptron_satisfiability` at `N = 15` and `N = 21` over 20 replicates
- **THEN** the share of satisfiable replicates against `p/N` falls from above 0.9 at 0.5 to below 0.1 at 1.2

#### Scenario: Algorithms compared
- **WHEN** `EvolutionaryOptimisersSystem` runs differential evolution, particle swarm, and annealing on Rastrigin for 100
  generations with the same seed
- **THEN** every run reports a best value that never increases, and differential evolution ends below 1e-3

### Requirement: Static systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL be marked implemented in
the system inventory, with a gallery example whose analysis plot comes from a `sweep` over its swept parameter.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `LQCellSurvivalSystem` in `biology/oncology`
  - `RandomEnergySystem` in `physics/statistical_mechanics`
  - `OligopolyEquilibriaSystem` in `economics/markets`
  - `VonThunenSystem` in `economics/resources`
  - `perceptron_satisfiability` in `computer_science/neural_networks`
  - `EvolutionaryOptimisersSystem` in `computer_science/evolutionary_computation`
  - `ContinuumPercolationSystem` in `physics/statistical_mechanics`
  - `GravityUrbanScalingSystem` in `social/organizations`
  - `MandelbrotLeastEffortSystem` in `social/language`

### Requirement: Dry active nematics
The library SHALL provide `DryActiveNematicsSystem` under the field path `physics/active_matter` as a system class
built only from public names, a `FieldElement` on a lattice under continuous time: a conserved density coupled to a
symmetric traceless alignment tensor, with an active current along the divergence of the tensor (1207.2929 §II.2). The
inventory SHALL list it as implemented.

#### Scenario: Dry active nematic conserves density
- **WHEN** `DryActiveNematicsSystem` runs on a periodic lattice for 100 time units
- **THEN** the total density stays constant within 1e-10 relative error

### Requirement: Active nematic and polar gels
The library SHALL provide `ActiveGelsSystem` under the field path `physics/active_matter` as a system class: a
symmetric traceless tensor, with `order='nematic'`, or a polarisation vector, with `order='polar'`, transported by its
co-rotational derivative in an overdamped Stokes flow driven by the elastic and active stresses (1207.2929 §III), with
the Landau-de Gennes energy in its own package. The inventory SHALL list it as implemented.

#### Scenario: Passive nematic order
- **WHEN** `ActiveGelsSystem` runs with `order='nematic'` at zero activity from a uniform tensor with Landau coefficients
  `a < 0` and `c`
- **THEN** the velocity stays below 1e-8, and the scalar order relaxes to `sqrt(-2 a / c)` within 1e-6

#### Scenario: Active turbulence conserves topological charge
- **WHEN** the extensile nematic gel runs on a periodic 2D lattice from a nearly aligned state with an activity well
  above the bend instability threshold
- **THEN** `defects` finds pairs of `+1/2` and `-1/2` defects, the total charge stays 0 at every frame, and the
  mean kinetic energy settles at a nonzero value

#### Scenario: Spontaneous flow of a polar film
- **WHEN** the polar gel runs in a film between walls of fixed polarisation, given by the `symmetry` setting, at
  activities 20% below and 20% above the threshold of the linear stability analysis
- **THEN** the flow decays below the threshold and grows to a steady nonzero flow above it

### Requirement: Engineering systems on components and Lagrangians
The library SHALL provide `HeatedRoomSystem` in `engineering/control`, and `DcMotorSystem` and `QuarterCarSystem` in
`engineering/mechatronics`, as system classes, each with the standard observables of its catalog entry.
`QuarterCarSystem` SHALL be built on `SpringDamperInteraction`, and the inertia of `DcMotorSystem` on `EulerLagrangeInteraction`.

#### Scenario: Heated room
- **WHEN** `HeatedRoomSystem` runs from a cold room with the thermostat set to 20 degrees and a hysteresis band of 1 degree
- **THEN** the room temperature rises to the band and then stays within it, with the heater switching on and off

#### Scenario: Quarter car over a step
- **WHEN** `QuarterCarSystem` drives over a road step with its standard values
- **THEN** the body and wheel displacements settle at the step height, and the energy of the springs decays

### Requirement: Acausal component systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL have an entry in the system
inventory, tagged with the time kind, space kind, component kinds, interaction bindings, and law classes of the system
it builds.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `HeatedRoomSystem` in `engineering/control`
  - `QuarterCarSystem` in `engineering/mechatronics`
  - `DcMotorSystem` in `engineering/mechatronics`

### Requirement: Lattice gauge system samples the Wilson action
The library SHALL provide `LatticeGaugeFieldsSystem` in `physics/waves`, after Schwichtenberg (2020 §2.1, §4.4). With its
default Monte Carlo dynamics, it SHALL sample the Wilson action of U(1) or Z_N link angles on the cell complex of a
periodic lattice under discrete time, through `WilsonSamplingInteraction`, a subclass of `SamplingInteraction` in its
package, and its catalog entry SHALL declare the mean plaquette as an observable.

#### Scenario: Plaquette against the exact result
- **WHEN** `LatticeGaugeFieldsSystem` runs with its defaults, U(1) on a periodic 32 × 32 lattice at β = 2, after its
  transient
- **THEN** the mean plaquette observable is within 0.01 of `I_1(2) / I_0(2) ≈ 0.698`

#### Scenario: Gauge-invariant run
- **WHEN** a random gauge transformation is applied to the initial links before a Monte Carlo run with a fixed seed
- **THEN** the mean plaquette series equals the series of the untransformed run within 1e-12

### Requirement: Lattice gauge system evolves Maxwell fields
With `dynamics='hamiltonian'`, `LatticeGaugeFieldsSystem` SHALL return a system under continuous time that evolves a lattice
Maxwell potential and electric field on links from optional static charges on sites. Its catalog entry SHALL declare
the Gauss residual and the field energy as observables.

#### Scenario: Maxwell dynamics conserve the Gauss law
- **WHEN** `LatticeGaugeFieldsSystem` runs with Hamiltonian dynamics and a static charge pair on a periodic 32 × 32 lattice
- **THEN** the Gauss residual stays below 1e-10 at every frame, and the field energy changes by less than 1e-3
  relative error

### Requirement: Topological Kuramoto system
The library SHALL provide `TopologicalKuramotoSystem` in `physics/synchronization`, after Millán, Torres, and Bianconi
(2203.06601 III): phases on the nodes and links of a clique complex, the link phases driven through the codifferential
and the exterior derivative of the sines of the phases, with node and link dynamics coupled by their global order
parameters.

#### Scenario: Link synchronization grows with coupling
- **WHEN** `TopologicalKuramotoSystem` runs on the clique complex of a random graph at zero coupling and at a coupling well
  above its transition
- **THEN** the order parameter of the link phases at the end of the strong-coupling run exceeds that of the run at zero
  coupling by at least 0.3

### Requirement: Gauge arbitrage market system
The library SHALL provide `GaugeArbitrageMarketSystem` in `economics/finance`, after Schwichtenberg (2020 §1.7, §2.1,
§2.2.1): exchange-rate potentials on the links of a lattice complex evolved by the discrete inhomogeneous Maxwell
equations, sourced by the trader current on links, and the good on sites moved along the links by its covariant
gradient.

#### Scenario: Conservation in the market
- **WHEN** `GaugeArbitrageMarketSystem` runs with its defaults
- **THEN** the total good stays constant within 1e-10 relative error, and the Gauss residual against the density of the
  good stays below 1e-8

### Requirement: Gauge network agents
The library SHALL provide `GaugeNetworkAgentsSystem` in `social/opinion_dynamics`, after Mihaylov and Cacciatori (2025,
arXiv 2509.01581): agent phases on the nodes of a network and connections on its edges, the agents relaxing towards
their neighbours' transported phases through `TransportedCouplingInteraction`, a subclass of `CouplingInteraction` in
its package, so frustrated loops prevent global agreement.

#### Scenario: Frustration blocks agreement
- **WHEN** `GaugeNetworkAgentsSystem` runs on a ring of 12 agents with a flat connection and with a total flux of pi around
  the ring
- **THEN** the flat ring reaches agreement with zero energy, and the frustrated ring keeps a residual energy and a twist

### Requirement: Link field systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL be marked implemented in
the system inventory.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `LatticeGaugeFieldsSystem` in `physics/waves`
  - `TopologicalKuramotoSystem` in `physics/synchronization`
  - `GaugeArbitrageMarketSystem` in `economics/finance`
  - `GaugeNetworkAgentsSystem` in `social/opinion_dynamics`

### Requirement: Elementary flux modes system
The library SHALL provide `ElementaryFluxModesSystem`, which enumerates the elementary flux modes and extreme pathways of a
metabolic network (Raman 2021 §9.8), Raman's toy network of Table 8.2 by default, ordered by growth yield on the
substrate, through `elementary_modes`, without any extra.

#### Scenario: Toy network modes and the FBA optimum
- **WHEN** `ElementaryFluxModesSystem` runs with its defaults
- **THEN** it finds 2 modes, each with growth yield `1/12` on `A` and `v6 = 20 · vbio`, and the FBA optimum of growth
  50 is their non-negative combination within 1e-9, so its maximum is set by the bound 1000 on `v6`

#### Scenario: Yield of a knock-out
- **WHEN** `ElementaryFluxModesSystem` runs with `v1` knocked out
- **THEN** it finds 1 mode, through `v2` and `v3`, with the same growth yield `1/12`

### Requirement: Carbon-13 metabolic flux analysis system
The library SHALL provide `Carbon13MFASystem`, which estimates the fluxes of a network with atom transitions from
labelling data of parallel tracer experiments (Raman 2021 §9.7), with confidence intervals and a goodness-of-fit test.
By default it SHALL use the TCA example network of Antoniewicz et al. (2007) with data simulated from known fluxes plus
seeded noise, through `labelling`, and it SHALL accept user networks and measurements. It SHALL run without
any extra.

#### Scenario: Recovered fluxes
- **WHEN** `Carbon13MFASystem` runs with its defaults, noise of standard deviation 0.002, and seed 0
- **THEN** the estimated split of α-ketoglutarate, `v3` and `v4`, is within 2 of the true 50 and 50, every net flux's
  95% interval contains its true value, and the chi-square test accepts the fit at the 5% level

#### Scenario: Published labelling pattern
- **WHEN** `Carbon13MFASystem` simulates glutamate with the true fluxes and the tracer of Antoniewicz et al. (2007)
- **THEN** the distribution is `[0.3464, 0.2695, 0.2708, 0.0807, 0.0286, 0.0039]` within 5e-5

### Requirement: Constraint-based systems have gallery examples
Each system of this change SHALL have a gallery example at `docs/examples/biology/<subfield>/plot_<name>.py` that runs
within the gallery budget. It SHALL show its Rerun primary view as the thumbnail and at least one analysis plot from
its own run, with a caption stating the behaviour shown.

#### Scenario: Dynamic FBA example
- **WHEN** a user opens the `dynamic_fba` example page
- **THEN** its thumbnail is the Rerun series of biomass and glucose. Below it, a plot of the logarithm of biomass
  against time shows a fitted slope of 0.874 h⁻¹ and the plateau after glucose depletion.

#### Scenario: Carbon-13 flux analysis example
- **WHEN** a user opens the `Carbon13MFASystem` example page
- **THEN** its thumbnail is the Rerun graph of the network with the estimated fluxes. Below it, a plot shows the
  estimated net fluxes with their 95% intervals against the true fluxes, and a plot shows the measured and simulated
  glutamate distributions.

#### Scenario: Coverage of the new systems
- **WHEN** the gallery coverage check runs after this change
- **THEN** each of the nine systems has exactly one example, and the check names any that does not

### Requirement: Constraint-based systems are recorded in the system inventory
Every system this change adds SHALL register under the field path of its subfield and SHALL have an entry in the system
inventory tagged with the system it builds, and a system the inventory records as pending this change SHALL register
under the name of its entry.

#### Scenario: Inventory systems registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `flux_balance` in `biology/systems_biology`
  - `metabolic_perturbations` in `biology/systems_biology`
  - `regulatory_fba` in `biology/systems_biology`
  - `dynamic_fba` in `biology/systems_biology`
  - `dynamic_multispecies_fba` in `biology/ecology`
  - `comets` in `biology/ecology`
  - `community_fba` in `biology/ecology`
  - `ElementaryFluxModesSystem` in `biology/systems_biology`
  - `Carbon13MFASystem` in `biology/systems_biology`

#### Scenario: Systems that need the extra
- **WHEN** a user without the extra `metabolism` creates a system that needs it
- **THEN** a named error names the system and the extra, and `flux_balance` with its default toy network still runs

#### Scenario: Systems blocked on this change
- **WHEN** the inventory check runs after this change
- **THEN** `ElementaryFluxModesSystem` and `Carbon13MFASystem`, which the inventory recorded as blocked on this change, are
  recorded as implemented with the sources `Rm 9.8` and `Rm 9.7`

### Requirement: Systems on radius reductions, pair couplings, and surrogates
The library SHALL provide `SzaboCellsSystem` and `ActivePassiveFlockingSystem` in `physics/active_matter`,
`KoopmanModelPredictiveControlSystem` in `engineering/control`, and `NextGenerationReservoirSystem` in
`computer_science/neural_networks` as system classes built from the general classes of the core, each with a
curated gallery example and an inventory entry tagged with the system it builds.

#### Scenario: Cells order collectively
- **WHEN** `SzaboCellsSystem` runs at a density above its ordering transition
- **THEN** the mean normalized velocity of the cells grows to an order parameter above 0.8

#### Scenario: Active agents flock
- **WHEN** `ActivePassiveFlockingSystem` runs with every agent active and flocking on
- **THEN** the polarization of the velocities rises above 0.9

#### Scenario: Active agents gather
- **WHEN** `ActivePassiveFlockingSystem` runs with the settings of Charbonneau (2017, figure 10.7), 45 active agents among
  342 at the compactness 0.672 from uniform random positions, with flocking on, for 50 time units
- **THEN** the mean distance between active agents over the last 10 time units, as a mean over four seeds, falls below the
  mean distance between passive agents by at least 0.015, and the snapshot shows flocks of active agents among still passive agents

#### Scenario: Active agents stay mixed without flocking
- **WHEN** `ActivePassiveFlockingSystem` runs with the same settings and flocking off
- **THEN** the two mean distances over the last 10 time units, as a mean over four seeds, stay within 0.01 of each other

#### Scenario: Surrogate predictive control
- **WHEN** `KoopmanModelPredictiveControlSystem` steers each of its plants to a reference with its fitted surrogate
- **THEN** the controlled state tracks the reference with a smaller mean squared error than the uncontrolled plant

#### Scenario: Reservoir forecast
- **WHEN** `NextGenerationReservoirSystem` is fitted to a training window of the Lorenz flow
- **THEN** its forecast stays within 10% of the attractor size of the true trajectory for at least one Lyapunov time

### Requirement: The next-generation reservoir reuses the analysis readout
`NextGenerationReservoirSystem` SHALL take its readout from the result of `reservoir(runs, kind='next_generation')` and SHALL
NOT fit a readout of its own.

#### Scenario: Agreement with the reservoir analysis
- **WHEN** `NextGenerationReservoirSystem` is built from the readout of a `reservoir(runs, kind='next_generation')` result
  fitted to a series, and both forecast from the end of that series
- **THEN** the system's forecast over one Lyapunov time agrees with the analysis's `forecast` within 1e-6 of the
  attractor size

### Requirement: Multiscale network systems
The library SHALL provide `PopularitySimilarityNetworkSystem` in `mathematics/networks` and `UrbanScalingInteractionsSystem` in
`social/organizations`, both grown by `TopologyInteraction` with attachment weights of the hyperbolic distance, each
with a curated gallery example and an inventory entry tagged with the system it builds.

#### Scenario: Hyperbolic degrees
- **WHEN** `PopularitySimilarityNetworkSystem` grows 2000 nodes with degree exponent 2.5, two edges, and temperature 0.2
- **THEN** the fitted degree exponent, as a mean over four seeds, is within 0.07 of the measured 2.69, against the 2.5 of
  the source, which an independent implementation of the published rule also reads high (2.71 over 16 seeds), and the
  clustering stays above 0.3

#### Scenario: Superlinear interactions
- **WHEN** `UrbanScalingInteractionsSystem` is swept over city size and its interactions are fitted by
  `scaling(..., kind='allometric')`
- **THEN** the exponent of the number of interactions is above 1, as Ribeiro and Netto (2024) review

### Requirement: Lattice Yang-Mills model samples SU(2) and SU(3) links
The library SHALL provide `LatticeYangMillsSystem` in `physics/waves`, after Satz (2022 §7) and Schwichtenberg (2020 §4.4).
It SHALL sample SU(2) or SU(3) link matrices on the cell complex of a periodic lattice by the Wilson action
`-(beta / N) sum_p Re tr U_p` under discrete time, through `WilsonSamplingInteraction` with a matrix group, from Haar
links or given links, with an extent of the last axis for finite temperature. Its catalog entry SHALL declare the mean
plaquette and the Polyakov loop as observables, and `gauge_observables` SHALL measure Wilson loops of given extents and the
Polyakov loop of matrix links.

#### Scenario: Four-dimensional SU(2)
- **WHEN** `LatticeYangMillsSystem` samples SU(2) at `beta = 2.3` on a periodic 4^4 lattice for 100 sweeps
- **THEN** the mean plaquette after 30 sweeps is within 0.03 of 0.60

#### Scenario: Deconfinement
- **WHEN** `LatticeYangMillsSystem` samples SU(2) on a 4 × 4 × 4 × 2 lattice at `beta = 1.5` and at `beta = 2.6`
- **THEN** the mean Polyakov loop at `beta = 2.6` is more than three times that at `beta = 1.5`

#### Scenario: Gauge-invariant loops of matrix links
- **WHEN** a random gauge transformation is applied to random SU(2) or SU(3) links
- **THEN** the 2 × 1 Wilson loop and the Polyakov loop are unchanged within 1e-12

### Requirement: Lattice Yang-Mills is recorded in the system inventory
The inventory SHALL record `LatticeYangMillsSystem` as implemented with its sources, the primitives `link-fields` and
`matrix-groups`, and the matrix domain, and the gallery SHALL hold its curated example.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for `LatticeYangMillsSystem`

### Requirement: Hydrodynamically coupled particles in blinking traps

The catalog SHALL include `HydrodynamicallyCoupledTrapsSystem`, two Brownian beads in optical traps that switch on and off
periodically, coupled by the Rotne-Prager-Yamakawa or Oseen mobility, after Argun et al. 2021, Problems 7.4 and 7.5.

#### Scenario: Free motion is correlated

- **WHEN** the traps are off
- **THEN** the displacements of the two beads along their line of centres are positively correlated as the mobility
  predicts, and the correlation vanishes without the coupling

### Requirement: Hydrodynamic systems are recorded in the system inventory

The system inventory SHALL list `squirmers` and `HydrodynamicallyCoupledTrapsSystem` as implemented, with their sources and
the `hydrodynamic-interactions` primitive, and the gallery SHALL hold an example of each.

#### Scenario: Inventory and gallery checks pass

- **WHEN** the inventory check and the gallery coverage check run
- **THEN** both systems are found implemented, with gallery examples

### Requirement: Multilevel selection with pairwise group conflict
The library SHALL provide the density of group compositions of defectors and altruistic punishers under within-group
replicator dynamics and pairwise between-group conflict with a victory probability that is not additively separable,
the group Fermi rule, the local rule normalized by the payoffs of both groups, or the Tullock contest function, as
`NonlinearPairwiseGroupConflictSystem` under `social/cooperation` (Cooney 2024, arXiv 2405.18419 §3.1 and §6). The
inventory SHALL record it as implemented.

#### Scenario: Strong between-group conflict sustains punishers
- **WHEN** the pairwise conflict with the group Fermi rule runs from a uniform density with `lambda = 2` and with
  `lambda = 0.1`
- **THEN** the mean fraction of punishers ends higher with `lambda = 2` than with `lambda = 0.1`, and the total stays
  one

### Requirement: Trimorphic multilevel selection on the simplex
The library SHALL provide the density on the 2-simplex of cooperators, defectors, and altruistic punishers with the
victory probability of the fraction of non-defectors or of the normalized payoff difference, as
`TrimorphicMultilevelSelectionSystem` under `social/cooperation` (arXiv 2405.18419 §3.2 and §7). The inventory SHALL record
it as implemented.

#### Scenario: Trimorphic selection lowers defection
- **WHEN** the trimorphic model with the victory probability of non-defectors runs from a uniform density on the
  simplex
- **THEN** the mean fraction of defectors decreases, and the total stays one

### Requirement: Size-structured forest with shading and dispersal
The library SHALL provide the McKendrick-von Foerster density of trees over position and height, with growth slowed by
Beer-Lambert shading from taller trees, mortality from canopy and resource competition, and recruitment at the
sapling height from seeds dispersed over a strip with uncolonized land beyond its edges, as `ForestSizeStructureSystem`
under `biology/ecology` (Grilletta et al. 2026, arXiv 2608.11918 Eq. 9, 10, and 16). The inventory SHALL record it as
implemented.

#### Scenario: Edges deplete small trees
- **WHEN** the forest runs on a strip towards its stationary state
- **THEN** the density of trees is lower at the edges than in the bulk, and smaller trees are depleted at the edges by a
  larger fraction than taller trees

### Requirement: Cellular Potts model sorts cells by differential adhesion
The library SHALL provide `CellularPottsSystem` in `biology/physiology`, after Nemati and de Graaf (2024, arXiv 2404.09055,
Eq. 3) and Graner and Glazier (1992). It SHALL sample a confluent tissue of cells, regions of sites with one label,
whose kinds have a symmetric adhesion matrix, with an area constraint and an optional perimeter constraint, by label
copies under discrete time on a periodic square lattice with the Moore neighbourhood, a triangular lattice, or a
disordered lattice, the periodic Voronoi tessellation of jittered points with contacts weighted by side length over the
mean and sites of their Voronoi area. Its catalog entry SHALL declare the share of homotypic contacts and the mean
relative deviation of the areas from the target as observables.

#### Scenario: Area constraint holds
- **WHEN** the tissue runs 200 sweeps on any of the three lattices
- **THEN** no cell vanishes and the mean relative deviation of the areas after 50 sweeps is below 0.1

#### Scenario: Differential adhesion sorts the kinds
- **WHEN** the tissue runs 1500 sweeps on the square or the disordered lattice with the default adhesion
- **THEN** the share of homotypic contacts rises by more than 0.1, while equal adhesions keep it within 0.1

### Requirement: Interacting urns model on a network
The library SHALL provide `InteractingUrnsNetworkSystem` in `economics/innovation`, after Iacopini et al. (2020, Eqs. 1 and
2): an urn with triggering on every node of a network, with reinforcement `rho` and triggering `nu`, that draws from
its own balls and one ball of every colour of the urns of its neighbours, the balls being agents of a variable group
on the nodes.

#### Scenario: Isolated urns
- **WHEN** 30 isolated urns with `rho = 4` and `nu = 2` draw for 200 steps
- **THEN** their mean number of novelties is within 5 percent of Eq. (4) without neighbours

#### Scenario: Star network
- **WHEN** the urns of a star of seven nodes with `rho = 6` and `nu = 3` draw for 150 steps
- **THEN** the centre discovers more than every leaf, as Eq. (4) predicts

### Requirement: TAP model with catalysis
The library SHALL provide `TapCatalysisSystem` in `economics/innovation`, after Hordijk and Kauffman (2022, Algorithm 1):
goods made from up to `K` parents at the rates `alpha^i C(M, i)` until `M` goods, production functions as hyperedges
of a ranked group, and catalysis edges both ways with probability `pc`.

#### Scenario: Production functions
- **WHEN** the TAP model grows to 200 goods with three parents at most
- **THEN** every new good has one production function of two to four members, whose parents are older goods

#### Scenario: Emergence of autocatalytic sets
- **WHEN** four runs grow to 300 goods at `pc = 0.004` and four at `pc = 0.04`
- **THEN** no run of the first holds a RAF of half its production functions, and every run of the second a RAF of
  more than 90 percent of them

### Requirement: Motter-Lai overload cascades
The library SHALL provide `MotterLaiOverloadSystem` in `mathematics/networks`, after Motter and Lai (2002) as Artime et al.
(2025, section 5) review it: nodes with the capacity `(1 + tolerance)` times their initial betweenness, one node removed
by load, degree, or at random, and every node whose betweenness on the network present exceeds its capacity removed at
every later step.

#### Scenario: Attack and failure
- **WHEN** the node of the largest load and a random node of Barabási-Albert graphs of 300 nodes are removed at the
  tolerance 0.1
- **THEN** the attack leaves less than a third of the nodes in the largest component, and the failure more than 90
  percent

#### Scenario: Overloaded nodes fail
- **WHEN** the cascade runs at tolerance 0.2
- **THEN** the nodes that fail at every step are exactly those whose betweenness on the network before the step
  exceeds their capacity

### Requirement: New systems are recorded in the system inventory
The inventory SHALL record `InteractingUrnsNetworkSystem`, `TapCatalysisSystem`, and `MotterLaiOverloadSystem` as implemented with
their sources and the primitives `member-sets`, `hyperedge-requests`, and `path-reductions`, and the gallery SHALL hold
their examples.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for the three systems

### Requirement: Audited maps and flows of dynamical systems
The library SHALL provide in `mathematics/dynamical_systems` the Fermi-Ulam map `FermiUlamMapSystem` (Datseris and Parlitz
Ex 1.11), the Nosé-Hoover oscillator `NoseHooverSystem` (Ex 2.11), Sprott's mixed flow `SprottMixedFlowSystem` (Ex 3.17), and,
after Dilão, the CDN map `CDNMapSystem`, the Lozi map `LoziMapSystem`, the conservative Hénon map `ConservativeHenonMapSystem`, the
Langford and Thomas flows `langford` and `thomas`, the hyperchaotic Rössler flow `HyperchaoticRosslerSystem`, and the Ikeda
delay equation `ikeda`.

#### Scenario: Hyperchaos
- **WHEN** the Lyapunov spectrum of `HyperchaoticRosslerSystem` is computed over 2000 time units
- **THEN** its two largest exponents are near 0.11 and 0.02

#### Scenario: Mixed state space
- **WHEN** `SprottMixedFlowSystem` runs from `(1, 0, 0)` and from `(2, 0, 0)`
- **THEN** the mean volume rate `2(y + z)` is near 0 on the first orbit and below -0.1 on the second

#### Scenario: Delay bifurcation
- **WHEN** `ikeda` runs with `a tau` below and above `π / 2`
- **THEN** it settles at `π` in the first case and oscillates about it in the second

### Requirement: Audited stochastic processes and fractals
The library SHALL provide `CascadingSampleSpaceReducingSystem` (Thurner et al. 3.3.5.4), `RandomBranchMapSystem` (Dilão Ex
5.10), and `ExponentialGrowthAtRandomTimesSystem` (Thurner et al. 3.3.6.1) in `mathematics/stochastic_processes`, and
`InverseJuliaSystem` (Dilão Ex 5.8) in `mathematics/fractals`.

#### Scenario: Any exponent
- **WHEN** cascades that multiply by `mu` run on 100 states
- **THEN** their visits fall as `i^(-mu)`

#### Scenario: Power law from two exponentials
- **WHEN** growth at the rate `mu` is observed at the rate `lam`
- **THEN** the Hill estimate of the tail of the values is `lam / mu`

#### Scenario: Julia set of the squaring map
- **WHEN** `InverseJuliaSystem` runs with `omega = 0`
- **THEN** every point lies on the unit circle

### Requirement: Audited network systems
The library SHALL provide `GraphletArrivalSystem` (Raman 3.2.4.2), `KroneckerGraphSystem` (Gros, chapter 1),
`configuration_model` (Jensen 8.5.1 and Thurner et al. 4.5.1), and `InterdependentNetworkCascadeSystem` (Artime et al.
section 5) in `mathematics/networks`, and `BarabasiAlbertSystem(fitness=True)` SHALL attach new edges in proportion to
`eta k` with a uniform fitness `eta` per node (Raman 3.2.4).

#### Scenario: Graphlet degree shares
- **WHEN** graphlets arrive for 1000 steps
- **THEN** the network is a tree whose shares of degree 1 and 2 are `2 (2 - alpha) / 5` and `alpha (7 - alpha) / 15`

#### Scenario: Interdependent collapse
- **WHEN** two Erdős-Rényi networks of mean degree 4 keep a share `p` of their nodes
- **THEN** the cascade stops at `S = p (1 - exp(-4 S))²` for `p = 0.85` and removes every node for `p = 0.55`

#### Scenario: Degree sequence kept
- **WHEN** the configuration model pairs the half-links of a degree sequence
- **THEN** the graph is simple and every node has its degree

### Requirement: Audited physics systems
The library SHALL provide, as system classes built from the general classes of the core, the physics systems of
the reference audit that the core supports: `NuclearQuadrupoleSystem`, `ForcedTodaSystem`, and `ImpulsiveOscillatorSystem` in
`physics/nonlinear_oscillators`, `BeamBeamMapSystem` and `FrictionOscillatorSystem` in `physics/mechanics`,
`KeplerianDumbbellSystem` and `RelativisticOrbitSystem` in `physics/celestial_mechanics`, `ForestDiebackSystem`,
`SocialForestDiebackSystem`, `LeadingFollowingCascadeSystem`, and `PopulationPollutionSystem` in `physics/climate`,
`RosslerDrivenLorenzSystem` and `AdaptiveKuramotoSystem` in `physics/synchronization`, `FixedEnergySandpileSystem`,
`ZhangSandpileSystem`, and `SelfOrganizedBranchingSystem` in `physics/criticality`, `HPPLatticeGasSystem` in `physics/fluids`,
`InertialActiveBrownianParticlesSystem`, `ActivePolymerSystem`, and `DeformableActiveParticlesSystem` in `physics/active_matter`,
and `AnisotropicBrownianParticlesSystem` and `RigidDimersSystem` in `physics/soft_matter`. Every one SHALL have tests of a known
result of its source, a gallery example, and an inventory entry with its sources and sections.

#### Scenario: Conserved quantities
- **WHEN** `NuclearQuadrupoleSystem`, `KeplerianDumbbellSystem`, or `FixedEnergySandpileSystem` runs with its defaults
- **THEN** the quadrupole energy, the dumbbell energy and angular momentum, or the number of grains stay constant
  within the tolerance of the solver

#### Scenario: Published fixed points and orbits
- **WHEN** `ImpulsiveOscillatorSystem`, `RelativisticOrbitSystem`, `DeformableActiveParticlesSystem`, or `ForestDiebackSystem` runs from
  the states of its analytic solution
- **THEN** it follows the exact periodic orbit, the precession period, the straight or circular motion, or the root of
  its balance equation that the source derives

#### Scenario: Self-organization and synchronization
- **WHEN** `SelfOrganizedBranchingSystem`, `RosslerDrivenLorenzSystem`, or `AdaptiveKuramotoSystem` runs long enough
- **THEN** the branching probability settles at 1/2, the two driven Lorenz copies converge for a coupling above the
  critical value and not below it, and identical oscillators split into frequency clusters

#### Scenario: Statistical laws of particles
- **WHEN** `InertialActiveBrownianParticlesSystem`, `ActivePolymerSystem`, `AnisotropicBrownianParticlesSystem`, or `RigidDimersSystem`
  runs
- **THEN** the projection of the velocity on the orientation, the motion of the centre of mass, the decorrelation of
  the axes and the body-frame diffusion, and the diffusion of the centre and the rod match their analytic forms

### Requirement: Variants as options of existing systems
`billiards` SHALL take the shape `'three_disk'`, the open billiard of three discs, `ParametricPendulumSystem` SHALL take the
direction `'horizontal'` of the pivot, `AggregateAveragingSynchronizationSystem` SHALL take the Bernoulli map and a delay
of the coupling, `TopologicalKuramotoSystem` SHALL take the model `'NLT'`, and `vicsek` SHALL take a number of nearest
neighbours, a vision cone, and a delay of the alignment, negative for a linear prediction. The defaults SHALL keep the
earlier behaviour of every system.

#### Scenario: Chaotic scattering
- **WHEN** a particle comes towards the centre of the three-disk table along the line through the centre of a disc
- **THEN** it bounces once and goes straight back, and other directions give several bounces

#### Scenario: Delayed synchronization
- **WHEN** two Bernoulli maps of slope 1.5 are coupled with a delay
- **THEN** they synchronize exactly for the couplings whose linear transverse map is stable, and not outside them

#### Scenario: Nodes-Links-Triangles
- **WHEN** `TopologicalKuramotoSystem` runs at a weak coupling with the models `'NL'` and `'NLT'`
- **THEN** the node phases synchronize in the first and stay incoherent in the second

#### Scenario: Vicsek variants
- **WHEN** `vicsek` aligns with the 8 nearest neighbours rather than the nearest one, or with a delay of 25 steps or
  a prediction 2 steps ahead
- **THEN** the flock orders more with more neighbours, and the long and negative delays leave fewer neighbours around
  every particle than no delay

#### Scenario: Variants that do not combine
- **WHEN** `vicsek` gets both a number of nearest neighbours and a vision cone
- **THEN** a parameter error names the two options

### Requirement: Fishery with measurement errors
The library SHALL provide `FisheryMeasurementErrorsSystem` in `economics/resources`, after Argun et al. (2021, Problem
14.8): independent logistic fish stocks with a noisy growth rate, harvested by the quota `x~ - K / 2` of a measurement
with a Gaussian error, floored at zero unless restocking is allowed, as an Euler-Maruyama map.

#### Scenario: Deterministic quota
- **WHEN** the stocks run without noise
- **THEN** they settle at `K / sqrt(2)` with the quota `K / sqrt(2) - K / 2`

#### Scenario: Extinction
- **WHEN** the growth is noisy and the measurement error has a standard deviation of 0.3
- **THEN** more than 30 percent of the stocks go extinct, and none with restocking

### Requirement: Rice food balance
The library SHALL provide `RiceFoodBalanceSystem` in `economics/resources`, the food security model of Bala et al. (2017,
Section 4.11.3), with its equations and defaults.

#### Scenario: Food security peaks
- **WHEN** the model runs for 100 years
- **THEN** the area approaches `L / (1 + T f)`, and the food security peaks between 10 and 35 years and ends below its
  start

### Requirement: Crop livestock household
The library SHALL provide `CropLivestockHouseholdSystem` in `economics/resources`, the crop livestock model of Bala et al.
(2017, Section 4.11.6), as a yearly map with truncated purchases and sales and a pipeline delay of three years.

#### Scenario: Maturation delay
- **WHEN** the household runs for 12 years
- **THEN** the animal bought in year 1 joins the herd in year 5, and the livestock income starts after five years

### Requirement: Brock-Mirman growth
The library SHALL provide `BrockMirmanGrowthSystem` in `economics/macroeconomics`, after Caiani et al. (2016, Section
1.1.4), the capital map `K' = alpha beta z K^alpha` with consumption `(1 - alpha beta) Y` and Markov productivity.

#### Scenario: Steady state and stationary mean
- **WHEN** the productivity is constant, or it switches between 0.9 and 1.1 for 4000 periods
- **THEN** the capital reaches `(alpha beta)^(1 / (1 - alpha))`, or the mean of `ln K` is
  `(ln(alpha beta) + E ln z) / (1 - alpha)` within 0.02

### Requirement: Network herding
The library SHALL provide `NetworkHerdingSystem` in `economics/macroeconomics`, the choices `c_t = Theta c_t-1 + gamma x_t`
of Caiani et al. (2016, Section 1.1.6) on a weighted network.

#### Scenario: Aggregation through the out-degrees
- **WHEN** the choices run without shocks
- **THEN** the mean choice follows Eq. 1.12 at every step and the choices reach `(I - Theta)^-1 gamma x`

### Requirement: Categorized purchasing power
The library SHALL provide `CategorizedPurchasingPowerSystem` in `economics/macroeconomics`, the three-sector system
`z' = A z + B x`, `y = C z + D x` of Forrest (2018, Sections 11.4 and 22), with an option for the feedback component
that decouples the sectors.

#### Scenario: Decoupling
- **WHEN** a step policy hits the luxury sector
- **THEN** with feedback only the luxury index moves, to the steady state of its decoupled system, and without feedback
  the other indices move too

### Requirement: Kirman exchange rate
The library SHALL provide `KirmanExchangeRateSystem` in `economics/finance`, Eq. 1.36 of Caiani et al. (2016, Section
1.3.2) with the share of fundamentalists of Kirman's recruitment chain.

#### Scenario: Regimes
- **WHEN** the market runs for 40000 steps
- **THEN** the share of fundamentalists is mostly near 0 or 1, and the rate changes more when chartists dominate

### Requirement: Default cascades
The library SHALL provide `DefaultCascadeSystem` in `economics/finance`, the threshold cascade of Thurner et al. (2018,
Section 5.5.3.2) with a recovery rate, and the random network of Gai and Kapadia reviewed by Bardoscia et al. (2021,
Section 3.1).

#### Scenario: Contagion window
- **WHEN** one bank of networks of mean degree 0.5, 4, and 12 is shocked
- **THEN** cascades are large only at the intermediate degree

### Requirement: Eisenberg-Noe clearing
The library SHALL provide `EisenbergNoeClearingSystem` in `economics/finance`, the iteration of proportional payments to
the greatest clearing vector.

#### Scenario: Linear programme
- **WHEN** the clearing runs to convergence
- **THEN** the payments solve the linear programme of the clearing vector

### Requirement: Fire sales of overlapping portfolios
The library SHALL provide `OverlappingPortfolioFireSalesSystem` in `economics/finance`, banks with common assets that
sell under a threshold or leverage targeting, with market impact, after Bardoscia et al. (2021, Section 3.2).

#### Scenario: Indirect contagion
- **WHEN** one asset loses 60 percent of its price
- **THEN** only its holders default without market impact, and more than twice as many with it

### Requirement: Guru order book
The library SHALL provide `GuruOrderBookSystem` in `economics/finance`, the order-driven market of noise traders who follow
gurus on a network rewired towards wealth, of Caiani et al. (2016, Chapter 3).

#### Scenario: Coordination and fat tails
- **WHEN** the market runs for 1000 days
- **THEN** cash and stocks are conserved, a guru gathers more than 40 followers, and the returns have an excess kurtosis
  above 3

### Requirement: Household demand
The library SHALL provide `HouseholdDemandSystem` in `economics/markets`, the Cobb-Douglas and the elasticity demand of
Hamill and Gilbert (2016, Sections 3.2 and 3.3) for households with unequal budgets.

#### Scenario: Price rise
- **WHEN** the price of food rises by 10 percent
- **THEN** Cobb-Douglas households keep their expenditure, and households with elasticities lower their demand by 3.5
  to 5 percent and raise their expenditure by 4.5 to 6 percent

### Requirement: Fair Food Program
The library SHALL provide `FairFoodProgramSystem` in `economics/markets`, the four rules of Chesney (2021, Chapter 7).

#### Scenario: Tipping point
- **WHEN** the programme runs for 5000 periods
- **THEN** farms join only after most buyers commit, and then all farms and buyers join

### Requirement: Gooding double auction
The library SHALL provide `GoodingDoubleAuctionSystem` in `economics/markets`, buyers and sellers who adapt their quotes,
after Gooding (2019, Section 11.3).

#### Scenario: Competitive price
- **WHEN** 100 buyers and 100 sellers of five units trade
- **THEN** the 500 units trade at a mean price within 5 of 70

### Requirement: Toy Trader
The library SHALL provide `ToyTraderSystem` in `economics/markets`, Gooding's Toy Trader with market sight, mutation, and
imitation under a fitness test, and the Three-Goods Trader as `n_goods=3`.

#### Scenario: Market sight
- **WHEN** traders see one unit or the whole torus for 300 iterations
- **THEN** the price falls below 9 or rises above 11, and money and toys are conserved

### Requirement: El Farol coalitions
The library SHALL provide `ElFarolCoalitionsSystem` in `economics/econophysics`, Collins' strategic group formation in the
El Farol bar of Rand and Wilensky.

#### Scenario: Coalitions lower returns
- **WHEN** patrons run alone and in coalitions for 500 nights
- **THEN** lone patrons attend near capacity and coalitions lower the returns by more than 0.1

### Requirement: CCC model
The library SHALL provide `CccModelSystem` in `economics/innovation`, the co-evolving combinatorial critical model of
Thurner et al. (2018, Section 5.6.4).

#### Scenario: Regimes
- **WHEN** the model runs with intermediate, mostly productive, and mostly destructive tensors
- **THEN** the first is punctuated, the second fills, and the third stays nearly empty

### Requirement: Audited cooperation systems
The library SHALL provide in `social/cooperation` Argun's evolutionary iterated prisoner's dilemma on a lattice
`IteratedLatticeDilemmaSystem` (Argun et al. 13.2 and 13.3), Cooney's two-level replicator equation `TwoLevelReplicatorSystem`
with the additively separable pairwise group conflict as its option `victory` (arXiv 2405.18419 Eq. 3.3 and section 5),
the finite two-level birth-death process `TwoLevelBirthDeathSystem` (section 3.1 and appendix A.1), and evolutionary
dynamics with game transitions `GameTransitionsSystem` (arXiv 1905.10269).

#### Scenario: Best reply in the iterated dilemma
- **WHEN** the years in prison of every strategy `n` against a strategy `m` of a ten-round game are computed
- **THEN** the fewest years come from `n = m - 1`

#### Scenario: Three regimes of two strategies
- **WHEN** `IteratedLatticeDilemmaSystem` with only the extreme strategies runs at `R = 0.82`, `0.84`, and `0.86`
- **THEN** cooperation dominates, coexists with defection, and vanishes

#### Scenario: Threshold of group selection
- **WHEN** `TwoLevelReplicatorSystem` with `G(z) = z` runs from a uniform density above and below `lambda* = c + q - p`
- **THEN** the mean fraction of punishers tends to `1 - (c + q - p) / lambda` above it and to 0 below it

#### Scenario: Transitions favour cooperation
- **WHEN** `GameTransitionsSystem` runs under pairwise comparison with `b1 - b2 > 2 c`, with and without transitions
- **THEN** cooperators take over with transitions and die out with game 1 alone

### Requirement: Audited opinion dynamics systems
The library SHALL provide in `social/opinion_dynamics` the three-state opinion network `ThreeStateOpinionSystem` and its
mean-field flows `UndecidedFlowsSystem` (Jensen 11.5, Project 1), the majority rule with willingness `WillingMajoritySystem`
(Jensen 11.5, Exercise 4), and the echo chamber model `EchoChamberRadicalizationSystem` (arXiv 1906.12325 Eqs. 1 and 2).

#### Scenario: Critical size of the undecided
- **WHEN** `UndecidedFlowsSystem` with flows driven by the undecided runs above and below `N_c = omega (1 / gamma + 1 / eta)`
- **THEN** `N - N_c` agents stay undecided above it and none below it

#### Scenario: Zealots
- **WHEN** `WillingMajoritySystem` runs with every agent willing, with zealots of both opinions, and with zealots of one
- **THEN** the first reaches consensus, the second does not, and the third ends at the zealots' opinion

#### Scenario: Echo chambers
- **WHEN** `EchoChamberRadicalizationSystem` runs at low controversy, without homophily, and with homophily
- **THEN** opinions reach the neutral consensus, radicalize to one side, and split into two opposite groups

### Requirement: Audited crowd, segregation, and language systems
The library SHALL provide Bellomo's kinetic crowd in a street with walls and panic `KineticCrowdSystem` in `social/crowds`
(Bellomo et al. 6.4 and 6.5), distinct from `evacuation`, the Big Sort as the option `choices` of `schelling` (Downey,
Exercise 9-1), and Miller's random typewriting `RandomTypewritingSystem` in `social/language` (Thurner et al. 3.3.6.2).

#### Scenario: Fundamental diagram
- **WHEN** `KineticCrowdSystem` runs with one cell over the street
- **THEN** its mean speed follows Eq. 6.30 of the density within ten percent

#### Scenario: Big Sort
- **WHEN** `schelling` runs with one, four, and ten choices
- **THEN** the like-neighbour share stays near its start with one choice and grows with the number of choices

#### Scenario: Zipf's law of random typing
- **WHEN** `RandomTypewritingSystem` types with five letters and a space bar of probability 0.2
- **THEN** words of one more letter are 6.25 times rarer and the rank-frequency exponent is near `1 - log(0.8) / log 5`

### Requirement: Audited organization systems
The library SHALL provide in `social/organizations` Chesney's Ajani's Choice `AjanisChoiceSystem` (Chesney 3.3 and 3.8) and
Namrata's Family `SupplyChainDiffusionSystem` (4.9 and 4.10), Meadows' watchmakers `watchmakers` (chapter 3), policy
resistance `PolicyResistanceSystem`, and drift to low performance `DriftToLowPerformanceSystem` (chapter 5), and the
exploration and preferential return `ExplorationPreferentialReturnSystem` and container model `ContainerMobilitySystem` of
human mobility (arXiv 2109.07381).

#### Scenario: Underpayment needs plentiful workers
- **WHEN** `AjanisChoiceSystem` runs with more workers than jobs and with fewer
- **THEN** workers are paid below the minimum wage only in the first case

#### Scenario: Power in supply chains
- **WHEN** a buyer with two suppliers and a buyer with one adopt good working conditions
- **THEN** the suppliers whose outflow is below the buyer's inflow adopt, and a single supplier never does

#### Scenario: Hierarchy of the watchmakers
- **WHEN** `watchmakers` runs for 20,000 actions with an interruption probability of 1%
- **THEN** Hora finishes a watch about every 1173 actions and Tempus none

#### Scenario: Eroding goals
- **WHEN** `DriftToLowPerformanceSystem` runs with eroding goals, absolute standards, and an upbeat bias
- **THEN** the performance drifts down, holds, and drifts up

#### Scenario: Scales of mobility
- **WHEN** `ContainerMobilitySystem` runs with the probabilities of travel 0.6, 0.3, and 0.1
- **THEN** the displacements happen at the three levels with these shares and grow with the level

### Requirement: Audited biology models
The library SHALL provide the biology models of the reference audit, listed in the proposal, as system
classes in the biology subfields built from the general classes of the core, each with a curated gallery example and
an inventory entry that cites its source section.

#### Scenario: Consumers coexist where their isoclines cross
- **WHEN** `ConsumerResourceSystem` runs with the settings of Mehta and Blumenthal (2024, figure 4)
- **THEN** the resources settle at `(5/3, 5/9)`, where both zero net growth isoclines cross

#### Scenario: Single-resource competition of Gros
- **WHEN** `ConsumerResourceSystem` runs with one supplied resource without dilution
- **THEN** the total population tends to `a / d`, the resource to `d / max r`, and only the fastest consumer survives

#### Scenario: Ricker period doubling
- **WHEN** `ricker` runs with `r = 5` and with `r = 9`
- **THEN** it settles on `ln r` in the first case and on a two-cycle in the second

#### Scenario: Doomsday of the zombie outbreak
- **WHEN** `ZombieOutbreakSystem` runs without births, deterministically or stochastically
- **THEN** the total is conserved and the humans vanish

#### Scenario: Record dynamics of outbreaks
- **WHEN** `RecordDynamicsEpidemicSystem` runs for 100 years over many seeds
- **THEN** the mean number of outbreak years is close to the harmonic number of the strains drawn

#### Scenario: Gamma rhythm of the neural mass
- **WHEN** `NextGenerationNeuralMassSystem` runs unforced with `H_e = 10`
- **THEN** the excitatory rate oscillates between 40 and 50 Hz

#### Scenario: Transient attractors of the dHAN
- **WHEN** `AssociativeThoughtProcessSystem` runs on the seven-site network of Gros (2015)
- **THEN** every lasting winning coalition is a stored clique and at least four cliques are visited

#### Scenario: Kleiber scaling of the distribution network
- **WHEN** `WestBrownEnquistSystem` is solved for growing depths
- **THEN** the blood volume and the capillaries follow the sums of the branching rules and the exponent rises towards 3/4

#### Scenario: Kimura fixation probability
- **WHEN** `KimuraDiffusionSystem` runs with many replicates until loss or fixation
- **THEN** the share of fixed replicates is `(1 - exp(-N s x)) / (1 - exp(-N s))`

#### Scenario: Critical noise of the Boolean network
- **WHEN** `NoisyBooleanNetworkSystem` runs with three inputs per node
- **THEN** it orders below the mean-field noise `1/6` and disorders above it

#### Scenario: Adaptation of the enzyme circuit
- **WHEN** `EnzymeAdaptationCircuitSystem` receives a step of its input
- **THEN** its output has a sensitivity above 1 and a precision above 10

#### Scenario: Ants heal a disruption
- **WHEN** `AntColonyOptimizationSystem` cuts its best path midway
- **THEN** the colony settles near the shortest path before the cut and again after it

#### Scenario: Shepherd brings the flock home
- **WHEN** `shepherding` runs from a scattered flock
- **THEN** the centre of the flock reaches the goal

### Requirement: Networks built from new topology requests
The library SHALL provide in `mathematics/networks` `ExplosivePercolationSystem` (Achlioptas, D'Souza, and Spencer 2009),
`DavidsenEbelBornholdtSystem` and `NodeMergingSystem` (Thurner et al. 4.5.4.6), `DuplicationDeletionSystem` (Jensen 8.5.1), and
`NetworkSuperpositionSystem` (Thurner et al. 4.5.2), whose linking probability is drawn once per Erdős-Rényi subgraph, the
reading that gives Eq. 4.34.

#### Scenario: Abrupt giant component
- **WHEN** `ExplosivePercolationSystem` adds edges by the product rule to 2000 nodes
- **THEN** its largest component jumps from below 10 percent to above half of the nodes between 0.8 and 0.95 edges per
  node, where Erdős-Rényi graphs already hold most nodes at 0.8

#### Scenario: Turnover sets the degree
- **WHEN** `DavidsenEbelBornholdtSystem` reaches its steady state with turnover `gamma`
- **THEN** its mean degree is within 15 percent of `1 + 1 / gamma` and its clustering is above 0.2

#### Scenario: Merging makes hubs
- **WHEN** `NodeMergingSystem` runs for 20 steps per node
- **THEN** its largest degree exceeds ten times the mean degree, far above the largest degree of an Erdős-Rényi graph
  of the same size and mean

#### Scenario: Steady connectance
- **WHEN** `DuplicationDeletionSystem` runs for 40 steps per node
- **THEN** its mean connectance is within 15 percent of Eq. 8.84

#### Scenario: Superposed degrees
- **WHEN** `NetworkSuperpositionSystem` draws power-law linking probabilities for 1000 subgraphs of 100 nodes
- **THEN** its degree distribution is within a total variation distance of 0.05 of Eq. 4.34

### Requirement: Links regulated by several signed regulators
The library SHALL provide `SpatialTriadicPercolationSystem` (Millán et al. 2023, arXiv 2311.14877, Methods, Eqs. 1-2) on a
torus, whose hyperedges list the ends of a link, its negative regulators, and then its positive regulators, whose
numbers the link holds, and whose links are active with probability `p` when no negative regulator is active and a
positive regulator is active, or no positive regulator exists.

#### Scenario: Signed regulators
- **WHEN** `SpatialTriadicPercolationSystem` runs one step with `p = 1` on the NumPy and Numba backends
- **THEN** every link is active exactly when none of its negative regulators and one of its positive regulators, if it
  has any, were active

#### Scenario: Blinking patterns
- **WHEN** `SpatialTriadicPercolationSystem` runs with `c+ = c- = 0.2`, `d0 = dr = 0.25`, `c = 0.4`, and `p = 0.8`
- **THEN** the giant component keeps changing size, the active nodes of consecutive steps are anti-correlated, and the
  activity of a node correlates with that of the nodes within `d0` of it

### Requirement: Audited engineering and computer science systems
The library SHALL provide, as system classes built from the general classes of the core, the engineering and
computer science systems of the reference audit that the core supports: `MemristiveMixedModeOscillatorSystem` and
`CoupledColpittsSystem` in `engineering/circuits`, `MagneticLevitationSystem` and `VibratingTableSystem` in
`engineering/mechatronics`, `MinimumTimeControlSystem` in `engineering/control`, `GrainStorageSystem` in
`engineering/processes`, `LightEmittingRobotsSystem` in `engineering/robotics`, and `FredkinReplicatorSystem`,
`TuringMachineSystem`, `SmallWorldLifeSystem`, `DNAComputationSystem`, and `lenia` in `computer_science/automata`. The module of
every system SHALL define the elements, interactions, views, and helpers that only this system uses. Every system
SHALL have tests of a known result of its source, a gallery example, and an inventory entry with its sources and
sections.

#### Scenario: Published waveforms
- **WHEN** `MemristiveMixedModeOscillatorSystem`, `MagneticLevitationSystem`, or `VibratingTableSystem` runs with its defaults
- **THEN** it settles on the mixed-mode oscillation `2^11` of period near 7.8, follows the closed-form periodic
  velocity, or vibrates steadily at the supply frequency close to its linearized model

#### Scenario: Control and synchronization
- **WHEN** `MinimumTimeControlSystem` steers a free particle from `(3, 2)`, or `CoupledColpittsSystem` runs with and without
  coupling
- **THEN** the particle reaches the origin near the time `2 + 2 sqrt(5)` and stays, or the phases drift apart without
  coupling and stay locked for `k = 0.4`

#### Scenario: Machines and automata
- **WHEN** `TuringMachineSystem`, `FredkinReplicatorSystem`, `SmallWorldLifeSystem`, `DNAComputationSystem`, or `lenia` runs
- **THEN** the busy beaver halts after 13 steps with six ones, the pattern reappears four times at the powers of two,
  the unrewired network follows the Game of Life, the filtered strands are the only Hamiltonian path, and Orbium
  glides with a steady mass while one step follows equation 1

#### Scenario: Stocks and robots
- **WHEN** `GrainStorageSystem` runs, or `LightEmittingRobotsSystem` runs with delays of 10 s and -10 s
- **THEN** the stock oscillates and settles at `6000 / (1 + 0.025 * 5)`, and the robots gather with the delay and
  spread with the prediction

### Requirement: Evolution by circumstance of RobbyGA
`RobbyGASystem` SHALL take the regime `'circumstance'`, in which strategies live `lifespan` generations, the fraction
`starvation` with the lowest scores starves, and random survivors breed in the places of the dead, with an optional
period that switches the starvation on and off. The default regime SHALL keep the earlier behaviour.

#### Scenario: Starvation as selection
- **WHEN** `RobbyGASystem` runs in the regime `'circumstance'` with a starvation of 0.3 and of 0
- **THEN** the mean score rises near zero with starvation and stays near the score of random strategies without it
