# model-library Specification

## Purpose

Provides ready-made elements and interactions for common scientific models, built only on the public framework, so a
user can assemble a model without writing laws.

## Requirements

### Requirement: Physics models
The library SHALL provide particles with position, velocity, and mass, constant forces, linear drag, and potentials
whose forces are derived from an energy expression, including harmonic potentials, springs, and gravity.

#### Scenario: Constant acceleration
- **WHEN** a particle of mass `m` starting at rest is subject to a constant force `F` and simulated for time `t`
- **THEN** its displacement matches `F * t**2 / (2 * m)`

#### Scenario: Energy is conserved
- **WHEN** two particles joined by a spring are simulated with no drag
- **THEN** their total energy stays constant within solver tolerance

### Requirement: Ecology models
The library SHALL provide populations with exponential growth, logistic growth, mortality, and predation.

#### Scenario: Logistic saturation
- **WHEN** a population with logistic growth is simulated for a long time
- **THEN** its count approaches the carrying capacity

### Requirement: Epidemiology models
The library SHALL provide compartments with infection and recovery interactions, each with a deterministic mode and a
stochastic mode.

#### Scenario: Switch to stochastic
- **WHEN** a user sets the infection and recovery interactions to stochastic mode
- **THEN** the system compiles to a jump process and the compartment counts stay non-negative integers

### Requirement: Economics models
The library SHALL provide a Solow growth model in continuous time and a cobweb market model in discrete time.

#### Scenario: Solow steady state
- **WHEN** a Solow economy is simulated for a long time
- **THEN** its capital approaches `(s / delta) ** (1 / (1 - alpha))`

#### Scenario: Cobweb convergence
- **WHEN** a cobweb market with a supply slope smaller than the demand slope is simulated
- **THEN** its price converges to the equilibrium price

### Requirement: Finance models
The library SHALL provide an asset with geometric Brownian motion.

#### Scenario: Asset price stays positive
- **WHEN** an asset with geometric Brownian motion is simulated
- **THEN** its price stays positive and its trajectory is reproducible with a fixed random state

### Requirement: Reflecting box and ideal gas
The physics module SHALL provide a box with elastically reflecting walls for particles and particle groups, and a
helper that builds an ideal gas of a given size and temperature.

#### Scenario: Ideal gas law
- **WHEN** an ideal gas of N molecules at temperature kT is simulated in a square box of area A
- **THEN** the pressure measured from wall impulses times A equals N times kT within 2 percent

#### Scenario: Molecules stay in the box
- **WHEN** an ideal gas is simulated for any duration
- **THEN** every molecule stays inside the box and keeps its speed

### Requirement: Pair potentials for particle groups
The physics module SHALL provide Lennard-Jones and soft-sphere pair interactions with a cutoff, built on the core
`PairCouplingInteraction` on positions of order 2, which run on the native solvers and on the LAMMPS backend.

#### Scenario: Lennard-Jones minimum
- **WHEN** two members interact through a Lennard-Jones potential with `sigma = 1`
- **THEN** the force between them is zero at the distance `2 ** (1 / 6)`

### Requirement: Langevin thermostat
The physics module SHALL provide a Langevin thermostat that adds drag and random forces to a particle group, built on
the core `RelaxationInteraction` acting on the first derivative of the position, which runs on the native SDE solvers and on the LAMMPS
backend.

#### Scenario: Equilibrium temperature
- **WHEN** an ideal gas starts at temperature 0.5 under a Langevin thermostat whose `temperature` is 2
- **THEN** its kinetic temperature approaches 2 within 5 percent after many relaxation times

### Requirement: Social agent-based models
The library SHALL provide a `social` module with Schelling segregation, Boltzmann wealth exchange, and the voter model.

#### Scenario: Schelling segregation increases
- **WHEN** a Schelling model with a similarity threshold of 0.5 runs until no agent moves or 100 steps pass
- **THEN** the share of similar neighbours is higher at the end than at the start

#### Scenario: Wealth stays constant
- **WHEN** a Boltzmann wealth model runs for any number of steps
- **THEN** the total wealth of all agents stays equal to its initial value

#### Scenario: Voter consensus on a small network
- **WHEN** a voter model runs on a connected network of 20 agents for long enough
- **THEN** every agent ends with the same opinion

### Requirement: Agent-based predator and prey
The ecology module SHALL provide a wolf and sheep model on a grid with grass regrowth, in which agents gain energy by
eating, reproduce, and die.

#### Scenario: Populations change
- **WHEN** the wolf and sheep model runs for 100 steps with its default parameters
- **THEN** both the wolf and sheep counts change over the run and are reported every frame

### Requirement: Coupled harvesting example
The library SHALL provide a socio-ecological model in which farmer agents harvest a resource whose stock regrows by a
logistic ODE, and the harvest depends on the number of farmers through an aggregate.

#### Scenario: Overharvesting collapses the resource
- **WHEN** the harvesting model runs with a harvest effort high enough that harvest exceeds the largest regrowth
- **THEN** the resource stock falls towards zero over the run

#### Scenario: Sustainable effort
- **WHEN** the harvesting model runs with a harvest effort below the largest regrowth
- **THEN** the resource stock settles at a positive level

### Requirement: Ready-made models are registered factories
Every ready-made model SHALL be a factory function registered in a catalog with its field path, description, tags,
references, and default run settings, and the catalog SHALL list, describe, create, and run models by name.

#### Scenario: Browse the catalog
- **WHEN** a user lists the catalog filtered by the field `biology`, or by the field path `biology/ecology`
- **THEN** the result holds every model of that field or subfield with its description, its references, and the
  formalism inferred from its laws

#### Scenario: Create by name
- **WHEN** a user creates `lotka_volterra` with a predation rate of 0.2
- **THEN** the result is a system that the simulation runs like the same model built from its components

### Requirement: Every registered model runs through the generic interface
Every registered model SHALL be created with its defaults, compiled, simulated, logged to Rerun, and cloned through the
same public interface, without code specific to the model.

#### Scenario: Conformance
- **WHEN** the conformance suite runs over the catalog
- **THEN** every model completes all five steps, and its trajectory holds every declared state and observable

### Requirement: The library uses only the public interface
Library modules SHALL build their models only from the public names of `skcomplex.core` and third-party numerical
libraries used inside laws, and no model package SHALL import another.

#### Scenario: Private import
- **WHEN** a library module imports a private name of another package
- **THEN** the library import check fails and names the module and the import

### Requirement: Higher-order models across fields
The library SHALL provide, as registered factories written in their natural order:
- the nonlinear pendulum as Newton's law;
- a driven damped oscillator;
- a series RLC circuit as a second-order charge equation;
- Sprott's jerk equation;
- Samuelson's multiplier-accelerator;
- a Fibonacci-type recurrence.

#### Scenario: Circuit and oscillator agree
- **WHEN** the RLC circuit and the damped oscillator run with matching coefficients `L = m`, `R = c`, and `1 / C = k`
- **THEN** the charge and the displacement trajectories are identical

### Requirement: Nonlinear dynamics models
The library SHALL provide the Lorenz, Rössler, Hénon, logistic map, Duffing, Van der Pol, and Hénon-Heiles models as
registered factories with their standard parameters.

#### Scenario: Strange attractor
- **WHEN** a user runs the Lorenz model with its defaults for 50 time units
- **THEN** the trajectory stays bounded and visits both lobes of the attractor

### Requirement: Delay, event, and forced models
The library SHALL provide Mackey-Glass, Hutchinson's delayed logistic, delayed Kuramoto, a bouncing ball, Newton's
cradle, a circular billiard, integrate-and-fire and Izhikevich neurons, the forced Duffing oscillator, stochastic
resonance, and a Brownian particle near a wall with multiplicative noise as registered factories.

#### Scenario: Izhikevich regular spiking
- **WHEN** the Izhikevich neuron runs with the regular-spiking parameters and a constant input of 10
- **THEN** it fires periodically with each spike reset recorded as an event

### Requirement: Lattice models
The library SHALL provide elementary automata, Game of Life, majority rule, Langton's ant, Nagel-Schreckenberg traffic,
the hodgepodge machine, the Bak-Tang-Wiesenfeld sandpile, the Drossel-Schwabl forest fire, Olami-Feder-Christensen
earthquakes, and the Ising, Potts, and XY models as registered factories.

#### Scenario: Traffic jams
- **WHEN** the Nagel-Schreckenberg model runs at densities 0.1 and 0.5 with slowdown probability 0.3
- **THEN** the mean flow at density 0.5 is lower than at the critical density near 0.15, and jams appear as backward
  moving clusters of stopped cars

### Requirement: Field models
The library SHALL provide heat diffusion, Fisher-KPP, Gray-Scott, Brusselator, FitzHugh-Nagumo spirals, Schnakenberg
Turing patterns, and pheromone foraging as registered factories.

#### Scenario: Fisher front speed
- **WHEN** the Fisher-KPP model runs with growth rate `r` and diffusion `D` from a step profile
- **THEN** the front moves at a speed within 5% of `2 * sqrt(r * D)`

#### Scenario: Turing spots
- **WHEN** the Schnakenberg model runs in its Turing-unstable regime from small noise
- **THEN** a stationary pattern with a dominant wavelength within 10% of the linear prediction forms

### Requirement: PDE models
The library SHALL provide Kuramoto-Sivashinsky, complex Ginzburg-Landau, Swift-Hohenberg, Burgers' shock, the
shallow-water dam break, decaying two-dimensional turbulence, Rayleigh-Bénard convection, and heat on an L-shaped
domain as registered factories.

#### Scenario: Swift-Hohenberg rolls
- **WHEN** the Swift-Hohenberg model runs above onset from small noise
- **THEN** a roll pattern forms with a wavelength within 5% of 2π

### Requirement: Mean-field models read group reductions
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
- **WHEN** `kuramoto_sakaguchi_chimera` runs with 32 oscillators per population
- **THEN** each population reads the means of `cos theta` and `sin theta` of both populations, and after the
  transient one population's order parameter exceeds 0.99 while the other's stays below 0.95

#### Scenario: Winfree mean influence
- **WHEN** `winfree` is built with 50 oscillators
- **THEN** its drift equals `omega_i - sin(theta_i) (K / N) sum_j (1 + cos theta_j)` within 1e-12, with no pair
  interaction and no self-influence correction

#### Scenario: Theta neurons with an instantaneous mean
- **WHEN** `theta_neurons` runs with 2000 neurons
- **THEN** its state holds only the phases, every neuron reads the mean pulse of all neurons at the current state, and
  the complex order parameter after the transient agrees with the Ott-Antonsen reduction of Luke, Barreto, and So
  (2013) within 0.05

#### Scenario: LEGION inhibitor reads the maximum
- **WHEN** `legion` runs on its default stimulus of two objects
- **THEN** the inhibitor is driven by the maximum oscillator activity, and after the transient no frame has both
  objects active at once while each object is active in some frame

#### Scenario: Winners take all with a group
- **WHEN** `winners_take_all` runs with its default inputs and `beta = 2`
- **THEN** its system has one inhibition interaction that reads the group total, whatever the number of units, and
  only the unit with input 1.1 stays active, with activity within 1e-3 of 1.1

#### Scenario: Replicator strategies as members
- **WHEN** `replicator_dynamics` runs rock, paper, scissors with 3 strategies as 3 members of a group
- **THEN** the shares sum to 1 within 1e-10 over `t` in `[0, 100]`, and the product of the shares stays within 1e-6
  of its initial value

#### Scenario: Hawks and doves
- **WHEN** `replicator_dynamics` runs hawks and doves with `V = 2` and `C = 3`
- **THEN** the hawk share approaches `2 / 3` within 1e-4

#### Scenario: Hypercycle on the simplex
- **WHEN** `hypercycles` runs with 4 molecules as members of a group on the cycle topology
- **THEN** the concentrations sum to 1 within 1e-10 and converge to a fixed point where every concentration exceeds
  0.01

#### Scenario: Degree classes as members
- **WHEN** `heterogeneous_mean_field_sis` runs with `lam` at half and twice `<k> / <k^2>`
- **THEN** its degree classes are members of a group with their degree and weight as parameters, and the infected
  density falls below 1e-6 at half and stays above 1e-3 at twice

#### Scenario: Wealth exchange with the mean
- **WHEN** `bouchaud_mezard_wealth` is built with no graph and 50 agents
- **THEN** each agent exchanges with the group mean, the exchange part of the drift equals that of the
  complete-graph Laplacian form within 1e-12, and the exchange conserves the total wealth within 1e-12

### Requirement: Models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are rebuilt on group reductions under their names and field paths, and marked implemented:
  - `kuramoto` in `physics/synchronization`
  - `kuramoto_sakaguchi_chimera` in `physics/synchronization`
  - `winfree` in `physics/synchronization`
  - `theta_neurons` in `biology/neuroscience`
  - `legion` in `biology/neuroscience`
  - `winners_take_all` in `biology/neuroscience`
  - `replicator_dynamics` in `biology/evolution`
  - `hypercycles` in `biology/evolution`
  - `heterogeneous_mean_field_sis` in `biology/epidemiology`
  - `bouchaud_mezard_wealth` in `economics/finance`

#### Scenario: Neighbourhood models stay blocked
- **WHEN** the change is applied and the inventory check runs
- **THEN** `szabo_cells` and `active_passive_flocking` in `physics/active_matter` stay blocked on
  `add-adaptive-networks`, which specifies the neighbourhood reductions they need

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Pulse-coupled oscillators are members of a group
The library SHALL build `mirollo_strogatz` in `physics/synchronization` with the oscillators as members of a group,
whose firing pulses reach the other members through a reduction, rather than as one vector state in a Euclidean space.

#### Scenario: Mirollo-Strogatz synchronizes as a group
- **WHEN** `mirollo_strogatz` runs with its default excitatory coupling from random phases
- **THEN** its oscillators are members of one group, and they fire in synchrony within the run

### Requirement: Boolean networks take any number of inputs
`random_boolean_network` and `yeast_cell_cycle` SHALL bind the inputs of every node through one variadic role, so a node
reads any number of inputs up to the number of nodes and no input role is bound to the target as padding.

#### Scenario: Eight inputs per node
- **WHEN** `random_boolean_network(n_nodes=12, k=8)` is built and simulated for 50 steps
- **THEN** every node's update reads 8 distinct nodes through a table of 256 entries, and the run completes

#### Scenario: Too many inputs
- **WHEN** `random_boolean_network(n_nodes=4, k=5)` is called
- **THEN** a named validation error is raised that names `k` and the number of nodes

#### Scenario: Yeast cycle unchanged
- **WHEN** `yeast_cell_cycle()` runs from its default start for 15 steps
- **THEN** its trajectory equals the one recorded from the five-role version before the rebuild, and it ends in the
  stationary G1 state with only Cdh1 and Sic1 on

### Requirement: Ring roads are one group of cars
`optimal_velocity` and `intelligent_driver_model` SHALL hold their cars as one group and one interaction that reads the
car ahead by a periodic member offset, for any number of cars.

#### Scenario: One group whatever the number of cars
- **WHEN** `optimal_velocity(n_cars=50)` is built
- **THEN** the system holds one group of 50 cars and one interaction, and its trajectory has the arrays `cars.x` and
  `cars.v` with 50 columns

#### Scenario: Same dynamics as one interaction per car
- **WHEN** `optimal_velocity(n_cars=20)` and `intelligent_driver_model(n_cars=20)` run with their default settings
- **THEN** positions and speeds match those of the per-car versions recorded before the rebuild within a relative
  tolerance of 1e-8

#### Scenario: Jam on the rebuilt ring
- **WHEN** the rebuilt optimal velocity model runs above its critical density from near-uniform flow
- **THEN** the spread of the car speeds grows and a stop-and-go wave forms

### Requirement: Agent models read groups through roles and the space through the model
`trafficking` and `meadow_capacity` SHALL bind every group their rules read to a role, `trafficking` and
`ring_road_drivers` SHALL read the space's extent from the model view, no rule of an interaction in the library SHALL
read a group other than through its roles, and no rule SHALL read its own parameters by its label. Rules of agent
types, which have no roles, SHALL read the groups of their own system by relative label.

#### Scenario: Trafficking with two roles
- **WHEN** `trafficking(random_state=0)` is built and run for 90 days
- **THEN** its daily interaction binds the roles `women` and `premises`, has no `width` or `height` argument, and the
  bills and trips equal those recorded before the rebuild

#### Scenario: Nested meadow
- **WHEN** the meadow model built with random state 0 is placed as the subsystem `farm` of another system and run
- **THEN** it runs, its grazing and herding interactions bind the role `pasture`, and its cow counts equal those of the
  model run at the root

#### Scenario: Ring road without a length parameter
- **WHEN** `ring_road_drivers(10, length=500.0)` runs
- **THEN** the drivers have no `road_length` parameter and every position stays in `[0, 500)`

#### Scenario: No rule reads its own label
- **WHEN** the library's layout test scans the source of every model
- **THEN** no rule reads a parameter by its own label, and no interaction names a group in `model.agents`,
  `model.count`, or `model.create` by a string or an attribute that is not one of its roles

### Requirement: Tumour growth laws
The library SHALL provide `exponential_linear_growth`, `power_law_growth`, `von_bertalanffy`, `generalized_logistic`,
`gompertz`, and `dynamic_carrying_capacity` as registered factories under `biology/oncology`, as the model inventory
lists them with their sources, each with a gallery example.

#### Scenario: Gompertz growth
- **WHEN** `gompertz` runs from a small tumour
- **THEN** its volume matches the closed-form Gompertz solution within 1e-6 relative and approaches the carrying
  capacity

### Requirement: Cell kill and toxicity
The library SHALL provide `skipper_log_kill`, `norton_simon`, and `drug_toxicity` as registered factories under
`biology/oncology` and `biology/physiology`, as the model inventory lists them with their sources, each with a gallery
example.

#### Scenario: Log kill
- **WHEN** `skipper_log_kill` receives equal impulsive doses on a non-growing tumour
- **THEN** each dose lowers the logarithm of the tumour size by the same amount within 1e-6

### Requirement: Chemotherapy
The library SHALL provide `de_pillis_radunskaya`, `multidrug_resistance`, `drug_resistance_switching`,
`delayed_resting_hunting_chemotherapy`, `delayed_metastasis`, `cell_cycle_chemotherapy`, and
`erythropoiesis_mercaptopurine` as registered factories under `biology/oncology`, as the model inventory lists them with
their sources, each with a gallery example.

#### Scenario: Chemotherapy clears a tumour
- **WHEN** `de_pillis_radunskaya` runs untreated and then with a constant chemotherapy infusion of 1
- **THEN** the untreated tumour settles at its high equilibrium, and the treated tumour falls below 1% of its initial
  size while the normal cells stay above 0.7 of their carrying capacity

### Requirement: Immunotherapy
The library SHALL provide `kirschner_panetta`, `dendritic_cell_vaccine`, `anti_pd1_checkpoint`, `delayed_dc_vaccine`,
and `tumour_immune_suppression` as registered factories under `biology/oncology`, as the model inventory lists them with
their sources, each with a gallery example.

#### Scenario: Tumour-immune oscillations and clearance
- **WHEN** `kirschner_panetta` runs with an antigenicity in the oscillatory range and then with adoptive therapy and
  IL-2 above the book's treatment factors `a1 = 500` and `a2 = 7e7`
- **THEN** the untreated tumour and effector cells oscillate, and the treated tumour is cleared

### Requirement: Anti-angiogenic therapy
The library SHALL provide `hahnfeldt` and `vascular_core_periphery` as registered factories under `biology/oncology`, as
the model inventory lists them with their sources, each with a gallery example.

#### Scenario: Hahnfeldt equilibrium
- **WHEN** `hahnfeldt` runs untreated
- **THEN** the tumour and carrying capacity approach the equilibrium volume `(b/d)^(3/2)` within 1%

### Requirement: Radiotherapy
The library SHALL provide `radiotherapy_competition` and `lq_radiosensitizer` as registered factories under
`biology/oncology`, as the model inventory lists them with their sources, each with a gallery example.

#### Scenario: Linear-quadratic fractions
- **WHEN** `lq_radiosensitizer` delivers a fraction of dose `d` without radiosensitizer
- **THEN** the surviving tumour fraction of that fraction is `exp(-alpha d - beta d^2)` within 1e-6

### Requirement: Hormone therapy
The library SHALL provide `continuous_androgen_suppression`, `ideta_intermittent_androgen`, `prostate_cell_quota`,
`droop_cell_quota`, `prostate_androgen_receptor`, `piecewise_linear_ias`, and `tamoxifen_cell_cycle` as registered
factories under `biology/ecology` and `biology/oncology`, as the model inventory lists them with their sources, each
with a gallery example.

#### Scenario: Intermittent suppression
- **WHEN** `ideta_intermittent_androgen` runs with its PSA thresholds
- **THEN** the therapy switches on as PSA rises through the upper threshold and off as it falls through the lower one,
  and PSA cycles around the thresholds, overshooting them by the androgen delay, while treatment is effective

### Requirement: Gene, viral, nanoparticle, and stem-cell therapies
The library SHALL provide `tcr_gene_therapy`, `sirna_tgf_beta`, `oncolytic_virus`, `oncolytic_virus_dendritic`,
`nanoparticle_corona`, `two_compartment_pharmacokinetics`, `stem_cell_differentiation`, `bone_marrow_blood_leukaemia`,
and `leukaemia_stem_cells` as registered factories under `biology/oncology`, `biology/physiology`, and
`chemistry/kinetics`, as the model inventory lists them with their sources, each with a gallery example.

#### Scenario: Two-compartment pharmacokinetics
- **WHEN** `two_compartment_pharmacokinetics` receives one bolus dose
- **THEN** the central concentration matches the closed-form bi-exponential solution within 1e-6 relative, and the total
  drug decays by elimination only

### Requirement: Combination therapies
The library SHALL provide `chemoimmunotherapy`, `clonal_chemoimmunotherapy`, `chemovirotherapy`,
`glioma_chemovirotherapy`, `her2_targeted_therapy`, and `angiogenesis_immunotherapy` as registered factories under
`biology/oncology`, as the model inventory lists them with their sources, each with a gallery example.

#### Scenario: Combination beats each therapy alone
- **WHEN** `chemovirotherapy` runs with the book's chemotherapy alone, the virus alone, and both
- **THEN** the tumour at the end of the default run of 64 days, before the virus bursts through the tumour, is
  smallest under the combination

### Requirement: Cellular network models
The library SHALL provide `phosphorylation_oscillator`, `biochemical_systems_theory`, `three_gene_boolean_network`,
`mammalian_cell_cycle`, `boolean_repressilator`, `goodwin_oscillator`, and `generalized_lotka_volterra` as registered
factories under `biology/ecology` and `biology/systems_biology`, as the model inventory lists them with their sources,
each with a gallery example.

#### Scenario: Goodwin oscillations
- **WHEN** `goodwin_oscillator` runs with three variables and Hill coefficients 6 and 12
- **THEN** it settles on its stable fixed point at 6 and on a limit cycle at 12, the threshold for three equal
  degradation rates being 8

### Requirement: Surveyed models are recorded in the inventory
The inventory SHALL record every model of Padmanabhan et al. 2021, Konieczny et al. 2023, and Raman 2021 under the
source keys `Pa`, `Ko`, and `Rm`: the models above as pending this change, models pending other changes, models blocked
on `add-constraint-based-models`, out-of-scope entries, and the new sources of models the library already implements.

#### Scenario: Source of an implemented model
- **WHEN** a contributor reads the inventory entry of `michaelis_menten`
- **THEN** its sources include `Ko 1.1.4` and `Rm 5.3.1`

### Requirement: Mean-field and disordered spin systems
The library SHALL provide `curie_weiss`, `mattis_spin_glass`, `edwards_anderson`, and `sherrington_kirkpatrick` as
registered factories under `physics/statistical_mechanics`, as the model inventory lists them with their sources, each
with a gallery example.

#### Scenario: Curie-Weiss magnetization
- **WHEN** `curie_weiss` with 2000 spins runs to stationarity below and above the critical temperature `T_c = J`
- **THEN** the mean absolute magnetization at `T = 0.5 J` matches the positive root of `m = tanh(m J / T)` within 0.02,
  and at `T = 2 J` it is below 0.05

### Requirement: Driven and noisy particles
The library SHALL provide `colored_noise_langevin`, `on_off_intermittency`, `biased_trap_walk`,
`tilted_periodic_potential`, `flashing_ratchet`, and `shaken_spring_block` as registered factories under
`mathematics/stochastic_processes` and `physics/soft_matter`, as the model inventory lists them with their sources, each
with a gallery example.

#### Scenario: Ratchet current
- **WHEN** `flashing_ratchet` runs with an asymmetric and with a symmetric sawtooth potential at the same switching rate
- **THEN** the asymmetric potential gives a mean current whose sign follows the asymmetry and exceeds three standard
  errors, and the symmetric one gives a current within three standard errors of zero

### Requirement: Birth-death and interacting particle processes
The library SHALL provide `birth_death_queue`, `absorbing_birth_death`, `zero_range_process`, `dissipative_zero_range`,
and `asep` as registered factories under `mathematics/stochastic_processes` and `physics/statistical_mechanics`, as the
model inventory lists them with their sources, each with a gallery example.

#### Scenario: Stationary queue
- **WHEN** `birth_death_queue` runs as an M/M/1 queue with load `rho = 0.5`
- **THEN** the time-averaged occupation distribution matches `(1 - rho) rho^n` within 0.01 for `n` up to 5

### Requirement: Active matter and traffic
The library SHALL provide `aligning_active_brownian_particles`, `binary_collision_flocking`, `langevin_traffic`, and
`traffic_density_equation` as registered factories under `engineering/traffic` and `physics/active_matter`, as the model
inventory lists them with their sources, each with a gallery example.

#### Scenario: Alignment transition
- **WHEN** `aligning_active_brownian_particles` runs at a low and a high alignment strength at fixed density and noise
- **THEN** the stationary polarization is below 0.2 at the low strength and above 0.6 at the high strength

### Requirement: Agents, evolution, and populations
The library SHALL provide `perishable_goods_duopoly`, `standardized_agents`, `origin_fixation_walk`,
`neutral_biodiversity`, `neutral_clustering`, and `heterogeneous_mean_field_sir` as registered factories under
`biology/ecology`, `biology/epidemiology`, `biology/evolution`, `economics/markets`, and `social/opinion_dynamics`, as
the model inventory lists them with their sources, each with a gallery example.

#### Scenario: Epidemic threshold on networks
- **WHEN** `heterogeneous_mean_field_sir` runs on a degree distribution with transmission rates below and above the
  threshold `<k>/(<k^2> - <k>)` of the recovery rate, close to `<k>/<k^2>`
- **THEN** the final epidemic size is below 1% below the threshold and above 5% above it

### Requirement: Maps and coupled oscillators
The library SHALL provide `cubic_chaotic_map`, `skew_tent_map`, `doubling_map`, `subcritical_hopf_form`,
`oscillator_death`, and `heterogeneous_global_coupling` as registered factories under `mathematics/dynamical_systems`
and `physics/synchronization`, as the model inventory lists them with their sources, each with a gallery example.

#### Scenario: Invariant density of the tent map
- **WHEN** `skew_tent_map` iterates an ensemble of 10000 points for 200 steps
- **THEN** the histogram of the final points is uniform on [0, 1] within the sampling error of each of 20 bins

### Requirement: Bertin models are recorded in the inventory
The inventory SHALL record every model of Bertin 2021 under the source key `Bt`: the models above as pending this
change, `levy_flight` and `trap_model` as pending `add-anomalous-processes`, `random_walk_with_memory` and
`diluted_hopfield` as pending `add-stochastic-maps`, `random_graph_ensembles` as pending `add-adaptive-networks`,
`random_energy_model` and `perceptron_satisfiability` as blocked on `add-static-models`, and the new sources of models
the library already has.

#### Scenario: Source of an existing model
- **WHEN** a contributor reads the inventory entry of `kuramoto`
- **THEN** its sources include `Bt 7.3.1`

### Requirement: Mandelbrot and Julia sets
The library SHALL provide the Mandelbrot set and the filled Julia sets of the complex quadratic map `z → z² + c`
(Dilão 2023 §5.5) as registered factories under the field path `mathematics/fractals`. Each SHALL iterate the map at
every site of a lattice of complex points and record the step at which the modulus first exceeds an escape radius.

#### Scenario: Mandelbrot escape times
- **WHEN** the Mandelbrot model runs for 100 steps on a lattice whose sites include `c = 1`, `c = -1`, `c = -2`, and
  `c = i`
- **THEN** the site at `c = 1` escapes at step 3, since `0 → 1 → 2 → 5`, and the sites at `c = -1`, `c = -2`, and
  `c = i` never escape

#### Scenario: Julia set of the squaring map
- **WHEN** the Julia model with `c = 0` runs for 100 steps on a 201 × 201 lattice over the square `[-2, 2]²`
- **THEN** every site with `|z₀| < 0.99` never escapes, and every site with `|z₀| > 1.01` has escaped

#### Scenario: Mandelbrot symmetry
- **WHEN** the Mandelbrot model runs on a window symmetric about the real axis
- **THEN** the escape-time image equals its mirror image across the real axis

### Requirement: Stochastic map models
The library SHALL provide, as registered factories built only on the public core, the persistent random walk under
`mathematics/stochastic_processes` (Jensen 2023 §10.4), Markov and hidden Markov sources under
`mathematics/stochastic_processes` (Lindgren 2024 §3.2), and iterated function systems drawn by the chaos game under
`mathematics/fractals` (Dilão 2023 §5.4). The inventory SHALL mark the three entries as implemented.

#### Scenario: Persistent random walk diffuses faster
- **WHEN** 2000 persistent random walkers that keep their direction with probability 0.8 run for 1000 steps
- **THEN** the mean squared displacement divided by the time is `(1 + ρ) / (1 - ρ) = 4` within 10%, where
  `ρ = 2 × 0.8 - 1`

#### Scenario: Markov source entropy rate
- **WHEN** a Markov source with transition matrix `[[0.9, 0.1], [0.4, 0.6]]` runs for 100000 steps
- **THEN** the entropy rate of its symbols estimated from block entropies is 0.569 bits within 0.01

#### Scenario: Hidden Markov source frequencies
- **WHEN** a hidden Markov source with hidden transition matrix `[[0.9, 0.1], [0.4, 0.6]]` and emission matrix
  `[[0.7, 0.3], [0.2, 0.8]]` runs for 100000 steps
- **THEN** the frequency of the first observed symbol is 0.6 within 0.01

#### Scenario: Sierpiński triangle by the chaos game
- **WHEN** a group of 100000 points iterates the three maps of the Sierpiński triangle with equal weights for 30 steps
- **THEN** the box-counting dimension of the final points, rasterised on a 128 by 128 grid with box sizes from 1 to 32
  cells, is `log 3 / log 2` within 0.05

### Requirement: Probabilistic Boolean networks
The library SHALL provide `probabilistic_boolean_network` in `biology/systems_biology`, a Boolean network that draws
each node's transfer function from its candidates with given probabilities at every step, as Raman 2021 section 7.3.1
poses it.

#### Scenario: Selection probabilities
- **WHEN** `probabilistic_boolean_network` runs for many steps with one node choosing between two functions with
  probabilities 0.7 and 0.3
- **THEN** the share of steps that used the first function is 0.7 within three standard errors

### Requirement: Bertin models on stochastic maps
The library SHALL provide `random_walk_with_memory` in `mathematics/stochastic_processes` and `diluted_hopfield` in `computer_science/neural_networks`, as Bertin 2021 poses them and the model inventory records them, built on this
change's capability, each with a gallery example.

#### Scenario: Registered under the inventory names
- **WHEN** the change is applied and the inventory check runs
- **THEN** both models are registered under their field paths and marked implemented

### Requirement: Intrinsic-instability populations
The library SHALL provide `intrinsic_instability` in `biology/ecology`, the resident and invading populations of
Jensen 2023 section 13.2, Eqs. (13.3) and (13.4), whose resident population follows an Ornstein-Uhlenbeck step with
noise scaled by the square root of its size and is suppressed by an invader that a rare mutation seeds and that then
grows exponentially, built on this change's capability, with a gallery example.

#### Scenario: Leaving the fixed point from within
- **WHEN** `intrinsic_instability` runs for 4000 steps with its default parameters and random state 0
- **THEN** the invader is seeded before step 3000, the resident population stays within 0.15 of 1 before the
  invader appears, and at the end the resident population is below 0.5 and the invader above 1

### Requirement: Substitution fractals
The library SHALL provide the Koch curve as an L-system with a turtle, the Cantor set as a segment substitution, and
the Sierpiński triangle and carpet as polygon substitutions, as registered factories whose generations are the
simulation's steps (Thurner et al. 2018 §3.5.2, Trzaska 2023 §3.5).

#### Scenario: Koch curve
- **WHEN** `run('koch_curve', until=5)` is called
- **THEN** the word at the last frame has 1024 symbols `F`, the curve's total length is `(4/3)^5` within 1e-12, and
  the box-counting dimension of its raster at resolution 2048, over boxes of 16 to 256 cells, is
  `log 4 / log 3 ≈ 1.262` within 0.05

#### Scenario: Cantor set
- **WHEN** `run('cantor_set', until=8)` is called
- **THEN** `trajectory['cantor.size']` is `2^n` at generation `n`, and the total length is `(2/3)^n` within 1e-12 at
  every frame

#### Scenario: Sierpiński carpet
- **WHEN** `run('sierpinski_substitution', until=4)` is called with `shape='carpet'`
- **THEN** the patch holds 4096 squares and their area is `(8/9)^4` of the unit square within 1e-12

#### Scenario: Gallery analysis
- **WHEN** the gallery example of any of the three models runs
- **THEN** its analysis figure plots the log measure against the generation with the exact line, and the log box count
  against the log box size with the slope of the similarity dimension

### Requirement: Rewriting models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `koch_curve` in `mathematics/fractals` (Th 3.5.2, Tr 3.5)
  - `cantor_set` in `mathematics/fractals` (Th 3.5.2, Tr 3.5)
  - `sierpinski_substitution` in `mathematics/fractals` (Th 3.5.2, Tr 3.5)

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Anomalous random walk models
The library SHALL provide, as registered factories under the field path `physics/soft_matter`:
- the continuous-time random walk, Gaussian jumps after power-law waiting times (Argun et al. 2021 §6.3, Metzler and
  Klafter 2000);
- the Lévy walk, flights at constant speed in random directions for power-law flight times (Argun et al. 2021 §6.3,
  Zaburdaev, Denisov, and Klafter 2015).

#### Scenario: Subdiffusive random walk
- **WHEN** the continuous-time random walk with waiting-time exponent `α = 0.5` runs for 10⁴ walkers to time 10⁴
- **THEN** the ensemble mean squared displacement grows with a scaling exponent within 0.1 of 0.5

#### Scenario: Superdiffusive Lévy walk
- **WHEN** the Lévy walk with flight-time exponent `σ = 1.5` runs for 2000 walkers to time 2000
- **THEN** the ensemble mean squared displacement grows with a scaling exponent within 0.1 of `3 - σ = 1.5`

### Requirement: Anomalous diffusion models with memory or heterogeneity
The library SHALL provide, as registered factories under the field path `physics/soft_matter`:
- annealed transient time motion, Brownian motion whose diffusion coefficient is redrawn at renewals with waiting
  times that depend on it (Argun et al. 2021 §6.3, Massignan et al. 2014);
- fractional Brownian motion of a given Hurst index (Argun et al. 2021 §6.1, Mandelbrot and Van Ness 1968).

#### Scenario: Fractional Brownian motion
- **WHEN** the fractional Brownian motion model with Hurst index 0.3 runs for 1000 members
- **THEN** the ensemble mean squared displacement grows with a scaling exponent within 0.05 of 0.6, and the Hurst
  exponent of the increments by detrended fluctuation analysis lies within 0.05 of 0.3

#### Scenario: Annealed transient time motion
- **WHEN** annealed transient time motion with diffusion coefficients of density `P(D) ~ D^(σ-1)` for `σ = 0.5`, and
  waiting times of mean `D^(-γ)` for `γ = 1`, runs for 10⁴ walkers to time 2000
- **THEN** the ensemble mean squared displacement grows with a scaling exponent within 0.1 of `σ / γ = 0.5`, as
  derived by Massignan et al. (2014)

### Requirement: No inventory model is blocked by anomalous processes
The model inventory SHALL mark the continuous-time random walk, the Lévy walk, annealed transient time motion, and
fractional Brownian motion in `physics/soft_matter` as implemented.

#### Scenario: Inventory status
- **WHEN** the inventory is read after this change
- **THEN** no entry has the status `blocked add-anomalous-processes`, and each of the four models resolves to a
  registered factory under `physics/soft_matter`

### Requirement: Lévy flights and trap models
The library SHALL provide `levy_flight` in `physics/soft_matter` and `trap_model` in `physics/statistical_mechanics`, as Bertin 2021 poses them and the model inventory records them, built on this
change's capability, each with a gallery example.

#### Scenario: Registered under the inventory names
- **WHEN** the change is applied and the inventory check runs
- **THEN** both models are registered under their field paths and marked implemented

### Requirement: Models on lattice geometries and graphs
The library SHALL provide, as registered factories:
- the Ising model on a graph under `physics/statistical_mechanics` (Argun et al. 2021 §2.1);
- the Drossel-Schwabl forest fire on triangular lattices and on graphs under `physics/criticality` (Argun et al. 2021
  ch. 3);
- the FHP lattice gas under `physics/fluids` (Dilão 2023 §5.16);
- the Bak-Sneppen model of coevolution with the extremal update under `biology/evolution` (Gros 2015 §6.6, Thurner et
  al. 2018 §5.4.4, Jensen 2023 §4.2).

#### Scenario: FHP conserves mass and momentum
- **WHEN** the FHP lattice gas runs for 500 steps on a periodic 64 × 64 triangular lattice from random occupations
- **THEN** the particle number and both components of the total momentum are the same at every step as at the start

#### Scenario: Bak-Sneppen self-organises
- **WHEN** Bak-Sneppen runs on a ring of 200 species for 10⁶ extremal updates after a transient of 10⁶ updates
- **THEN** the median fitness exceeds 0.75, and the largest minimum recorded in the events lies between 0.6 and 0.72,
  around the critical value 0.667

#### Scenario: Ising on a graph orders
- **WHEN** the Ising model on a graph runs on a random 4-regular graph of 1000 nodes at temperatures 1.0 and 6.0
  with ensembles
- **THEN** the mean absolute magnetisation is above 0.9 at temperature 1.0 and below 0.1 at temperature 6.0

#### Scenario: Forest fire on a graph
- **WHEN** the forest fire runs on a connected Watts-Strogatz graph of 500 nodes, fully forested, with no growth and no
  lightning, from one burning node
- **THEN** every node burns, and each node starts burning at the step equal to its graph distance from the first one

### Requirement: Network structure analysis model
The library SHALL provide `network_structure_analysis` under `mathematics/networks` (Rm 3.1, 3.3, 3.4, Bt 6.1.1,
6.1.2), a registered factory built on the public core that takes a NetworkX graph, records it as one agent per node,
and reports its centralities, mixing, communities, and triangle z-score as observables. The inventory SHALL mark it as
implemented.

#### Scenario: Measures of a generated network
- **WHEN** `network_structure_analysis` runs on a stochastic block model of 4 blocks of 100 nodes
- **THEN** its trajectory holds the network and observables of clustering, mean path length, assortativity, modularity,
  and the agreement of the detected communities with the blocks above 0.9

#### Scenario: Given graph
- **WHEN** `network_structure_analysis` runs on Zachary's karate club graph from NetworkX
- **THEN** its reported clustering and mean path length equal those of NetworkX within 1e-12, and the community
  detection splits the club into at least two communities with modularity above 0.35

#### Scenario: Gallery example
- **WHEN** a user opens the gallery example of `network_structure_analysis`
- **THEN** it shows the Rerun screenshot of the graph colored by community, and below it the degree distribution and the
  motif z-scores as analysis figures

### Requirement: Models are organized by scientific field
The library SHALL organize models into packages per scientific field, `physics`, `chemistry`, `biology`,
`economics`, `social`, `engineering`, `mathematics`, and `computer_science`, each split into subfield packages. Every
model SHALL live in exactly one subfield.

#### Scenario: Import by field
- **WHEN** a user imports `skcomplex.models.physics.fluids`, `skcomplex.models.biology.ecology`, or
  `skcomplex.models.chemistry.pattern_formation`
- **THEN** each package exports the factories and component types of its subfield, such as `burgers`,
  `lotka_volterra`, and `gray_scott`

#### Scenario: Technique packages are gone
- **WHEN** a user imports `skcomplex.models.fluids`, `skcomplex.models.lattices`, `skcomplex.models.fields`, or
  `skcomplex.models.nonlinear`
- **THEN** the import fails, and every model of those packages is found in its scientific subfield

#### Scenario: One home per model
- **WHEN** the catalog is listed in full
- **THEN** every model name appears once, with a field path of the form `<field>/<subfield>`

### Requirement: Models document their pattern
Every registered model SHALL document how it is built from the interface: its states, parameters, and laws as written,
and its equations as the symbolic layer shows them, so a user can copy it as the pattern for a model of their own.

#### Scenario: Equations in the documentation
- **WHEN** a user reads the documentation page of the field `biology`
- **THEN** every ODE, SDE, map, and field model of the field shows its equations as `equations_of` returns them, and
  every agent, lattice, and Monte Carlo model shows its laws as written

### Requirement: Physics model collection
The physics packages SHALL provide models across mechanics, celestial mechanics, oscillators and synchronization,
statistical mechanics, stochastic processes and soft matter, active matter, fluids, waves, and cosmology, as the model
inventory lists them.

#### Scenario: Kepler orbit
- **WHEN** the Kepler two-body model runs for ten periods
- **THEN** its energy and angular momentum stay within solver tolerance, and its period matches Kepler's third law
  within 0.1%

#### Scenario: Active Brownian particles
- **WHEN** active Brownian particles run in two dimensions with speed `v` and rotational diffusion `D_r`
- **THEN** their mean squared displacement at long times grows as `(4 D_t + 2 v^2 / D_r) t` within 5%

### Requirement: Chemistry model collection
The chemistry packages SHALL provide reaction kinetics, oscillating reactions, and pattern formation models, as the
model inventory lists them.

#### Scenario: Oregonator oscillates
- **WHEN** the Oregonator runs with its standard parameters
- **THEN** it settles on a relaxation limit cycle with a stable period

### Requirement: Biology model collection
The biology packages SHALL provide ecology, epidemiology, evolution, neuroscience, systems biology, physiology, and
collective behaviour models, as the model inventory lists them.

#### Scenario: Repressilator oscillates
- **WHEN** the repressilator runs as ODEs and as a jump process with the same rates
- **THEN** both oscillate, and their periods agree within 10%

#### Scenario: Hodgkin-Huxley spikes
- **WHEN** the Hodgkin-Huxley neuron receives a constant current above threshold
- **THEN** it fires periodically, and its spike threshold current lies within 10% of the published value

### Requirement: Earth science model collection
The earth packages SHALL provide climate, oceanography, atmosphere, and seismology models, as the model inventory
lists them.

#### Scenario: Ice-albedo bistability
- **WHEN** the energy-balance model runs from a warm and from a cold initial temperature at the same solar constant
- **THEN** the runs settle in different stable climates

### Requirement: Economics model collection
The economics packages SHALL provide macroeconomics, markets, finance, and econophysics models, as the model
inventory lists them.

#### Scenario: Kinetic wealth exchange
- **WHEN** the random money exchange model runs with conserved money
- **THEN** the total money stays constant and the wealth distribution approaches an exponential

### Requirement: Social science model collection
The social packages SHALL provide opinion dynamics, segregation, spreading, cooperation, and crowd models, as the model
inventory lists them.

#### Scenario: Bounded confidence clusters
- **WHEN** the Deffuant model runs with confidence bound 0.2 from uniform opinions
- **THEN** opinions converge to a few clusters, about `1 / (2 * 0.2)` of them

### Requirement: Engineering model collection
The engineering packages SHALL provide circuits, control, traffic, and power-system models, as the model inventory
lists them.

#### Scenario: Optimal velocity jams
- **WHEN** the optimal velocity model of cars on a ring runs above its critical density from uniform flow
- **THEN** a stop-and-go wave forms and moves against the direction of travel

### Requirement: Mathematics model collection
The mathematics packages SHALL provide dynamical systems, stochastic processes, network processes, and fractal models,
as the model inventory lists them.

#### Scenario: Standard map
- **WHEN** the Chirikov standard map runs above `K = 0.971635` from an orbit near the golden-mean torus
- **THEN** the orbit diffuses in momentum, and below that value it stays bounded

### Requirement: Computer science model collection
The computer science packages SHALL provide automata, evolutionary computation, and neural network models, as the model
inventory lists them.

#### Scenario: Random Boolean network phases
- **WHEN** Kauffman networks with connectivity 1 and 3 run from random states
- **THEN** perturbations die out at connectivity 1 and spread at connectivity 3

### Requirement: Quantum spin chain models
The library SHALL provide the quantum Heisenberg model (Mihailović et al. 2023 §7.3) under
`physics/statistical_mechanics`: a chain of spins one half with the XXZ Hamiltonian
`J Σ (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j) - h Σ S^z_i` over neighbouring pairs, optional dephasing at a rate
`gamma`, and an initial Néel state, a single flipped spin, a domain wall, or the ground state.

#### Scenario: Ground state of the ring
- **WHEN** the spectrum of `quantum_heisenberg` with 4 periodic sites, `J = 1`, `Δ = 1`, and `h = 0` is computed
- **THEN** its lowest eigenvalue is -2 within 1e-12

#### Scenario: Single magnon spreads as a Bessel light cone
- **WHEN** `quantum_heisenberg` runs with 12 periodic sites, `J = 1`, `Δ = 0`, and one flipped spin at site 6, to
  time 2
- **THEN** at every frame the probability `1/2 - <S^z_{6+d}>` of finding the flip at distance `d` equals
  `J_d(J t)²` within 1e-5, and the total magnetisation stays constant within 1e-8

#### Scenario: Dephasing destroys coherence
- **WHEN** `quantum_heisenberg` runs with 6 periodic sites from a Néel state with `gamma = 0.1`
- **THEN** the solver is the Lindblad solver, the purity never increases from frame to frame and ends below 1, and
  the total magnetisation stays constant within 1e-8

### Requirement: Operator formulation of the Schrödinger model
The `schrodinger` model SHALL accept a `formulation` of `'fields'`, the default, which keeps its two real fields, or
`'operators'`, which builds a particle on the same lattice as a single-particle Hilbert space with the kinetic operator
of the lattice's spectral Laplacian and the harmonic potential as a diagonal operator, and accepts position dephasing
at a rate `gamma`. Both formulations SHALL register under the one name `schrodinger` in `physics/waves`.

#### Scenario: Formulations agree
- **WHEN** `schrodinger` runs with 128 sites to time 4 once with each formulation
- **THEN** the probability of every site, `(u² + w²) h` of the fields with the spacing `h` and `|psi|²` of the
  operators, agrees within 1e-6 at every frame

#### Scenario: Dephased packet
- **WHEN** `schrodinger` runs with the operator formulation and `gamma = 0.5`
- **THEN** the trace stays 1 within 1e-10 and the purity never increases from frame to frame

### Requirement: Quantum models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as out of scope or blocked by this change SHALL register under the name of
its entry. A model that the library already implements SHALL gain the new formulation under the same name, not be
registered twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `quantum_heisenberg` in `physics/statistical_mechanics` (Mi 7.3)
  - `schrodinger` in `physics/waves` (Sc 4), with the operator formulation

#### Scenario: Gallery example of a quantum model
- **WHEN** the gallery runs the example `docs/examples/physics/statistical_mechanics/plot_quantum_heisenberg.py`
- **THEN** it creates `quantum_heisenberg` by name, streams the run to Rerun within the recording budget, its
  thumbnail is the space-time image of the local magnetisation with a light cone, and its analysis figure compares the
  ground-state energy per site with `1/4 - ln 2`

### Requirement: Economic and social agent models
The library SHALL provide registered agent models that exercise the phases and the core's matching, feedback, choice,
and evolution interactions in distinct ways, and SHALL cover spatial cooperation with the existing lattice model of the
spatial prisoner's dilemma rather than registering it again.

#### Scenario: Spatial cooperation
- **WHEN** the spatial prisoner's dilemma runs with imitate-the-best updating and a temptation payoff of 1.6
- **THEN** cooperators survive in clusters for 200 steps instead of dying out

### Requirement: Agent economy models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `credit_networks` in `economics/macroeconomics`
  - `fractional_reserve_banking` in `economics/macroeconomics`
  - `double_auction` in `economics/markets`
  - `labour_market` in `economics/markets`
  - `fair_food` in `economics/markets`
  - `sugarscape` in `economics/markets`
  - `brock_hommes` in `economics/markets`
  - `kinetic_wealth_exchange` in `economics/econophysics`
  - `minority_game` in `economics/econophysics`
  - `el_farol` in `economics/econophysics`
  - `kolkata_restaurant` in `economics/econophysics`
  - `axelrod` in `social/opinion_dynamics`

#### Scenario: Example entries left pending
- **WHEN** the change is applied
- **THEN** these entries, which repeat patterns the implemented models already exercise, stay pending this change:
  - `goods_traders` in `economics/markets`
  - `shops` in `economics/markets`
  - `barter` in `economics/markets`
  - `order_book` in `economics/finance`
  - `tragedy_of_the_commons` in `economics/resources`
  - `ridesharing` in `engineering/traffic`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: System-dynamics and stock-flow consistent models
The library SHALL provide Meadows' systems zoo (bathtub, coffee cooling, thermostat, inventory with delays, renewable
and nonrenewable resources), the system traps (success to the successful, shifting the burden, escalation), Bass
diffusion, the beer game, and a stock-flow consistent monetary economy as registered factories, with `bathtub` and
`car_dealer` rebuilt on flows under their names. The tragedy of the commons belongs to `add-agent-economies`.

#### Scenario: Inventory oscillation
- **WHEN** the car-dealership inventory model receives a 10% step in demand
- **THEN** the inventory oscillates before settling, and shortening the perception delay reduces the oscillation's
  amplitude

### Requirement: Stock-flow models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries, which probe flows, sources and sinks, auxiliaries, lookups, delays, smoothing, loops, and
  discrete flows in distinct ways, are registered under their field paths and marked implemented:
  - `stock_flow_consistent_economy` in `economics/macroeconomics`
  - `commodity_cycle` in `economics/markets`
  - `beer_game` in `economics/markets`
  - `bass_diffusion` in `economics/innovation`
  - `success_to_the_successful` in `social/organizations`
  - `shifting_the_burden` in `social/organizations`
  - `escalation` in `social/organizations`
  - `rice_milling` in `engineering/processes`
  - `solid_waste` in `engineering/processes`
  - `food_relief` in `engineering/processes`
  - `irrigation` in `engineering/processes`

#### Scenario: Repeated patterns registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries, which repeat patterns of stocks and flows that the models above probe, are registered under
  their field paths and marked implemented: `energy_return` in `physics/climate`, and `cocoa`, `hilsa_fishery`, and
  `food_security` in `economics/resources`

#### Scenario: Meadows' resource models rebuilt
- **WHEN** the change is applied
- **THEN** Meadows' nonrenewable and renewable resource models stay registered as `oil_economy` and
  `fishery_capital`, rebuilt on flows, and the duplicate survey entries `renewable_resource` and
  `nonrenewable_resource` are removed from the inventory

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Every registered model has a gallery example
Every model registered in the library SHALL ship with a gallery example and a reviewed screenshot. A change that adds or
rebuilds a model SHALL add or update its example in the same change, and a model whose screenshot cannot show its
behaviour within the gallery budget SHALL say so in its caption and show the closest faithful view.

#### Scenario: New model without an example
- **WHEN** a change registers a new model and adds no example
- **THEN** the gallery coverage check fails and names the model

#### Scenario: Behaviour beyond the budget
- **WHEN** a model's known behaviour, such as a 100 thousand year glacial cycle, needs a longer run than the budget
- **THEN** its example runs it with coarser frames within the budget, and its caption states what the image shows

### Requirement: The chemical Langevin equation switches by noise
`chemical_langevin_equation` with its defaults SHALL switch between its two stable states within 200 time units.

#### Scenario: Noise-driven switching
- **WHEN** the default chemical Langevin equation runs until 200 with Euler-Maruyama steps of 1e-3 from seed 0
- **THEN** after time 20 the number of molecules is below 30 at some frame and above 90 at another

### Requirement: The hodgepodge machine forms spirals
`hodgepodge` with its defaults SHALL keep travelling waves on a 100 × 100 lattice.

#### Scenario: Waves persist
- **WHEN** the default hodgepodge machine on a 100 × 100 lattice runs for 300 steps
- **THEN** more than 10 % of the cells are ill at the last step

### Requirement: Social impact flips under net pressure
`social_impact` SHALL flip an opinion when the net pressure to change plus the random field is positive.

#### Scenario: No pressure keeps opinions
- **WHEN** a social impact model without noise starts with every agent of opinion -1
- **THEN** no opinion changes in 5 steps

### Requirement: Model defaults show the described behaviour
`restricted_three_body`, `diffusion_in_a_speckle_field`, and `legion` with their defaults SHALL show chaos, anomalous
diffusion, and alternating objects within short runs, and `burridge_knopoff` SHALL register a solver that does not
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

### Requirement: Specific realizations live in the model library
The library SHALL hold every specific realization of a core primitive or a general analysis in its own private module
of the model package that uses it, next to the model definitions, exported from that package. A realization that
several packages of one field share SHALL live in `skcomplex.models.<field>.common`, grouped in family packages such as
`particles`, `oscillators`, `excitable`, or `forecasting`, with no loose modules beside the families, and only that
field's packages import it. A realization SHALL be a subclass of a general core class, even a pure renaming, and no field
SHALL define the same realization twice. A field whose packages share no code SHALL have no `common`
package. A model class whose mechanism the core provides SHALL use the core class and keep only what is its own.

#### Scenario: Order book from the library
- **WHEN** the finance models build their order book, and an engineering model pairs riders with cars
- **THEN** both use the core `MatchingInteraction`, the order book binds it to bids, asks, money, and goods with
  `ContinuousMatchingSpec`, and the earlier of two same-limit sell orders trades first

#### Scenario: Potential from the library
- **WHEN** a user imports the Lennard-Jones potential from `skcomplex.models.physics.common.particles`
- **THEN** it is a realization built on the core `PairCouplingInteraction`, and the existing LAMMPS agreement tests pass
  with it

#### Scenario: Realization of one package
- **WHEN** only the climate models use `EarlyWarningAnalysis`
- **THEN** it lives in `skcomplex.models.physics.climate`

#### Scenario: No copy across fields
- **WHEN** the ecology meadow and the economics commons models are built
- **THEN** both build foraging and regrowth from the core `TransferInteraction`, `LifeCycleInteraction`, and
  `MapInteraction`

### Requirement: Library models keep their behaviour across the redesign
Every library model SHALL produce the same trajectory on fixed seeds after migrating to the new vocabulary, within the
tolerances of the conformance suite.

#### Scenario: Conformance before and after
- **WHEN** the conformance suite runs every catalog model before and after the migration with random state 0
- **THEN** every trajectory agrees within its declared tolerance

### Requirement: Echo state network model
The library SHALL provide `echo_state_network` in `computer_science/neural_networks`, a reservoir of leaky tanh units driven by an input, with a curated gallery example.

#### Scenario: Echo state property
- **WHEN** `echo_state_network` runs from two initial states with spectral radius 0.9
- **THEN** the two reservoir trajectories converge, which is the echo state property

### Requirement: Chemical Langevin equations of reaction networks
The chemistry module SHALL describe binding and dissociation reactions as chemical Langevin equations, in which one
Wiener process per reaction drives every species the reaction changes, and SHALL provide the chemical Langevin
equation of the enzyme network `E + S <-> C -> E + P` as a registered factory.

#### Scenario: Binding conserves a difference
- **WHEN** `A + B -> C` runs as a chemical Langevin equation with `EulerMaruyama`
- **THEN** `A - B` stays at its initial value within 1e-9 relative error at every frame, while `A` fluctuates

#### Scenario: Enzyme totals under noise
- **WHEN** the chemical Langevin network runs for its default time with a fixed random state
- **THEN** `E + C` and `S + C + P` stay at their initial values within 1e-9 relative error at every frame

#### Scenario: Two descriptions at once
- **WHEN** a binding reaction is both stochastic and in Langevin mode
- **THEN** compiling raises a named error that names the interaction

### Requirement: Correlated geometric Brownian motions
The finance module SHALL provide two assets whose prices follow geometric Brownian motions driven by Wiener processes
with a given correlation, as a registered factory.

#### Scenario: Correlated returns
- **WHEN** the model runs with correlation 0.7 over 10 000 frames
- **THEN** the sample correlation of the two assets' log returns per frame is 0.7 within 0.05

### Requirement: Correlated noise models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `chemical_langevin_network` in `chemistry/kinetics`
  - `correlated_geometric_brownian_motion` in `economics/finance`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Fractional models
The library SHALL provide fractional relaxation, a fractional damped oscillator, fractional Van der Pol, and fractional
Lorenz as registered factories with the order as an argument.

#### Scenario: Slower than exponential
- **WHEN** fractional relaxation runs with orders 1 and 0.5 from the same state
- **THEN** the order 0.5 trajectory stays above the exponential one at long times

### Requirement: Fractional dynamics models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `fractional_relaxation` in `mathematics/fractional_dynamics`
  - `fractional_oscillator` in `mathematics/fractional_dynamics`
  - `fractional_van_der_pol` in `mathematics/fractional_dynamics`
  - `fractional_lorenz` in `mathematics/fractional_dynamics`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Repeated agent economy models
The library SHALL provide `goods_traders`, `shops`, `barter`, `order_book`, `tragedy_of_the_commons`, and
`ridesharing` as registered factories, each with a curated gallery example. Each SHALL be a `System` under discrete
time with a phases schedule, with agents as elements with rule laws in groups, built on the core's matching, transfer,
and invariant laws and on the trading realizations of `skcomplex.models.economics.realizations`, and none SHALL add a
core primitive.

#### Scenario: Barter reaches the contract curve
- **WHEN** `barter` runs Edgeworth exchange between two traders with Cobb-Douglas preferences until no trade improves
  both
- **THEN** the final allocation lies on the contract curve within 2% of the total endowment

#### Scenario: Commons collapse
- **WHEN** `tragedy_of_the_commons` runs with more herders than the meadow carries and no regulating adaptive parameter
- **THEN** the grass falls below a tenth of its capacity and the herd shrinks after it

#### Scenario: Trades conserve goods
- **WHEN** `goods_traders` runs for 50 steps with an invariant on the total of each good
- **THEN** no invariant is violated, since every trade is a transfer between two agents

### Requirement: Evolutionary Sherrington-Kirkpatrick game
The library SHALL provide `evolutionary_sk_game` in `physics/criticality`, after Lesmana et al. (2026, arXiv
2604.15669): learning agents on a Sherrington-Kirkpatrick landscape with Bak-Sneppen replacement of the least fit,
built from the core's group, reduction, draw, and update primitives under discrete time, with the extremal replacement
an update switched by the group minimum, and with its softmax choice defined in its model package rather than imported
from another field's realizations.

#### Scenario: Rules of the game
- **WHEN** a user reads the factory of `evolutionary_sk_game`
- **THEN** couplings are symmetric Gaussian with variance `1/N`, each agent's incentive is updated from its local field
  with a memory weight, actions are softmax draws at the agent's temperature, and the least fit agent is replaced with a
  temperature redrawn uniformly on [0, 2]

#### Scenario: Self-organisation to the edge of ergodicity breaking
- **WHEN** `evolutionary_sk_game` runs with 256 agents and memory 0.75 past its transient
- **THEN** the replacement cascades follow a power law with exponent within 0.15 of 3/2, and the mean reward exceeds the
  reward of the same game run at every fixed temperature tried in a sweep

#### Scenario: Example of the evolutionary game
- **WHEN** a user opens `docs/examples/physics/criticality/plot_evolutionary_sk_game.py`
- **THEN** its thumbnail shows the evolved temperature distribution and its analysis compares the cascade sizes and the
  ergodicity-breaking diagnostic with the paper's results

### Requirement: Emergence models are recorded in the model inventory
The inventory SHALL record `evolutionary_sk_game` with its arXiv source, its primitives, and its priority, pending this
change until it is implemented.

#### Scenario: Entry of the evolutionary game
- **WHEN** a contributor reads the inventory entry of `evolutionary_sk_game`
- **THEN** it cites arXiv 2604.15669, is tagged with discrete time, no space, a group, no binding, and the discrete
  law class, and has priority `core`

### Requirement: Kinetic models of active particles
The library SHALL provide wealth and welfare dynamics, criminality against security forces, tumour-immune
competition, and opinion formation as registered kinetic factories built on site groups and encounter laws, with their
transition tables in their model packages and their standard moments declared by their catalog entries.

#### Scenario: Immune response
- **WHEN** the tumour-immune model runs with an immune activity above its threshold
- **THEN** the tumour density decays, and below the threshold it grows

### Requirement: Kinetic models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `immune_competition` in `biology/oncology`
  - `kinetic_wealth_welfare` in `economics/econophysics`
  - `kinetic_opinion` in `social/opinion_dynamics`
  - `evacuation` in `social/crowds`
  - `criminality` in `social/crowds`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Control models
The library SHALL provide optimal harvesting, SIR vaccination, Ramsey optimal growth, a PID-stabilised inverted
pendulum, and a linear-quadratic regulated oscillator as registered factories.

#### Scenario: Vaccination lowers the peak
- **WHEN** the SIR vaccination problem is solved with a bounded vaccination rate and a cost on infections
- **THEN** the optimally controlled epidemic peak is lower than the uncontrolled one

### Requirement: Feedback controllers are engineering realizations
The library SHALL provide PID and state-feedback controllers in `skcomplex.models.engineering.control`, the package of
the control models that use them, as components whose laws read states and set bounded adaptive parameters during a
run, with the integral of the PID error in an element, so their internal states are part of the compiled system and
the core holds no controller.

#### Scenario: Inverted pendulum
- **WHEN** a cart with an inverted pendulum runs from a tilt of 0.1 radians with a tuned PID controller
- **THEN** the angle returns within 0.01 radians of upright within 5 seconds

#### Scenario: No controller in the core
- **WHEN** the field-term and import-graph tests of the core run
- **THEN** they pass, and the controllers are importable only from the control models

### Requirement: Optimal control models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `sir_vaccination` in `biology/epidemiology`
  - `optimal_consumption` in `economics/macroeconomics`
  - `optimal_harvesting` in `economics/resources`
  - `pid_inverted_pendulum` in `engineering/control`
  - `lqr_oscillator` in `engineering/control`
  - `optimal_chemotherapy` in `biology/oncology`
  - `optimal_immunotherapy` in `biology/oncology`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Group automata models
The library SHALL provide the majority automaton on the free group, the lamplighter automaton, the discrete Laplacian
as a linear automaton, the reversible Q2R Ising automaton, the golden mean shift, and the majority automaton on the
Grigorchuk group as registered factories, each with a gallery example whose thumbnail is its Rerun screenshot and
which shows an analysis plot from its run.

#### Scenario: Q2R conserves energy
- **WHEN** the Q2R automaton runs on a periodic lattice
- **THEN** the Ising energy of the configuration stays constant at every step

#### Scenario: Grigorchuk automaton
- **WHEN** a user creates `grigorchuk_automaton` by name and runs it with its default parameters
- **THEN** it runs the majority rule with the memory set `{1, a, b, c, d}` on a ball of radius 8 with 271 sites, every
  interior cell takes the majority value of itself and its four neighbours, and the space reports the ball sizes 1, 5,
  11, 23, 40, 68, 108, 176, and 271 for radii 0 to 8

#### Scenario: Grigorchuk gallery plot
- **WHEN** a user opens the gallery example of `grigorchuk_automaton`
- **THEN** below the Rerun screenshot of the Cayley ball it plots the ball sizes for radii 0 to 10, and the text states
  that the local log-log slope rises while `log |B(n)| / n` falls, so the growth is faster than polynomial and slower
  than exponential

### Requirement: Group cellular automaton models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `free_group_automaton` in `computer_science/automata`
  - `lamplighter_automaton` in `computer_science/automata`
  - `heisenberg_automaton` in `computer_science/automata`
  - `baumslag_solitar_automaton` in `computer_science/automata`
  - `linear_ring_automaton` in `computer_science/automata`
  - `subshifts` in `computer_science/automata`
  - `grigorchuk_automaton` in `computer_science/automata` (CC 6.9)

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Surveyed physics models
The library SHALL provide `swarmalators`, `ehrenfest_urn`, `directed_percolation_langevin`, `sk_game`,
`keller_segel`, `benard_marangoni`, `charged_schrodinger`, `cahn_hilliard`, and `allen_cahn` as
registered factories built from core primitives, each with a curated gallery example.

#### Scenario: Ehrenfest urn relaxes to the binomial
- **WHEN** `ehrenfest_urn` runs from all balls in one urn
- **THEN** the count in the first urn relaxes to a binomial distribution with mean half the balls

#### Scenario: Directed percolation threshold
- **WHEN** `directed_percolation_langevin` sweeps its control parameter across the absorbing transition
- **THEN** the stationary density vanishes below a threshold and grows above it

### Requirement: Surveyed biology models
The library SHALL provide `hormone_cascade_amplifier`, `min_protein_skeleton`, `nanotube_gap_junction_exchange`,
`closed_contour_cell_exchange`, `adaptive_response_costs`, `glucose_dual_hormone_control`,
`coagulation_positive_feedback`, and `couzin_leadership` as registered factories built from core primitives, each with
a curated gallery example.

#### Scenario: Informed minority leads the group
- **WHEN** `couzin_leadership` runs with a small informed minority sharing a preferred direction
- **THEN** the group's mean heading turns towards that direction, and the accuracy grows with the minority's size

### Requirement: Surveyed earth, economics, and mathematics models
The library SHALL provide `sutera_fraedrich_tipping`, `ramped_stochastic_fold`, `metaphoria`,
`wilhite_trade_networks`, `koopman_slow_manifold`, and `ou_driven_rossler` as registered factories built from core
primitives, each with a curated gallery example.

#### Scenario: Tipping lags a ramp
- **WHEN** `ramped_stochastic_fold` ramps its control parameter through the fold faster than the system relaxes
- **THEN** the state tips after the control passes the static fold point, later for faster ramps

### Requirement: Network models
The library SHALL provide Kuramoto on complete graphs and networks, the power-grid swing equation, SIS and SIR on
networks, adaptive SIS with rewiring, Barabási-Albert growth, voter and majority models on networks, random Boolean
networks, coupled logistic maps, Vicsek flocking, and DebtRank interbank contagion as registered factories.

#### Scenario: Random Boolean network phases
- **WHEN** random Boolean networks with in-degree 1, 2, and 3 run from random states
- **THEN** damage from a single flipped node dies out at in-degree 1, stays near constant at 2, and spreads at 3

### Requirement: Adaptive network models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `kuramoto_on_networks` in `physics/synchronization`
  - `coupled_map_lattices_on_networks` in `physics/synchronization`
  - `vicsek` in `physics/active_matter`
  - `gregoire_chate` in `physics/active_matter`
  - `ginelli_nematics` in `physics/active_matter`
  - `sequentially_linear_ecosystem` in `biology/ecology`
  - `quenched_mean_field_sis` in `biology/epidemiology`
  - `adaptive_sis` in `biology/epidemiology`
  - `activity_driven_epidemics` in `biology/epidemiology`
  - `jain_krishna` in `biology/evolution`
  - `sole_manrubia` in `biology/evolution`
  - `hegselmann_krause` in `social/opinion_dynamics`
  - `coevolving_voter` in `social/opinion_dynamics`
  - `join_a_club` in `social/opinion_dynamics`
  - `barabasi_albert` in `mathematics/networks`
  - `holme_kim` in `mathematics/networks`
  - `vertex_copying` in `mathematics/networks`
  - `network_attack` in `mathematics/networks`
  - `random_graph_ensembles` in `mathematics/networks`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path

### Requirement: Dynamics analysis probe models
The library SHALL provide `kinouchi_copelli`, `nonreciprocal_cahn_hilliard`, `nonreciprocal_vicsek`, and `tipping_cascade_network`, each with a curated gallery example.

#### Scenario: Maximal dynamic range
- **WHEN** `kinouchi_copelli` sweeps the branching ratio
- **THEN** the dynamic range peaks at a branching ratio of 1, as Kinouchi and Copelli (2006) found

#### Scenario: Traveling patterns
- **WHEN** `nonreciprocal_cahn_hilliard` runs with nonreciprocal coupling above threshold
- **THEN** its demixed domains travel, as Brauns and Marchetti (2023) predict

### Requirement: Field measures with logic of their own live with their models
A measure named by one field that only picks a choice of a general analysis, such as the Binder cumulant or the polar
order, SHALL have no class of its own: models, examples, and tests SHALL use the general analysis. A field measure with
logic of its own, such as the susceptibility peak or the crackling-noise relation, SHALL be a subclass of the closest
general analysis in the model package whose models it measures.

#### Scenario: Binder cumulant without a class
- **WHEN** a user computes the Binder cumulant of magnetisation samples
- **THEN** they run `DistributionAnalysis('moment-ratio')` with orders 4 and 2, and no model package defines a class for
  it

#### Scenario: Crackling noise with the sandpiles
- **WHEN** a user checks the crackling-noise relation of a sandpile's avalanches
- **THEN** they run `CracklingNoiseAnalysis` from `skcomplex.models.physics.criticality`, a subclass of the event
  analysis

### Requirement: Specialized code moves to the model library
When a library package is generalised, its specialized code that still means something in a field, such as a field's
measure, a field's naming, or a composite of general operations, SHALL move to the model library as realizations of the
general classes, and SHALL NOT be removed. The model library SHALL serve as worked examples of how to extend the library
packages by subclassing their bases and general classes.

#### Scenario: Crackling noise kept
- **WHEN** the avalanche analysis dissolves into the general event analysis
- **THEN** the crackling-noise relation stays available as a realization of the criticality models, with the same
  values as before

### Requirement: Higher-order probe models
The library SHALL provide the probe models of higher-order interactions that the inventory records as pending or
blocked by this change, each with a curated gallery example, built on the classes and reductions this change adds.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their names and field paths and marked implemented:
  - `hypergraph_kuramoto` in `physics/synchronization`
  - `hypergraph_consensus` in `social/opinion_dynamics`
  - `simplicial_social_dilemmas` and `hypergraph_public_goods` in `social/cooperation`
  - `simplicial_contagion` and `hypergraph_exposure_sis` in `social/spreading`
  - `triadic_percolation`, `hypergraph_random_walk`, and `growing_simplicial_complex` in `mathematics/networks`

#### Scenario: Explosive synchronization
- **WHEN** `hypergraph_kuramoto` sweeps the three-body coupling up and down
- **THEN** the order parameter shows hysteresis, as Majhi et al. (2022) review

#### Scenario: Bistable contagion
- **WHEN** `simplicial_contagion` runs with a triangle infection rate above the critical value
- **THEN** the epidemic has two stable prevalences depending on the initial seed

### Requirement: Group mechanisms run as symbolic groupings
`galam_majority` and `public_goods_game` SHALL be built on `GroupingInteraction(form='symbolic')`, and SHALL agree with
their previous trajectories over 20 seeds. Kolkata and the minority game SHALL keep their agent forms.

#### Scenario: Galam on the symbolic grouping
- **WHEN** `galam_majority` runs from a share of 0.6 for 20 steps over 20 seeds
- **THEN** the mean final share agrees with the previous agent form within the standard error of the 20 seeds

### Requirement: Hamiltonian mechanics models
The library SHALL build the Kepler problem `two_body_orbit`, the restricted three-body problem, and Hénon-Heiles on
`HamiltonianInteraction`, and their catalog entries SHALL pass the Hamiltonian as an observable with an invariant
tolerance, so the `InvariantObserver` checks its drift.

#### Scenario: Kepler orbit
- **WHEN** `two_body_orbit` runs for ten periods of a bound orbit with `Yoshida4Solver`
- **THEN** the orbit closes on itself within 1e-6 relative error and the energy drift stays below 1e-8

#### Scenario: Hénon-Heiles below the escape energy
- **WHEN** `henon_heiles` runs from an energy of 1/12 to time 1000
- **THEN** the orbit stays bounded and the energy stays within its invariant tolerance

### Requirement: Variational mechanics models are recorded in the model inventory
Every model this change rebuilds SHALL keep its name and field path in the model inventory, and SHALL list the
`variational-mechanics` primitive.

#### Scenario: Inventory models rebuilt
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are rebuilt on `HamiltonianInteraction` under their names and field paths, list the primitive
  `variational-mechanics`, and are marked implemented:
  - `two_body_orbit` in `physics/celestial_mechanics`
  - `restricted_three_body` in `physics/celestial_mechanics`
  - `henon_heiles` in `physics/nonlinear_oscillators`

### Requirement: Elman simple recurrent network
The library SHALL provide Elman's simple recurrent network (Gros 2015 §10.5.1, Eq. 10.18, and Elman 1990) as a
registered factory of a system under discrete time in epochs. It SHALL nest the discrete map
`s(t) = σ(A s(t-1) + D x(t) + b)`, `y(t) = σ(W s(t) + c)` with adaptive shaped parameters, and train it with a
`TrainingInteraction` on the squared error of predicting the next input, by default on Elman's temporal XOR stream.

#### Scenario: Temporal XOR
- **WHEN** `run('elman_network', settings={'until': 200, 'random_state': 0})` trains 4 hidden units with Adam on a
  stream of 3000 bits
- **THEN** after 200 epochs the metric `xor_error`, the mean squared error of the predictions of every third bit, is
  below 0.05, the metric `random_error` of the predictions of the other bits stays above 0.2, and
  `training/loss` at the last frame is below its value at frame 0

#### Scenario: Elman's original rule
- **WHEN** `elman_network` is created with `truncation=1`
- **THEN** it trains with the previous hidden states as constant context inputs, and `xor_error` still falls below
  `random_error` within 500 epochs

#### Scenario: Gallery example of the Elman network
- **WHEN** the gallery runs `docs/examples/computer_science/neural_networks/plot_elman_network.py`
- **THEN** it records at most 300 frames, its thumbnail is the time series view of the loss and the two metrics, in
  which `xor_error` falls toward 0 while `random_error` stays near 0.25, and below the screenshot a Matplotlib figure
  shows the mean squared prediction error at each position of the bit triple before and after training, with the
  third position alone near 0 after training

### Requirement: The generalized Lotka-Volterra matrix is a parameter
`generalized_lotka_volterra` SHALL hold its interaction matrix as a shaped parameter of a `CouplingInteraction`, so a
schedule or an experiment changes it, and SHALL keep its trajectory.

#### Scenario: Rebuilt community keeps its trajectory
- **WHEN** `generalized_lotka_volterra` runs with its catalog defaults
- **THEN** its trajectory equals the trajectory of the previous constant matrix within 1e-10

#### Scenario: Matrix swept by an experiment
- **WHEN** a grid experiment sets the matrix parameter to two matrices that differ off the diagonal
- **THEN** the two members of the experiment differ in their final abundances

### Requirement: Random Lotka-Volterra community with k-body interactions
The library SHALL provide `higher_order_random_lotka_volterra` in `biology/ecology` (Sidhom and Galla 2024, arXiv
2409.10990): a group of species with the growth `x_i (k - x_i)` and the rate `x_i sum_(j<k) a_ijk x_j x_k` of the
paper's Eq. 2, written `x_i contraction(a, x, x) / 2`, from a random coupling tensor `a` of order 3 with the mean, variance,
and correlation of its Eqs. 4 and 5, a shaped parameter drawn by the factory from its `random_state`, so a schedule or an
experiment changes it, with a curated gallery example.

#### Scenario: No interactions
- **WHEN** the model runs with a tensor of zeros
- **THEN** every species converges to its carrying capacity within 1e-6

#### Scenario: Random community
- **WHEN** the model runs with its defaults, the equations, scaling, and parameters of Sidhom and Galla (2024, arXiv
  2409.10990), over 20 seeds
- **THEN** every abundance stays non-negative and finite, and the mean fraction of surviving species matches the
  paper's result at those parameters within its stated precision: 200 species, `mu = -4`, `sigma = 1`, and `gamma = 0`
  give 0.928 from its Eq. 17, and the paper states no number, so the check allows 0.02, three standard errors of 20 seeds

### Requirement: Learning models are recorded in the model inventory
Every model this change implements SHALL register under the field path of its subfield and SHALL have an entry in the
model inventory, and a model the inventory records as blocked by this change SHALL register under the name of its
entry.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `elman_network` in `computer_science/neural_networks`, with the sources Gr 10.5.1 and Elman 1990
  - `higher_order_random_lotka_volterra` in `biology/ecology`

### Requirement: Nonlocal field models
The library SHALL provide, as registered factories built on the general interactions and the nonlocal operators:
- `amari_neural_field` under `biology/neuroscience` (2112.12147 §1.3.1);
- `double_gyre` under `physics/oceanography` (1910.00583 §III.6.1);
- `gkt_traffic` under `engineering/traffic` (cond-mat/0012229 §III.E.4);
- `active_viscoelastic_gel` under `physics/active_matter` (2306.08868 VI).

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
- **WHEN** the gas-kinetic-based traffic model runs on a ring road with a small local density perturbation, once at a
  density of 15 vehicles per kilometre and once at a density in its unstable range
- **THEN** the perturbation decays at the low density, and at the unstable density it grows into congested clusters
  whose downstream fronts move upstream at between 10 and 25 kilometres per hour

#### Scenario: Contractile gel
- **WHEN** the active viscoelastic gel runs from a small perturbation of uniform cell density above its contractile
  threshold
- **THEN** the density concentrates into peaks, and the velocity at every step solves the force balance within 1e-10

### Requirement: Field couplings are interactions
`legion` and `nanotube_gap_junction_exchange` SHALL write the coupling of their fields with `CouplingInteraction`, so
`TermanWangElement` and `ExchangingCellElement` hold only the dynamics of one site, and SHALL keep their trajectories.

#### Scenario: Rebuilt couplings keep the trajectories
- **WHEN** `legion` and `nanotube_gap_junction_exchange` run with their catalog defaults
- **THEN** the trajectory of `nanotube_gap_junction_exchange` equals that of the previous elements within 1e-10, and the
  rates of `legion` equal those of the previous elements within 1e-12 at many random states, with its trajectories
  equal within 1e-10 over a short window before rounding grows

### Requirement: Kinetic equations of gases and plasmas
The library SHALL provide the spatially inhomogeneous Boltzmann equation, its BGK model, the Vlasov-Poisson equation
with Landau damping and two-stream set-ups, and the local Enskog equation as registered factories under
`physics/fluids` (Bellomo et al. 2017 §2.2–2.3), built on `PhaseSpace`, `AdvectionInteraction`,
`EncounterInteraction`, and `RelaxationInteraction`. The inventory SHALL record them as implemented.

#### Scenario: Sod shock tube near the fluid limit
- **WHEN** the BGK model runs the Sod shock tube in one position and one velocity dimension at a Knudsen number of 1e-4
- **THEN** the density moment at time 0.2 matches the exact Euler solution within 2% in the L1 norm

#### Scenario: Landau damping model
- **WHEN** the Vlasov-Poisson factory runs with its default Landau damping set-up
- **THEN** the measured damping rate of the electric energy lies within 5% of 0.1533, and the model passes the
  conformance suite

### Requirement: Kinetic theory of flocking
The library SHALL provide the Bertin-Droz-Grégoire kinetic theory of flocking over position and orientation as a
registered factory under `physics/active_matter` (Marchetti et al. 2013, arXiv 1207.2929 §V.2.3), whose angular
diffusion and binary alignment are `EncounterInteraction` subclasses on a periodic class axis. The inventory SHALL
record it as implemented.

#### Scenario: Flocking onset
- **WHEN** the model runs from a homogeneous disordered state with a small polar perturbation, at densities 20% below
  and 20% above the transition density that Bertin, Droz, and Grégoire give in closed form for the chosen noise
- **THEN** the polarization moment decays below the transition and grows above it

### Requirement: Static models
The library SHALL provide `lq_cell_survival` in `biology/oncology`, `random_energy_model` and `continuum_percolation`
in `physics/statistical_mechanics`, `oligopoly_equilibria` in `economics/markets`, `von_thunen` in
`economics/resources`, `gravity_urban_scaling` in `social/organizations`, and `mandelbrot_least_effort` in
`social/language`, as registered factories of systems under `StaticTime` that run through `create` and `run` and sweep
their parameters through `Experiment`.

#### Scenario: Fractionation
- **WHEN** `lq_cell_survival` evaluates 30 fractions of 2 Gy with `alpha/beta = 10`
- **THEN** the biologically effective dose is 72 Gy (Padmanabhan et al. 2021 Eq. 6.8), and the dose-protraction factor
  of a constant dose rate tends to 1 as the delivery time tends to 0

#### Scenario: Glass transition
- **WHEN** an `Experiment` sweeps the temperature of `random_energy_model` with `N = 20` over 10 replicates
- **THEN** the entropy follows `ln 2 - 1/(4 T^2)` above `T_g = 0.6006` and stays below 0.1 at `T = 0.3`

#### Scenario: Mixed price equilibrium
- **WHEN** `oligopoly_equilibria` solves the pricing game of `m` firms with loyal share `alpha` and switchers
  `beta = 1 - m alpha`
- **THEN** the symmetric equilibrium price distribution is `F(P) = 1 - (alpha (1 - P)/(beta P))^(1/(m - 1))` on
  `[alpha/(alpha + beta), 1]`, and each firm's expected profit is `alpha` (Forrest 2018 Lemma 20.1)

#### Scenario: Von Thünen rings
- **WHEN** `von_thunen` assigns the cells of a plain around one market to the crop of highest bid rent
  `Y_i (p_i - c_i) - Y_i f_i d` at distance `d`, after Giordano (2024, arXiv 2407.02181 §7)
- **THEN** the land use forms concentric rings whose boundaries lie where the bid rents of neighbouring crops are equal,
  within one cell, and raising a crop's transport rate `f_i` shrinks its ring

#### Scenario: Percolation threshold
- **WHEN** an `Experiment` sweeps the filling factor of `continuum_percolation` over 0.8 to 1.5 for two sizes
- **THEN** the spanning probabilities that `ClusterAnalysis` gives cross between filling factors 1.0 and 1.25

#### Scenario: Urban scaling exponent
- **WHEN** an `Experiment` runs `gravity_urban_scaling` at five populations from 100 to 1600 with area exponent `alpha`
  and interaction exponent `gamma < 2`, and `ScalingAnalysis(kind='power')` fits the total interaction against the
  population
- **THEN** the fitted exponent is within 0.05 of `2 - alpha gamma / 2`, for one seed up to `gamma = 1` and for totals
  averaged over seeds at `gamma = 1.5`, where close pairs make single cities heavy-tailed

#### Scenario: Least-effort frequencies
- **WHEN** `mandelbrot_least_effort` solves for word costs `c_i = log(i + 1)`
- **THEN** the probabilities follow `(i + 1)^-lambda` with the solved `lambda` within 1e-9, and the ratio of cost to
  entropy at the solved `lambda` is lower than at `lambda ± 0.1`

### Requirement: Dynamic equilibrium, satisfiability, and search models
The library SHALL provide the fictitious play of `oligopoly_equilibria` with `dynamics='fictitious-play'`, built on
`ChoiceInteraction` and `ReinforcementInteraction`, `perceptron_satisfiability` in `computer_science/neural_networks`,
built on `SatisfiabilityInteraction`, and `evolutionary_optimisers` in `computer_science/evolutionary_computation`,
built on `OptimizationInteraction` and on `SamplingInteraction` with `cooling`, all under discrete time.

#### Scenario: Fictitious play of a discretised duopoly
- **WHEN** `oligopoly_equilibria` runs with `dynamics='fictitious-play'` for 2000 rounds
- **THEN** the empirical distribution of prices is within 0.2 of `F(P)` in the Kolmogorov distance

#### Scenario: Satisfiable fraction
- **WHEN** an `Experiment` runs `perceptron_satisfiability` at `N = 15` and `N = 21` over 20 replicates
- **THEN** the share of satisfiable replicates against `p/N` falls from above 0.9 at 0.5 to below 0.1 at 1.2

#### Scenario: Algorithms compared
- **WHEN** `evolutionary_optimisers` runs differential evolution, particle swarm, and annealing on Rastrigin for 100
  generations with the same seed
- **THEN** every run reports a best value that never increases, and differential evolution ends below 1e-3

### Requirement: Static models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL be marked implemented in
the model inventory, with a gallery example whose analysis plot comes from an `Experiment` over its swept parameter.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `lq_cell_survival` in `biology/oncology`
  - `random_energy_model` in `physics/statistical_mechanics`
  - `oligopoly_equilibria` in `economics/markets`
  - `von_thunen` in `economics/resources`
  - `perceptron_satisfiability` in `computer_science/neural_networks`
  - `evolutionary_optimisers` in `computer_science/evolutionary_computation`
  - `continuum_percolation` in `physics/statistical_mechanics`
  - `gravity_urban_scaling` in `social/organizations`
  - `mandelbrot_least_effort` in `social/language`

### Requirement: Dry active nematics
The library SHALL provide `dry_active_nematics` under the field path `physics/active_matter` as a registered factory
built only from public names, a `FieldElement` on a lattice under continuous time: a conserved density coupled to a
symmetric traceless alignment tensor, with an active current along the divergence of the tensor (1207.2929 §II.2). The
inventory SHALL list it as implemented.

#### Scenario: Dry active nematic conserves density
- **WHEN** `dry_active_nematics` runs on a periodic lattice for 100 time units
- **THEN** the total density stays constant within 1e-10 relative error

### Requirement: Active nematic and polar gels
The library SHALL provide `active_gels` under the field path `physics/active_matter` as a registered factory: a
symmetric traceless tensor, with `order='nematic'`, or a polarisation vector, with `order='polar'`, transported by its
co-rotational derivative in an overdamped Stokes flow driven by the elastic and active stresses (1207.2929 §III), with
the Landau-de Gennes energy in its own package. The inventory SHALL list it as implemented.

#### Scenario: Passive nematic order
- **WHEN** `active_gels` runs with `order='nematic'` at zero activity from a uniform tensor with Landau coefficients
  `a < 0` and `c`
- **THEN** the velocity stays below 1e-8, and the scalar order relaxes to `sqrt(-2 a / c)` within 1e-6

#### Scenario: Active turbulence conserves topological charge
- **WHEN** the extensile nematic gel runs on a periodic 2D lattice from a nearly aligned state with an activity well
  above the bend instability threshold
- **THEN** `DefectAnalysis` finds pairs of `+1/2` and `-1/2` defects, the total charge stays 0 at every frame, and the
  mean kinetic energy settles at a nonzero value

#### Scenario: Spontaneous flow of a polar film
- **WHEN** the polar gel runs in a film between walls of fixed polarisation, given by the `symmetry` setting, at
  activities 20% below and 20% above the threshold of the linear stability analysis
- **THEN** the flow decays below the threshold and grows to a steady nonzero flow above it

### Requirement: Acausal component library
The library SHALL provide the electrical connector and components with ports and equations in
`skcomplex.models.engineering.common.electrical`, and the thermal connector and components in
`skcomplex.models.engineering.control`.

#### Scenario: Connector from the library
- **WHEN** a user imports the electrical connector from `skcomplex.models.engineering.common.electrical`
- **THEN** it is a `ConnectorSpec`, and no module of the core names it

#### Scenario: Component Chua circuit
- **WHEN** Chua's circuit is assembled from the electrical components with the values of `chua`
- **THEN** its capacitor voltages follow the trajectory of `chua` within the tolerance of its test

### Requirement: Engineering models on components and Lagrangians
The library SHALL provide `heated_room` in `engineering/control`, and `dc_motor` and `quarter_car` in
`engineering/mechatronics`, as registered factories, each with the standard observables of its catalog entry.
`quarter_car` SHALL be built on `SpringDamperInteraction`, and the inertia of `dc_motor` on `EulerLagrangeInteraction`.

#### Scenario: Heated room
- **WHEN** `heated_room` runs from a cold room with the thermostat set to 20 degrees and a hysteresis band of 1 degree
- **THEN** the room temperature rises to the band and then stays within it, with the heater switching on and off

#### Scenario: Quarter car over a step
- **WHEN** `quarter_car` drives over a road step with its standard values
- **THEN** the body and wheel displacements settle at the step height, and the energy of the springs decays

### Requirement: Acausal component models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, tagged with the time kind, space kind, component kinds, interaction bindings, and law classes of the system
it builds.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `heated_room` in `engineering/control`
  - `quarter_car` in `engineering/mechatronics`
  - `dc_motor` in `engineering/mechatronics`

### Requirement: Lattice gauge model samples the Wilson action
The library SHALL provide `lattice_gauge_fields` in `physics/waves`, after Schwichtenberg (2020 §2.1, §4.4). With its
default Monte Carlo dynamics, it SHALL sample the Wilson action of U(1) or Z_N link angles on the cell complex of a
periodic lattice under discrete time, through `WilsonSamplingInteraction`, a subclass of `SamplingInteraction` in its
package, and its catalog entry SHALL declare the mean plaquette as an observable.

#### Scenario: Plaquette against the exact result
- **WHEN** `lattice_gauge_fields` runs with its defaults, U(1) on a periodic 32 × 32 lattice at β = 2, after its
  transient
- **THEN** the mean plaquette observable is within 0.01 of `I_1(2) / I_0(2) ≈ 0.698`

#### Scenario: Gauge-invariant run
- **WHEN** a random gauge transformation is applied to the initial links before a Monte Carlo run with a fixed seed
- **THEN** the mean plaquette series equals the series of the untransformed run within 1e-12

### Requirement: Lattice gauge model evolves Maxwell fields
With `dynamics='hamiltonian'`, `lattice_gauge_fields` SHALL return a system under continuous time that evolves a lattice
Maxwell potential and electric field on links from optional static charges on sites. Its catalog entry SHALL declare
the Gauss residual and the field energy as observables.

#### Scenario: Maxwell dynamics conserve the Gauss law
- **WHEN** `lattice_gauge_fields` runs with Hamiltonian dynamics and a static charge pair on a periodic 32 × 32 lattice
- **THEN** the Gauss residual stays below 1e-10 at every frame, and the field energy changes by less than 1e-3
  relative error

### Requirement: Gauge observables are measured in the gauge package
The package `skcomplex.models.physics.waves` SHALL hold `GaugeAnalysis`, a subclass of `BaseAnalysis` that measures
Wilson loops of given extents, Polyakov loops along the last axis, the topological charge of a 2D U(1) link state, and
the Gauss residual of an electric link state against a site charge. No public name of `skcomplex.core` SHALL contain
`wilson`, `gauge`, or `polyakov`.

#### Scenario: Invariance of loops
- **WHEN** a random gauge transformation is applied to a U(1) configuration
- **THEN** the mean plaquette and the 2 × 2 Wilson loop are unchanged within 1e-12

#### Scenario: Integer charge
- **WHEN** the topological charge of any U(1) configuration on a periodic 2D lattice is computed
- **THEN** it equals an integer within 1e-9

### Requirement: Topological Kuramoto model
The library SHALL provide `topological_kuramoto` in `physics/synchronization`, after Millán, Torres, and Bianconi
(2203.06601 III): phases on the nodes and links of a clique complex, the link phases driven through the codifferential
and the exterior derivative of the sines of the phases, with node and link dynamics coupled by their global order
parameters.

#### Scenario: Link synchronization grows with coupling
- **WHEN** `topological_kuramoto` runs on the clique complex of a random graph at zero coupling and at a coupling well
  above its transition
- **THEN** the order parameter of the link phases at the end of the strong-coupling run exceeds that of the run at zero
  coupling by at least 0.3

### Requirement: Gauge arbitrage market model
The library SHALL provide `gauge_arbitrage_market` in `economics/finance`, after Schwichtenberg (2020 §1.7, §2.1,
§2.2.1): exchange-rate potentials on the links of a lattice complex evolved by the discrete inhomogeneous Maxwell
equations, sourced by the trader current on links, and the good on sites moved along the links by its covariant
gradient.

#### Scenario: Conservation in the market
- **WHEN** `gauge_arbitrage_market` runs with its defaults
- **THEN** the total good stays constant within 1e-10 relative error, and the Gauss residual against the density of the
  good stays below 1e-8

### Requirement: Gauge network agents
The library SHALL provide `gauge_network_agents` in `social/opinion_dynamics`, after Mihaylov and Cacciatori (2025,
arXiv 2509.01581): agent phases on the nodes of a network and connections on its edges, the agents relaxing towards
their neighbours' transported phases through `TransportedCouplingInteraction`, a subclass of `CouplingInteraction` in
its package, so frustrated loops prevent global agreement.

#### Scenario: Frustration blocks agreement
- **WHEN** `gauge_network_agents` runs on a ring of 12 agents with a flat connection and with a total flux of pi around
  the ring
- **THEN** the flat ring reaches agreement with zero energy, and the frustrated ring keeps a residual energy and a twist

### Requirement: Link field models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL be marked implemented in
the model inventory.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `lattice_gauge_fields` in `physics/waves`
  - `topological_kuramoto` in `physics/synchronization`
  - `gauge_arbitrage_market` in `economics/finance`
  - `gauge_network_agents` in `social/opinion_dynamics`

### Requirement: Agent models in three-dimensional spaces and on region layers
The library SHALL provide, as registered factories built on `MovementInteraction` and agent-form interactions:
- boids in three dimensions under `biology/collective_behaviour` (Think Complexity ch. 10),
- the Kauai population sentiment model on region layers under `social/organizations` (Carmichael et al. 2019 ch. 3),
  with a small synthetic island dataset in its package.

#### Scenario: Boids align in three dimensions
- **WHEN** 50 boids with random headings fly in a periodic three-sided `ContinuousSpace` with only the alignment rule
  for 200 steps
- **THEN** the polarization, the norm of the mean unit velocity, rises from below 0.3 to above 0.9, and the third
  coordinates of the boids still spread over more than half of the box depth

#### Scenario: Population follows the census layer
- **WHEN** the Kauai model is created from the synthetic island's census layer with populations 100, 50, and 0
- **THEN** it has 150 resident agents, each with a home point inside its census region

#### Scenario: Positive event raises sentiment
- **WHEN** the Kauai model runs with a positive event on a topic that half of the residents weight above zero
- **THEN** the mean sentiment of those residents rises and stays within -1 and 1, and the others' mean changes only
  through interaction
