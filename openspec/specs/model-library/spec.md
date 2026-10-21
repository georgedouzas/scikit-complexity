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
The physics module SHALL provide Lennard-Jones and soft-sphere pair interactions with a cutoff, which run on the native
solvers and on the LAMMPS backend.

#### Scenario: Lennard-Jones minimum
- **WHEN** two members interact through a Lennard-Jones potential with `sigma = 1`
- **THEN** the force between them is zero at the distance `2 ** (1 / 6)`

### Requirement: Langevin thermostat
The physics module SHALL provide a Langevin thermostat that adds drag and random forces to a particle group, which runs
on the native SDE solvers and on the LAMMPS backend.

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
The library SHALL organize models into packages per scientific field, `physics`, `chemistry`, `biology`, `earth`,
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
The library SHALL provide registered agent models that exercise the phases, market protocols, institutions, and
learning blocks in distinct ways, and SHALL cover spatial cooperation with the existing lattice model of the spatial
prisoner's dilemma rather than registering it again.

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
  their field paths and marked implemented: `energy_return` in `earth/climate`, and `cocoa`, `hilsa_fishery`, and
  `food_security` in `economics/resources`

#### Scenario: Meadows' resource models rebuilt
- **WHEN** the change is applied
- **THEN** Meadows' nonrenewable and renewable resource models stay registered as `oil_economy` and
  `fishery_capital`, rebuilt on flows, and the duplicate survey entries `renewable_resource` and
  `nonrenewable_resource` are removed from the inventory

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
