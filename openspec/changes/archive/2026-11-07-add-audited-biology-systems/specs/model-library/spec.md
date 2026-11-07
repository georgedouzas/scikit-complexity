# Spec Delta

## ADDED Requirements

### Requirement: Audited biology models
The library SHALL provide the biology models of the reference audit, listed in the proposal, as registered
factories in the biology subfields built from the general classes of the core, each with a curated gallery example and
an inventory entry that cites its source section.

#### Scenario: Consumers coexist where their isoclines cross
- **WHEN** `consumer_resource` runs with the settings of Mehta and Blumenthal (2024, figure 4)
- **THEN** the resources settle at `(5/3, 5/9)`, where both zero net growth isoclines cross

#### Scenario: Single-resource competition of Gros
- **WHEN** `consumer_resource` runs with one supplied resource without dilution
- **THEN** the total population tends to `a / d`, the resource to `d / max r`, and only the fastest consumer survives

#### Scenario: Ricker period doubling
- **WHEN** `ricker` runs with `r = 5` and with `r = 9`
- **THEN** it settles on `ln r` in the first case and on a two-cycle in the second

#### Scenario: Doomsday of the zombie outbreak
- **WHEN** `zombie_outbreak` runs without births, deterministically or stochastically
- **THEN** the total is conserved and the humans vanish

#### Scenario: Record dynamics of outbreaks
- **WHEN** `record_dynamics_epidemic` runs for 100 years over many seeds
- **THEN** the mean number of outbreak years is close to the harmonic number of the strains drawn

#### Scenario: Gamma rhythm of the neural mass
- **WHEN** `next_generation_neural_mass` runs unforced with `H_e = 10`
- **THEN** the excitatory rate oscillates between 40 and 50 Hz

#### Scenario: Transient attractors of the dHAN
- **WHEN** `associative_thought_process` runs on the seven-site network of Gros (2015)
- **THEN** every lasting winning coalition is a stored clique and at least four cliques are visited

#### Scenario: Kleiber scaling of the distribution network
- **WHEN** `west_brown_enquist` is solved for growing depths
- **THEN** the blood volume and the capillaries follow the sums of the branching rules and the exponent rises towards 3/4

#### Scenario: Kimura fixation probability
- **WHEN** `kimura_diffusion` runs with many replicates until loss or fixation
- **THEN** the share of fixed replicates is `(1 - exp(-N s x)) / (1 - exp(-N s))`

#### Scenario: Critical noise of the Boolean network
- **WHEN** `noisy_boolean_network` runs with three inputs per node
- **THEN** it orders below the mean-field noise `1/6` and disorders above it

#### Scenario: Adaptation of the enzyme circuit
- **WHEN** `enzyme_adaptation_circuit` receives a step of its input
- **THEN** its output has a sensitivity above 1 and a precision above 10

#### Scenario: Ants heal a disruption
- **WHEN** `ant_colony_optimization` cuts its best path midway
- **THEN** the colony settles near the shortest path before the cut and again after it

#### Scenario: Shepherd brings the flock home
- **WHEN** `shepherding` runs from a scattered flock
- **THEN** the centre of the flock reaches the goal
