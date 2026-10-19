# Spec Delta

## ADDED Requirements

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
