# Spec Delta

## ADDED Requirements

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
