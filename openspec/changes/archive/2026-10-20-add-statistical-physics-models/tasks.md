# Tasks

## 1. Inventory

- [x] 1.1 Record the source `Bt`, every model of Bertin 2021 with its status, and the new sources of existing models in
  the inventory, and verify the inventory check passes

## 2. Model groups

For every task in this section:
- implement each model as a factory on shared components of its subfield package, built only from public core names;
- verify its scenario and a behavioural check per model;
- add its gallery example with a Rerun thumbnail and an analysis plot, as `add-model-gallery` requires;
- mark its inventory entry implemented.

- [x] 2.1 Mean-field and disordered spin systems: curie_weiss, mattis_spin_glass, edwards_anderson,
  sherrington_kirkpatrick
- [x] 2.2 Driven and noisy particles: colored_noise_langevin, on_off_intermittency, biased_trap_walk,
  tilted_periodic_potential, flashing_ratchet, shaken_spring_block
- [x] 2.3 Birth-death and interacting particle processes: birth_death_queue, absorbing_birth_death, zero_range_process,
  dissipative_zero_range, asep
- [x] 2.4 Active matter and traffic: aligning_active_brownian_particles, binary_collision_flocking, langevin_traffic,
  traffic_density_equation
- [x] 2.5 Agents, evolution, and populations: perishable_goods_duopoly, standardized_agents, origin_fixation_walk,
  neutral_biodiversity, neutral_clustering, heterogeneous_mean_field_sir
- [x] 2.6 Maps and coupled oscillators: cubic_chaotic_map, skew_tent_map, doubling_map, subcritical_hopf_form,
  oscillator_death, heterogeneous_global_coupling

## 3. Gate

- [x] 3.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
