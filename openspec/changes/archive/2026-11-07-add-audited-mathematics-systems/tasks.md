# Tasks

## 1. Dynamical systems

- [x] 1.1 Add the maps `fermi_ulam_map`, `lozi_map`, `conservative_henon_map`, and `cdn_map`, verified by
  `tests/systems/mathematics/dynamical_systems/test_maps.py`
- [x] 1.2 Add the flows `nose_hoover`, `sprott_mixed_flow`, `langford`, `thomas`, `hyperchaotic_rossler`, and the
  delay equation `ikeda`, verified by `tests/systems/mathematics/dynamical_systems/test_flows.py`

## 2. Stochastic processes and fractals

- [x] 2.1 Add `cascading_sample_space_reducing`, `random_branch_map`, and `exponential_growth_at_random_times`,
  verified by the tests of `_reducing`, `_random_maps`, and `_power_laws`
- [x] 2.2 Add `inverse_julia`, verified by `tests/systems/mathematics/fractals/test_quadratic.py`

## 3. Networks

- [x] 3.1 Add the option `fitness` of `barabasi_albert` and `graphlet_arrival`, verified by
  `tests/systems/mathematics/networks/test_growth.py`
- [x] 3.2 Add `kronecker_graph`, `configuration_model`, and `interdependent_network_cascade`, verified by
  `tests/systems/mathematics/networks/test_ensembles.py`
- [x] 3.3 Record the audited network models the core cannot express in the design

## 4. Documentation

- [x] 4.1 Add the inventory entries and the gallery examples, verified by the inventory check and the strict gallery
  coverage
