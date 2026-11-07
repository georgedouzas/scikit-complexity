## Why

The reference audit listed mathematics models of Dilão, Datseris and Parlitz, Gros, Thurner et al., Jensen, Raman, and
the reviewed papers that the current core supports but the catalog lacks. Each one is a known result the catalog can
reproduce and a worked example of the core primitives.

## What Changes

- Add the maps `fermi_ulam_map`, `lozi_map`, `conservative_henon_map`, and `cdn_map`, the flows `nose_hoover`,
  `sprott_mixed_flow`, `langford`, `thomas`, and `hyperchaotic_rossler`, and the delay equation `ikeda` to
  `mathematics/dynamical_systems`.
- Add `cascading_sample_space_reducing`, `random_branch_map`, and `exponential_growth_at_random_times` to
  `mathematics/stochastic_processes`, and `inverse_julia` to `mathematics/fractals`.
- Add `graphlet_arrival`, `kronecker_graph`, `configuration_model`, and `interdependent_network_cascade` to
  `mathematics/networks`, and the option `fitness` of `barabasi_albert` for the Bianconi-Barabási model.
- Record every new system in the inventory with its sources and add its gallery example.
- Leave out the audited network models whose rules the core cannot express, listed in the design.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-library`: new systems of dynamical systems, stochastic processes, fractals, and networks, and the fitness
  option of Barabási-Albert growth.

## Impact

New modules under `src/skcomplex/systems/mathematics`, their tests under `tests/systems/mathematics`,
`docs/overview/inventory.toml`, and examples under `docs/examples/mathematics`. No change to the core.
