# Spec Delta

## ADDED Requirements

### Requirement: Audited maps and flows of dynamical systems
The library SHALL provide in `mathematics/dynamical_systems` the Fermi-Ulam map `fermi_ulam_map` (Datseris and Parlitz
Ex 1.11), the Nosé-Hoover oscillator `nose_hoover` (Ex 2.11), Sprott's mixed flow `sprott_mixed_flow` (Ex 3.17), and,
after Dilão, the CDN map `cdn_map`, the Lozi map `lozi_map`, the conservative Hénon map `conservative_henon_map`, the
Langford and Thomas flows `langford` and `thomas`, the hyperchaotic Rössler flow `hyperchaotic_rossler`, and the Ikeda
delay equation `ikeda`.

#### Scenario: Hyperchaos
- **WHEN** the Lyapunov spectrum of `hyperchaotic_rossler` is computed over 2000 time units
- **THEN** its two largest exponents are near 0.11 and 0.02

#### Scenario: Mixed state space
- **WHEN** `sprott_mixed_flow` runs from `(1, 0, 0)` and from `(2, 0, 0)`
- **THEN** the mean volume rate `2(y + z)` is near 0 on the first orbit and below -0.1 on the second

#### Scenario: Delay bifurcation
- **WHEN** `ikeda` runs with `a tau` below and above `π / 2`
- **THEN** it settles at `π` in the first case and oscillates about it in the second

### Requirement: Audited stochastic processes and fractals
The library SHALL provide `cascading_sample_space_reducing` (Thurner et al. 3.3.5.4), `random_branch_map` (Dilão Ex
5.10), and `exponential_growth_at_random_times` (Thurner et al. 3.3.6.1) in `mathematics/stochastic_processes`, and
`inverse_julia` (Dilão Ex 5.8) in `mathematics/fractals`.

#### Scenario: Any exponent
- **WHEN** cascades that multiply by `mu` run on 100 states
- **THEN** their visits fall as `i^(-mu)`

#### Scenario: Power law from two exponentials
- **WHEN** growth at the rate `mu` is observed at the rate `lam`
- **THEN** the Hill estimate of the tail of the values is `lam / mu`

#### Scenario: Julia set of the squaring map
- **WHEN** `inverse_julia` runs with `omega = 0`
- **THEN** every point lies on the unit circle

### Requirement: Audited network systems
The library SHALL provide `graphlet_arrival` (Raman 3.2.4.2), `kronecker_graph` (Gros, chapter 1),
`configuration_model` (Jensen 8.5.1 and Thurner et al. 4.5.1), and `interdependent_network_cascade` (Artime et al.
section 5) in `mathematics/networks`, and `barabasi_albert(fitness=True)` SHALL attach new edges in proportion to
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

### Requirement: Audited mathematics systems are recorded in the inventory
The inventory SHALL record every new system as implemented with the sections of its sources, and the gallery SHALL
hold its example.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for the new systems
