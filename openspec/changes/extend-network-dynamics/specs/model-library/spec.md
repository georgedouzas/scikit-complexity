# Spec Delta

## ADDED Requirements

### Requirement: Interacting urns model on a network
The library SHALL provide `interacting_urns_network` in `economics/innovation`, after Iacopini et al. (2020, Eqs. 1 and
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
The library SHALL provide `tap_catalysis` in `economics/innovation`, after Hordijk and Kauffman (2022, Algorithm 1):
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
The library SHALL provide `motter_lai_overload` in `mathematics/networks`, after Motter and Lai (2002) as Artime et al.
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
The inventory SHALL record `interacting_urns_network`, `tap_catalysis`, and `motter_lai_overload` as implemented with
their sources and the primitives `member-sets`, `hyperedge-requests`, and `path-reductions`, and the gallery SHALL hold
their examples.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for the three systems
