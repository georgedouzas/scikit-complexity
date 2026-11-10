## ADDED Requirements

### Requirement: Dilemma on a dynamic network
The library SHALL provide `DynamicNetworkDilemmaSystem` in `social/cooperation` (Su, McAvoy, and Plotkin 2023, arXiv
2301.11982): the donation game with benefit `b` and cost `c` under weak selection and death-birth updates on a network
whose structure switches among given networks by a Markov chain, and the function `critical_benefit_cost_ratio`, which
returns the critical ratio `(b/c)*` of the paper for given networks and transition probabilities.

#### Scenario: Dynamic pair of communities
- **WHEN** a user computes the critical ratio of the two networks of dense and sparse communities of Figure 2 of the
  paper with community size `a = 0.5` and mean duration `t = 1` for `N = 1000`
- **THEN** it lies within 0.2 of 7

#### Scenario: Static network favours spite
- **WHEN** a user computes the critical ratio of one of those networks alone for `N = 100`
- **THEN** it is negative

#### Scenario: Simulated fixation agrees with the ratio
- **WHEN** the system runs a small dynamic pair under weak selection to fixation over many seeds at a ratio above the
  critical ratio of the function and at one below it
- **THEN** cooperators fix with a probability above `1 / N` at the first ratio and below it at the second, outside the
  sampling error

### Requirement: Competing gossip groups
The library SHALL provide `GossipGroupNormsSystem` in `social/cooperation` (Kessinger, Tarnita, and Plotkin 2023, arXiv
2204.10811): gossip groups that hold one view of every individual on the edges of a complete network, donors that
help a recipient their group views as good, observers that assess every donor by the norm of their group as a lookup
table of action and recipient view, errors of action `u_x` and of assessment `u_a`, and group sizes that grow with the
payoffs of their members.

#### Scenario: Norm tables
- **WHEN** a user reads the norms of the system
- **THEN** Stern Judging gives a good view to cooperating with good and defecting against bad, and a bad view
  otherwise, and Scoring gives a good view to cooperating alone

#### Scenario: Stern Judging overtakes Scoring
- **WHEN** a group with Stern Judging at 20% of the population competes with a group with Scoring at 80%, with `b = 5`,
  `c = 1`, and both errors 0.02
- **THEN** the Stern Judging group exceeds 90% of the population by the end of the run in at least 9 of 10 seeds

### Requirement: Public goods with peer punishment on regular graphs
The library SHALL provide `GraphPunishmentGameSystem` in `social/cooperation` (Wang, Perc, and Szolnoki 2024, arXiv
2401.11686): cooperators, defectors, and peer punishers on a random regular graph of degree `k`, each playing the
public goods game with synergy `r` in the group of every node and its neighbours, punishers paying `alpha` per defecting
co-player and defectors paying `beta` per punishing co-player, with pairwise comparison updates.

#### Scenario: Two strategies match the well-mixed condition
- **WHEN** cooperators and defectors alone run on a random regular graph of degree 4 under weak selection, with `r`
  above and below `k + 1`
- **THEN** the share of cooperators grows above the threshold and falls below it, as the paper's equation predicts

#### Scenario: Threshold of punishment
- **WHEN** the system runs from a state with most nodes defecting, with `r < k + 1`, and with the fine `beta` above and
  below the threshold `beta*` of equation 6 of the paper
- **THEN** defection dies out above the threshold and persists below it

### Requirement: The cooperation systems are recorded in the inventory
The inventory SHALL record `dynamic_network_dilemma`, `gossip_group_norms`, and `graph_punishment_game` under
`social.cooperation` as implemented with their sources and primitives, and the gallery SHALL hold an example of each.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for these systems
