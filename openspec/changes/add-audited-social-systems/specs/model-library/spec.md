# Spec Delta

## ADDED Requirements

### Requirement: Audited cooperation systems
The library SHALL provide in `social/cooperation` Argun's evolutionary iterated prisoner's dilemma on a lattice
`iterated_lattice_dilemma` (Argun et al. 13.2 and 13.3), Cooney's two-level replicator equation `two_level_replicator`
with the additively separable pairwise group conflict as its option `victory` (arXiv 2405.18419 Eq. 3.3 and section 5),
the finite two-level birth-death process `two_level_birth_death` (section 3.1 and appendix A.1), and evolutionary
dynamics with game transitions `game_transitions` (arXiv 1905.10269).

#### Scenario: Best reply in the iterated dilemma
- **WHEN** the years in prison of every strategy `n` against a strategy `m` of a ten-round game are computed
- **THEN** the fewest years come from `n = m - 1`

#### Scenario: Three regimes of two strategies
- **WHEN** `iterated_lattice_dilemma` with only the extreme strategies runs at `R = 0.82`, `0.84`, and `0.86`
- **THEN** cooperation dominates, coexists with defection, and vanishes

#### Scenario: Threshold of group selection
- **WHEN** `two_level_replicator` with `G(z) = z` runs from a uniform density above and below `lambda* = c + q - p`
- **THEN** the mean fraction of punishers tends to `1 - (c + q - p) / lambda` above it and to 0 below it

#### Scenario: Transitions favour cooperation
- **WHEN** `game_transitions` runs under pairwise comparison with `b1 - b2 > 2 c`, with and without transitions
- **THEN** cooperators take over with transitions and die out with game 1 alone

### Requirement: Audited opinion dynamics systems
The library SHALL provide in `social/opinion_dynamics` the three-state opinion network `three_state_opinion` and its
mean-field flows `undecided_flows` (Jensen 11.5, Project 1), the majority rule with willingness `willing_majority`
(Jensen 11.5, Exercise 4), and the echo chamber model `echo_chamber_radicalization` (arXiv 1906.12325 Eqs. 1 and 2).

#### Scenario: Critical size of the undecided
- **WHEN** `undecided_flows` with flows driven by the undecided runs above and below `N_c = omega (1 / gamma + 1 / eta)`
- **THEN** `N - N_c` agents stay undecided above it and none below it

#### Scenario: Zealots
- **WHEN** `willing_majority` runs with every agent willing, with zealots of both opinions, and with zealots of one
- **THEN** the first reaches consensus, the second does not, and the third ends at the zealots' opinion

#### Scenario: Echo chambers
- **WHEN** `echo_chamber_radicalization` runs at low controversy, without homophily, and with homophily
- **THEN** opinions reach the neutral consensus, radicalize to one side, and split into two opposite groups

### Requirement: Audited crowd, segregation, and language systems
The library SHALL provide Bellomo's kinetic crowd in a street with walls and panic `kinetic_crowd` in `social/crowds`
(Bellomo et al. 6.4 and 6.5), distinct from `evacuation`, the Big Sort as the option `choices` of `schelling` (Downey,
Exercise 9-1), and Miller's random typewriting `random_typewriting` in `social/language` (Thurner et al. 3.3.6.2).

#### Scenario: Fundamental diagram
- **WHEN** `kinetic_crowd` runs with one cell over the street
- **THEN** its mean speed follows Eq. 6.30 of the density within ten percent

#### Scenario: Big Sort
- **WHEN** `schelling` runs with one, four, and ten choices
- **THEN** the like-neighbour share stays near its start with one choice and grows with the number of choices

#### Scenario: Zipf's law of random typing
- **WHEN** `random_typewriting` types with five letters and a space bar of probability 0.2
- **THEN** words of one more letter are 6.25 times rarer and the rank-frequency exponent is near `1 - log(0.8) / log 5`

### Requirement: Audited organization systems
The library SHALL provide in `social/organizations` Chesney's Ajani's Choice `ajanis_choice` (Chesney 3.3 and 3.8) and
Namrata's Family `supply_chain_diffusion` (4.9 and 4.10), Meadows' watchmakers `watchmakers` (chapter 3), policy
resistance `policy_resistance`, and drift to low performance `drift_to_low_performance` (chapter 5), and the
exploration and preferential return `exploration_preferential_return` and container model `container_mobility` of
human mobility (arXiv 2109.07381).

#### Scenario: Underpayment needs plentiful workers
- **WHEN** `ajanis_choice` runs with more workers than jobs and with fewer
- **THEN** workers are paid below the minimum wage only in the first case

#### Scenario: Power in supply chains
- **WHEN** a buyer with two suppliers and a buyer with one adopt good working conditions
- **THEN** the suppliers whose outflow is below the buyer's inflow adopt, and a single supplier never does

#### Scenario: Hierarchy of the watchmakers
- **WHEN** `watchmakers` runs for 20,000 actions with an interruption probability of 1%
- **THEN** Hora finishes a watch about every 1173 actions and Tempus none

#### Scenario: Eroding goals
- **WHEN** `drift_to_low_performance` runs with eroding goals, absolute standards, and an upbeat bias
- **THEN** the performance drifts down, holds, and drifts up

#### Scenario: Scales of mobility
- **WHEN** `container_mobility` runs with the probabilities of travel 0.6, 0.3, and 0.1
- **THEN** the displacements happen at the three levels with these shares and grow with the level

### Requirement: Audited social systems are recorded in the inventory
The inventory SHALL record every new system as implemented with the sections of its sources, and the gallery SHALL
hold its example.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for the new systems
