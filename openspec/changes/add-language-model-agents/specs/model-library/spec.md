## ADDED Requirements

### Requirement: Naming game of language-model agents
The library SHALL provide `LlmNamingGameSystem` in `social/language` (Flint Ashery, Aiello, and Baronchelli 2024, arXiv
2410.08948): pairs of agents drawn at random each name an option from a pool shuffled per player, both gain a reward on
a match and lose on a mismatch, every agent keeps a memory of its last `H` rounds, and `committed` agents always name
one option. The player SHALL be an oracle or the rule-based player that names the option of largest summed reward in
memory.

#### Scenario: Consensus of rule-based players
- **WHEN** the system runs with the rule-based player, no committed agents, and 20 seeds
- **THEN** in every run one name reaches a share of at least 0.9 within 1000 rounds

#### Scenario: Committed minority above the tipping size
- **WHEN** a population of rule-based players in consensus on one name receives committed agents of another name at a
  share of 0.4 of the population
- **THEN** the other name reaches a share of at least 0.9 within 2000 rounds

#### Scenario: Defaults of the paper
- **WHEN** a user builds the system with its defaults
- **THEN** it has 24 agents, a pool of 10 names, a memory of 5 rounds, the reward 100, and the penalty 50

#### Scenario: Prompt of the oracle player
- **WHEN** the oracle player of an agent with an empty memory is queried
- **THEN** the request holds the rules, the reward and the penalty, and the `W` names in an order that differs between
  two players of the same round for a seed

### Requirement: Learned players of the naming game
`LlmNamingGameSystem` SHALL declare its decisions as the actions of an environment of `add-learning-foundation` and
SHALL take `player='learned'`, a policy shared by every agent over its memory, trained by `train_policy`.

#### Scenario: Learned players reach consensus
- **WHEN** the learned player, a policy shared by every agent over its memory, is trained by `train_policy` with
  `'reinforce'` and `random_state=0`, and the trained system runs with 20 seeds
- **THEN** in most runs one name reaches a share of at least 0.9 within 1000 rounds

### Requirement: Network growth by language-model agents
The library SHALL provide `LlmNetworkGrowthSystem` in `mathematics/networks` (De Marzo, Pietronero, and Garcia 2023,
arXiv 2312.06619): nodes with random names arrive one at a time and link to `m` existing nodes, chosen by an oracle
from the list of names, with or without their degrees, or by the preferential rule of Barabási and Albert.

#### Scenario: Preferential player
- **WHEN** the system grows 2000 nodes with `m = 2` and the preferential player
- **THEN** the exponent of the power-law fit of the degree distribution lies between 2.5 and 3.5

#### Scenario: Requests with and without degrees
- **WHEN** the oracle player is queried with `show_degrees=True` and with `False`
- **THEN** the first request lists every existing name with its degree and the second lists the names alone

### Requirement: Group guessing game
The library SHALL provide `GroupGuessingGameSystem` in `social/organizations` (Riedl 2025, arXiv 2510.05174): `N`
agents guess integers in a range, the group hears whether the sum of the guesses is above, below, or equal to a hidden
target, and the game ends on a hit or after a number of rounds. The player SHALL be an oracle with the prompts of the
plain, persona, and theory-of-mind conditions of the paper, or the feedback player that moves its guess by a share of
the signed error with noise.

#### Scenario: Feedback players find the target
- **WHEN** ten feedback players play 200 games with targets drawn in the range of the paper
- **THEN** the share of games that hit the target within the round limit exceeds 0.5

#### Scenario: Prompts of the conditions
- **WHEN** the oracle player is queried in the persona condition and in the theory-of-mind condition
- **THEN** the first request holds the persona of the agent and the second asks the agent to reason about the others

### Requirement: The language-model systems are recorded in the inventory
The inventory SHALL record `llm_naming_game` under `social.language`, `llm_network_growth` under `mathematics.networks`,
and `group_guessing_game` under `social.organizations` as implemented with their sources and the primitive
`oracle-queries`, and the gallery SHALL hold an example of each that replays recorded queries labelled with the model,
the endpoint, the date, and the seed.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for these systems
