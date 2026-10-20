# Spec Delta

## ADDED Requirements

### Requirement: Stochastic map models
The library SHALL provide, as registered factories built only on the public core, the persistent random walk under
`mathematics/stochastic_processes` (Jensen 2023 §10.4), Markov and hidden Markov sources under
`mathematics/stochastic_processes` (Lindgren 2024 §3.2), and iterated function systems drawn by the chaos game under
`mathematics/fractals` (Dilão 2023 §5.4). The inventory SHALL mark the three entries as implemented.

#### Scenario: Persistent random walk diffuses faster
- **WHEN** 2000 persistent random walkers that keep their direction with probability 0.8 run for 1000 steps
- **THEN** the mean squared displacement divided by the time is `(1 + ρ) / (1 - ρ) = 4` within 10%, where
  `ρ = 2 × 0.8 - 1`

#### Scenario: Markov source entropy rate
- **WHEN** a Markov source with transition matrix `[[0.9, 0.1], [0.4, 0.6]]` runs for 100000 steps
- **THEN** the entropy rate of its symbols estimated from block entropies is 0.569 bits within 0.01

#### Scenario: Hidden Markov source frequencies
- **WHEN** a hidden Markov source with hidden transition matrix `[[0.9, 0.1], [0.4, 0.6]]` and emission matrix
  `[[0.7, 0.3], [0.2, 0.8]]` runs for 100000 steps
- **THEN** the frequency of the first observed symbol is 0.6 within 0.01

#### Scenario: Sierpiński triangle by the chaos game
- **WHEN** a group of 100000 points iterates the three maps of the Sierpiński triangle with equal weights for 30 steps
- **THEN** the box-counting dimension of the final points, rasterised on a 128 by 128 grid with box sizes from 1 to 32
  cells, is `log 3 / log 2` within 0.05

### Requirement: Probabilistic Boolean networks
The library SHALL provide `probabilistic_boolean_network` in `biology/systems_biology`, a Boolean network that draws
each node's transfer function from its candidates with given probabilities at every step, as Raman 2021 section 7.3.1
poses it.

#### Scenario: Selection probabilities
- **WHEN** `probabilistic_boolean_network` runs for many steps with one node choosing between two functions with
  probabilities 0.7 and 0.3
- **THEN** the share of steps that used the first function is 0.7 within three standard errors

### Requirement: Bertin models on stochastic maps
The library SHALL provide `random_walk_with_memory` in `mathematics/stochastic_processes` and `diluted_hopfield` in `computer_science/neural_networks`, as Bertin 2021 poses them and the model inventory records them, built on this
change's capability, each with a gallery example.

#### Scenario: Registered under the inventory names
- **WHEN** the change is applied and the inventory check runs
- **THEN** both models are registered under their field paths and marked implemented


### Requirement: Intrinsic-instability populations
The library SHALL provide `intrinsic_instability` in `biology/ecology`, the resident and invading populations of
Jensen 2023 section 13.2, Eqs. (13.3) and (13.4), whose resident population follows an Ornstein-Uhlenbeck step with
noise scaled by the square root of its size and is suppressed by an invader that a rare mutation seeds and that then
grows exponentially, built on this change's capability, with a gallery example.

#### Scenario: Leaving the fixed point from within
- **WHEN** `intrinsic_instability` runs for 4000 steps with its default parameters and random state 0
- **THEN** the invader is seeded before step 3000, the resident population stays within 0.15 of 1 before the
  invader appears, and at the end the resident population is below 0.5 and the invader above 1
