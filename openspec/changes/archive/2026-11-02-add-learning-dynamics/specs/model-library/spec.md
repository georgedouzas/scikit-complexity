# Spec Delta

## ADDED Requirements

### Requirement: Elman simple recurrent network
The library SHALL provide Elman's simple recurrent network (Gros 2015 §10.5.1, Eq. 10.18, and Elman 1990) as a
registered factory of a system under discrete time in epochs. It SHALL nest the discrete map
`s(t) = σ(A s(t-1) + D x(t) + b)`, `y(t) = σ(W s(t) + c)` with adaptive shaped parameters, and train it with a
`TrainingInteraction` on the squared error of predicting the next input, by default on Elman's temporal XOR stream.

#### Scenario: Temporal XOR
- **WHEN** `run('elman_network', settings={'until': 200, 'random_state': 0})` trains 4 hidden units with Adam on a
  stream of 3000 bits
- **THEN** after 200 epochs the metric `xor_error`, the mean squared error of the predictions of every third bit, is
  below 0.05, the metric `random_error` of the predictions of the other bits stays above 0.2, and
  `training/loss` at the last frame is below its value at frame 0

#### Scenario: Elman's original rule
- **WHEN** `elman_network` is created with `truncation=1`
- **THEN** it trains with the previous hidden states as constant context inputs, and `xor_error` still falls below
  `random_error` within 500 epochs

#### Scenario: Gallery example of the Elman network
- **WHEN** the gallery runs `docs/examples/computer_science/neural_networks/plot_elman_network.py`
- **THEN** it records at most 300 frames, its thumbnail is the time series view of the loss and the two metrics, in
  which `xor_error` falls toward 0 while `random_error` stays near 0.25, and below the screenshot a Matplotlib figure
  shows the mean squared prediction error at each position of the bit triple before and after training, with the
  third position alone near 0 after training

### Requirement: The generalized Lotka-Volterra matrix is a parameter
`generalized_lotka_volterra` SHALL hold its interaction matrix as a shaped parameter of a `CouplingInteraction`, so a
schedule or an experiment changes it, and SHALL keep its trajectory.

#### Scenario: Rebuilt community keeps its trajectory
- **WHEN** `generalized_lotka_volterra` runs with its catalog defaults
- **THEN** its trajectory equals the trajectory of the previous constant matrix within 1e-10

#### Scenario: Matrix swept by an experiment
- **WHEN** a grid experiment sets the matrix parameter to two matrices that differ off the diagonal
- **THEN** the two members of the experiment differ in their final abundances

### Requirement: Random Lotka-Volterra community with k-body interactions
The library SHALL provide `higher_order_random_lotka_volterra` in `biology/ecology` (Sidhom and Galla 2024, arXiv
2409.10990): a group of species with the growth `x_i (k - x_i)` and the rate `x_i sum_(j<k) a_ijk x_j x_k` of the
paper's Eq. 2, written `x_i contraction(a, x, x) / 2`, from a random coupling tensor `a` of order 3 with the mean, variance,
and correlation of its Eqs. 4 and 5, a shaped parameter drawn by the factory from its `random_state`, so a schedule or an
experiment changes it, with a curated gallery example.

#### Scenario: No interactions
- **WHEN** the model runs with a tensor of zeros
- **THEN** every species converges to its carrying capacity within 1e-6

#### Scenario: Random community
- **WHEN** the model runs with its defaults, the equations, scaling, and parameters of Sidhom and Galla (2024, arXiv
  2409.10990), over 20 seeds
- **THEN** every abundance stays non-negative and finite, and the mean fraction of surviving species matches the
  paper's result at those parameters within its stated precision: 200 species, `mu = -4`, `sigma = 1`, and `gamma = 0`
  give 0.928 from its Eq. 17, and the paper states no number, so the check allows 0.02, three standard errors of 20 seeds

### Requirement: Learning models are recorded in the model inventory
Every model this change implements SHALL register under the field path of its subfield and SHALL have an entry in the
model inventory, and a model the inventory records as blocked by this change SHALL register under the name of its
entry.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `elman_network` in `computer_science/neural_networks`, with the sources Gr 10.5.1 and Elman 1990
  - `higher_order_random_lotka_volterra` in `biology/ecology`
