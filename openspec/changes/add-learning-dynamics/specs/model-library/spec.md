# Spec Delta

## ADDED Requirements

### Requirement: Elman simple recurrent network
The library SHALL provide Elman's simple recurrent network (Gros 2015 §10.5.1, Eq. 10.18, and Elman 1990) as a
registered factory that returns a learning system. Its inner model SHALL be the discrete map
`s(t) = σ(A s(t-1) + D x(t) + b)`, `y(t) = σ(W s(t) + c)` in symbolic laws with matrix parameters. Its loss SHALL be
the squared error of predicting the next input, by default on Elman's temporal XOR stream.

#### Scenario: Temporal XOR
- **WHEN** `run('elman_network', settings={'until': 200, 'random_state': 0})` trains 4 hidden units with Adam on a
  stream of 3000 bits
- **THEN** after 200 epochs the metric `xor_error`, the mean squared error of the predictions of every third bit, is
  below 0.05, the metric `random_error` of the predictions of the other bits stays above 0.2, and `elman/loss` at the
  last frame is below its value at frame 0

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

### Requirement: Learning models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** this entry is registered under its field path and marked implemented, with the sources Gr 10.5.1 and
  Elman 1990:
  - `elman_network` in `computer_science/neural_networks`

#### Scenario: Out-of-scope entry moved
- **WHEN** the change starts
- **THEN** the inventory entry of `elman_network` changes from out of scope to `pending add-learning-dynamics`, with no
  reason and no library, and the inventory check passes
