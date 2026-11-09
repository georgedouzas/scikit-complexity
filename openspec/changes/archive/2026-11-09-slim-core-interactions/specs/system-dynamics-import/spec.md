## MODIFIED Requirements

### Requirement: Remaining rates become relaxations or rates
The part of a stock's net rate that no flow variable holds SHALL become a `RateInteraction`. When it has the goal-seeking
form `(goal - stock) / time`, the module SHALL write the goal and the time as their own functions and the rate as
`(goal - stock) / time` of them, and otherwise it SHALL write the remaining rate as one function.

#### Scenario: Goal seeking becomes relaxation
- **WHEN** a system whose stock changes by `(target - stock) / adjustment_time` with no flow variable is translated
- **THEN** the stock's rate is a `RateInteraction` whose rate is `(target - stock) / adjustment_time`, the module holds
  the goal and the time as functions, and its run matches PySD within 1e-6

#### Scenario: No relaxation class
- **WHEN** the module of any translated system is read
- **THEN** it imports no relaxation interaction, since the core has none
