## ADDED Requirements

### Requirement: Reductions differentiate with respect to the reading member
The derivative of `total`, `mean`, or `variance` of an expression over a set of neighbours, with respect to a state of
the reading member, SHALL be the reduction of the derivative of the expression, for `total` and `mean`, or its
derivative by the chain rule, for `variance`, with the set `over` and the condition `where` held fixed. The derivative
SHALL lower on every backend that lowers the reduction.

#### Scenario: Mean distance
- **WHEN** a law differentiates the mean over the neighbours within a radius of the distance `|x_j - x|` with respect
  to the position `x` of the reading member
- **THEN** it evaluates to the mean of the unit vectors from the neighbours to the member, within 1e-12 of the
  finite-difference value at a configuration with no neighbour on the edge of the radius

#### Scenario: Gradient flow over a reduction
- **WHEN** a `GradientInteraction` moves positions down the potential `mean((|x_j - x| - 1)^2)` over the neighbours
  within a radius of 2 of a pair of members at distance 1.5
- **THEN** the pair relaxes to the distance 1 within 1e-6

#### Scenario: Reduction that the derivative cannot reach
- **WHEN** a law differentiates `maximum` over neighbours with respect to a state of the reading member
- **THEN** compiling raises `ValidationError`, whose message names `maximum` and the reductions that differentiate
