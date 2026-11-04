# Spec Delta

## MODIFIED Requirements

### Requirement: Series are embedded in delay coordinates
The analysis layer SHALL build delay embeddings of a series, choose the delay at the first minimum of the auto mutual
information, and choose the dimension where the fraction of false nearest neighbours falls below a threshold.

#### Scenario: Lorenz embedding
- **WHEN** a user embeds 20,000 samples of the Lorenz-63 `x` with step 0.01
- **THEN** the chosen delay lies between 0.10 and 0.25 time units, and the chosen dimension is 3

#### Scenario: Map embedding
- **WHEN** a user embeds 10,000 iterates of the Hénon map `x`
- **THEN** the chosen dimension is 2
