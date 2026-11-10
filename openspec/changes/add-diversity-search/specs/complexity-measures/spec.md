## ADDED Requirements

### Requirement: Historical novelty of a run
The analysis layer SHALL hold `open_endedness(source, *, kind='historical-novelty', embedding)`, which embeds every
frame, gives the novelty of every frame as one minus the largest cosine similarity with an earlier frame, and returns
the mean novelty as the open-endedness score of Kumar et al. (2024, arXiv 2412.17799).

#### Scenario: A fixed point is not novel
- **WHEN** a user measures a run whose frames are equal after the first
- **THEN** the novelty of every later frame is 0 within 1e-12

#### Scenario: Orthogonal frames
- **WHEN** a user measures a run whose embedded frames are distinct basis vectors
- **THEN** the novelty of every frame after the first is 1

### Requirement: Novelty and learnability to an observer
`open_endedness(source, *, kind='observer', observer)` SHALL fit a copy of a scikit-learn estimator with
`score_samples` to the artefacts up to every time `t`, give the loss matrix `l(t, T)` as the mean negative score of the
artefacts at `T > t`, and test novelty, the loss rising with `T` at fixed `t`, and learnability, the loss falling with
`t` at fixed `T`, as Hughes et al. (2024, arXiv 2406.04268) define them.

#### Scenario: Noise is learnable but not novel
- **WHEN** a user measures a stream of independent draws of one Gaussian with a Gaussian mixture observer
- **THEN** the result is learnable and not novel

#### Scenario: Drifting stream
- **WHEN** a user measures a stream of Gaussian draws whose mean grows slowly with time
- **THEN** the result is novel and learnable
