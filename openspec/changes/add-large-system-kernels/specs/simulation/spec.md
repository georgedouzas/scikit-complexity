# Spec Delta

## ADDED Requirements

### Requirement: Runs can record only observables
`simulate` SHALL take `record_states`, true by default. With `record_states=False` the trajectory SHALL hold the times
and the observables and no states, the JAX backend SHALL evaluate the observables on the device and copy only them to
the host, and reading a state SHALL raise `ValidationError`, whose message names `record_states`. Experiments whose
outputs are all observables SHALL run with `record_states=False`.

#### Scenario: Order parameter of a large flock
- **WHEN** a Vicsek system of 100000 particles runs 100 frames with `record_states=False` on `JaxBackend(device='cpu')`
- **THEN** the trajectory holds 100 values of `polarization`, and the run copies no positions to the host

#### Scenario: Reading a state that was not recorded
- **WHEN** a user asks such a trajectory for `flock.x`
- **THEN** it raises `ValidationError`, whose message names `record_states`

#### Scenario: Ensemble of observables
- **WHEN** an ensemble keeps only the observable `polarization` of a Vicsek system
- **THEN** its runs record no states, and its result equals the result with every state recorded
