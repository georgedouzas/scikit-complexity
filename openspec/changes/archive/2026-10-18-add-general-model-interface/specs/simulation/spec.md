# Spec Delta

## ADDED Requirements

### Requirement: Trajectories record observables
Trajectories SHALL hold the value of every observable of every system at every frame under the observable's path,
alongside the states.

#### Scenario: Observable by path
- **WHEN** a user asks a trajectory for `market/volume`, an observable of the subsystem `market`
- **THEN** it returns one value per frame
