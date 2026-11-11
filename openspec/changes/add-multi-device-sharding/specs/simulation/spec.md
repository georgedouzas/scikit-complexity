# Spec Delta

## ADDED Requirements

### Requirement: Trajectories stream to a store
`simulate` SHALL take `store`, a path or a Zarr store. With a store, every recorded frame of every state SHALL be
written to the store, each process writing the shards it holds, and the trajectory SHALL read its states from the
store while it holds its times and observables in memory.

#### Scenario: Run on one device with a store
- **WHEN** a Gray-Scott system of 512 by 512 sites runs 50 frames with a store on the NumPy backend
- **THEN** the states read from the trajectory equal those of the same run without a store

#### Scenario: Sharded run with a store
- **WHEN** an Ising system of 512 by 512 sites runs 20 sweeps on four emulated CPU devices with a store
- **THEN** the store holds 20 frames of 512 by 512 spins, equal to the run on one device
