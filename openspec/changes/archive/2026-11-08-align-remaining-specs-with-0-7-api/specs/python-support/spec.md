## MODIFIED Requirements

### Requirement: A fresh install works

The declared dependency ranges SHALL exclude every version known to break the package, so an install from the index,
which takes the newest versions the ranges allow rather than the lock file, passes the fast tests on the newest declared
Python version.

#### Scenario: Hypergraph system after a fresh install

- **WHEN** a user installs the package from the index into a new environment and simulates `HypergraphKuramotoSystem`
- **THEN** the simulation runs
