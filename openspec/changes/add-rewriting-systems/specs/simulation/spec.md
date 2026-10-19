# Spec Delta

## ADDED Requirements

### Requirement: Frames and trajectories carry rewritten structures
A frame of a rewriting system SHALL carry the structure of every rewriting element at that generation, its word,
patch, or host graph, with its geometry where an interpretation is declared. A trajectory SHALL return the structure of
an element at a frame by the element's path, and the recorded structures SHALL be copies that later generations do not
change.

#### Scenario: Structure of a generation
- **WHEN** the Cantor substitution has run 4 generations and the user asks the trajectory for the structure of
  `cantor` at frame 2
- **THEN** it returns a patch of 4 intervals, and their geometry is the same after the run as during it

#### Scenario: Unknown element
- **WHEN** the user asks a trajectory for the structure of a path that is not a rewriting element
- **THEN** it raises a key error that lists the rewriting elements of the system
