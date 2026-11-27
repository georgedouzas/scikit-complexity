## ADDED Requirements

### Requirement: Values of groups whose size changes
`Trajectory.get_values` SHALL return the per-agent values of a group whose size changes as a list of one array per
frame, in the order of the members at that frame.

#### Scenario: Growing vocabulary
- **WHEN** a Simon Yule run grows its words and the count of every word is read with `get_values`
- **THEN** it returns one array per frame whose length is the number of words at that frame, equal to what
  `get_agents` gives
