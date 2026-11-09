## MODIFIED Requirements

### Requirement: Drives read the step they run in
`ArrayUpdateLaw` SHALL pass the time of the step to a function that names `time` in its signature, so an array update
can set a state to member `k` of an ensemble at step `k`.

#### Scenario: Replay of an ensemble
- **WHEN** an array update sets the positions of a 20-site chain to member `min(time, 37)` of an ensemble of 38
  members, and the system runs to time 40
- **THEN** the positions at frame 10 are member 10, and the positions at frame 40 are member 37
