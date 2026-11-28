## ADDED Requirements

### Requirement: Stiff reaction diffusion runs fast
Ready-made reaction diffusion systems SHALL run 40 by 40 sites to the end times of Turing pattern studies in seconds.

#### Scenario: Schnakenberg
- **WHEN** `SchnakenbergSystem(shape=(40, 40), Dv=40)` simulates until 40
- **THEN** it finishes in under 30 seconds and forms the same pattern as before
