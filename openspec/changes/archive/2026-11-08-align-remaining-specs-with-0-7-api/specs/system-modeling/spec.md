## MODIFIED Requirements

### Requirement: Boundary laws constrain states after each step
An interaction SHALL be able to return boundary laws, which map states to new values after every internal solver step,
however long the frame, combine with any law kind without changing the inferred form, and describe walls and periodic
boxes. A frame SHALL never report a state that a boundary law would change.

#### Scenario: Reflecting wall in an ODE system
- **WHEN** a particle group with rate laws also has reflecting wall boundaries
- **THEN** the system compiles to the ODE form and no member crosses a wall

#### Scenario: Periodic box
- **WHEN** a member leaves a periodic box through one side
- **THEN** it reappears at the opposite side with the same velocity

#### Scenario: Long frames keep particles in the box
- **WHEN** a particle at 0.5 with velocity 1 in a reflecting box of side 1 runs one frame of duration 5 with `RK4`
  and with `ScipyODESolver`
- **THEN** at time 5 its position is 0.5 and its velocity is -1 within 1e-6, after four reflections
