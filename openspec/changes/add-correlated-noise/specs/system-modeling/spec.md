# Spec Delta

## ADDED Requirements

### Requirement: Noise laws name the Wiener process they share
A noise law SHALL accept an optional process. A law with no process SHALL drive every component and member with its
own independent Wiener process. Laws of one component, or of one member, that name the same local process SHALL share
one increment, and so SHALL the components of a vector target that names one process.

#### Scenario: One process drives two states
- **WHEN** an element drives its states `x` and `y` with unit noise on the same local process and runs from equal values
- **THEN** `x` and `y` have sample correlation 1 over the run, and `x - y` stays at 0 at every frame

#### Scenario: No process named
- **WHEN** a system whose noise laws name no process is compiled
- **THEN** every component and member gets its own independent Wiener process, as before this change

### Requirement: Wiener elements declare correlated processes
A Wiener element SHALL declare a number of Wiener processes and an optional constant covariance matrix, the identity
by default. Noise laws of any component SHALL name its processes through a role, and group laws that name one SHALL
share it across all members. A covariance that is not square, symmetric, or positive semidefinite SHALL be rejected.

#### Scenario: Covariance reproduced
- **WHEN** two states with zero drift take unit noise from the two processes of a Wiener element with covariance
  `[[1, 0.6], [0.6, 2]]`, over a run of 10 000 unit frames
- **THEN** the sample covariance of their frame increments matches the matrix within 0.1 in every entry

#### Scenario: Perfect correlation
- **WHEN** a Wiener element has covariance `[[1, 1], [1, 1]]` and its two processes drive two states with unit noise
- **THEN** the system compiles, and the two states stay equal at every frame

#### Scenario: Invalid covariance
- **WHEN** a Wiener element of dimension 2 has covariance `[[1, 2], [2, 1]]`
- **THEN** compiling raises a named error that names the element and its negative eigenvalue

#### Scenario: Process used as a value
- **WHEN** a rate law uses a Wiener element's process handle inside its expression
- **THEN** compiling raises a named error that names the law and says the handle is only a noise law's process
