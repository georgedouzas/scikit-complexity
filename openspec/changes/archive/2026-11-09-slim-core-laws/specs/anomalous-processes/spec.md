## MODIFIED Requirements

### Requirement: States are driven by fractional Gaussian noise
A noise law with a Hurst index SHALL drive a state with the increments of fractional Brownian motion of the declared
Hurst index `0 < H < 1`, scaled by an amplitude. The increments of one law SHALL have the covariance of fractional
Gaussian noise across the whole run, and SHALL be independent of other laws and other group members. A Hurst index of
1/2 SHALL give the same process as a noise law without a Hurst index.

#### Scenario: Increment correlation
- **WHEN** a noise law with `H = 0.75` drives 1000 group members for 1000 steps
- **THEN** the correlation of consecutive increments lies within 0.02 of `2^(2H-1) - 1`

#### Scenario: Superdiffusion
- **WHEN** a user computes the ensemble mean squared displacement of 1000 members driven by a noise law with
  `H = 0.75`
- **THEN** the scaling exponent lies within 0.05 of `2H = 1.5`

#### Scenario: Hurst index of one half
- **WHEN** a noise law with `H = 0.5` and the same amplitude as a noise law without a Hurst index run over ensembles
- **THEN** the variances of the state at time 1 agree within 2 standard errors

### Requirement: Fractional noise laws are validated
Compiling SHALL reject a Hurst index outside the open interval from 0 to 1, and an amplitude that depends on a state
when the Hurst index differs from 1/2. Preparing a noise law SHALL reject a Hurst index together with a named process or
a conserved noise.

#### Scenario: Hurst index out of range
- **WHEN** a noise law has `H = 1`
- **THEN** compiling raises a named error that names the law and the value, and states the open interval

#### Scenario: State-dependent amplitude
- **WHEN** a noise law with `H = 0.7` has an amplitude proportional to its own state
- **THEN** compiling raises a named error that names the law and the state, and says that only state-independent
  amplitudes are supported for a Hurst index other than 1/2

#### Scenario: Hurst index with a shared process
- **WHEN** a noise law with `H = 0.7` names the process `'shared'`
- **THEN** preparing it raises a `ValidationError` that names the target and asks to remove the process

#### Scenario: Hurst index on a conserved noise
- **WHEN** a conserved noise law has `H = 0.7`
- **THEN** preparing it raises a `ValidationError` that names the target and says a conserved noise is Brownian

### Requirement: Fractional noise is integrated on a declared horizon
Systems with fractional noise SHALL run with a stepper that draws the correlated increments of the whole run on a fixed
step up to a horizon, which a simulation run sets from its end time. Advancing beyond the horizon, or choosing a stepper
without correlated increments, SHALL raise a named error.

#### Scenario: Beyond the horizon
- **WHEN** a frame-by-frame simulation with fractional noise and a horizon of 10 is advanced to time 12
- **THEN** it raises a named error that names the horizon and the requested time, and asks for a larger horizon

#### Scenario: Wrong stepper
- **WHEN** a system with a noise law with `H = 0.7` is run with the Euler-Maruyama stepper
- **THEN** it raises a named error that names the fractional Euler-Maruyama stepper as the one to use
