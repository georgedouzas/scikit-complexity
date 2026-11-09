## ADDED Requirements

### Requirement: Runs record their draws on request
`simulate` SHALL take `record_draws`, and with `record_draws=True` the trajectory SHALL hold every named draw of the
run by name and step, with the arguments of its distribution, and every firing of a jump law with its time and law.
Draws of laws and draws that agent rules make through the named draw of the system view SHALL both be recorded.

#### Scenario: Draws of a law
- **WHEN** a map with the law `x + normal('kick', 0, s)` runs 100 steps with `record_draws=True`
- **THEN** the trajectory holds 100 values of `kick`, each with the deviation `s` it was drawn with

#### Scenario: Draws of an agent rule
- **WHEN** the stochastic Deffuant system runs 10 steps with `record_draws=True`
- **THEN** the trajectory holds the outcome draw of every encounter of every step

#### Scenario: Gillespie firings
- **WHEN** the stochastic SIR system runs under Gillespie with `record_draws=True`
- **THEN** the trajectory holds the time and the law of every firing, and the states rebuilt from them match the
  recorded states

### Requirement: Runs replay recorded draws with their log-probability
`simulate` SHALL take `replay`, a trajectory recorded with its draws, and SHALL force every named draw and every jump
firing to its recorded value while it computes the arguments of each distribution from the current parameters and
states. The replayed trajectory SHALL hold the total log-probability of the forced draws as `log_probability`.

#### Scenario: Replay at the recording parameters
- **WHEN** a recorded run of the stochastic Deffuant system is replayed with the same parameters
- **THEN** the replayed states equal the recorded states, and `log_probability` equals the sum of the log-probabilities
  of the recorded draws

#### Scenario: Replay of a jump process
- **WHEN** a recorded Gillespie run of the stochastic SIR system is replayed with other rates
- **THEN** `log_probability` equals the sum of the log propensities of the firings minus the integral of the total
  propensity over the run, at those rates

#### Scenario: Replay that does not match
- **WHEN** a trajectory is replayed on a system whose draws have other names
- **THEN** `simulate` raises `ValidationError`, whose message names the draws the system makes and the trajectory lacks
