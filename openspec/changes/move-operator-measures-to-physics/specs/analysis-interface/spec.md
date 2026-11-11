## ADDED Requirements

### Requirement: The core analyses measure what several fields share
Every analysis of `skcomplex.core.analysis` SHALL measure a quantity that systems of several fields share. An analysis
of one field SHALL live in the common package of that field and SHALL follow the same contract as the core analyses:
a public function of a trajectory, a system, or data, with keyword settings and a frozen result. The discovery of
applicable analyses SHALL list the core analyses only.

#### Scenario: Discovery lists the core analyses
- **WHEN** a user calls `applicable_analyses` on a trajectory of a chain of quantum spins
- **THEN** the result lists neither `operator_measures` nor any metabolism analysis, and every analysis it lists is
  a public name of `skcomplex.core.analysis`

#### Scenario: Field analysis with the core contract
- **WHEN** a user calls `operator_measures(trajectory, kind='purity')` from `skcomplex.systems.physics.common.quantum`
  on a trajectory of a dephasing qubit
- **THEN** it returns a frozen result whose purity starts at 1 and never increases from frame to frame
