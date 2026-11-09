## MODIFIED Requirements

### Requirement: Injectivity and surjectivity are decided on finite universes
`automaton` SHALL take a `CompositeSystem` with a `LocalRuleInteraction` as its source, and SHALL decide injectivity, surjectivity,
and pre-injectivity of the automaton on finite quotients and
windows, find Garden of Eden patterns, and find the inverse of a reversible automaton.

#### Scenario: Life has a Garden of Eden
- **WHEN** a user checks the Game of Life on a 6 × 6 torus
- **THEN** it is reported not surjective, with a configuration that has no preimage

#### Scenario: Reversible rule
- **WHEN** a user asks for the inverse of the elementary shift automaton
- **THEN** the result is the shift in the opposite direction, and their composition is the identity

### Requirement: A cellular automaton is a system
A cellular automaton SHALL be a `CompositeSystem` with a `LocalRuleInteraction`, and no class SHALL duplicate it. The `run` of a
`LocalRuleLaw` SHALL be one step of the map on configurations, and several steps SHALL be a loop over that `run` or a
`simulate` run of the system.

#### Scenario: A step and a run agree
- **WHEN** a user applies the `run` of the `LocalRuleLaw` of a shift twice to a configuration, and simulates the system
  of the same rule from that configuration until 2
- **THEN** the last frame of the simulation equals the result of the loop

#### Scenario: No class for an automaton
- **WHEN** a user lists the public names of `skcomplex`
- **THEN** none is `Automaton`, `LinearAutomaton`, or `Subshift`
