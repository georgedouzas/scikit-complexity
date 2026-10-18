# Spec Delta

## Purpose

Lets a user run cellular automata over finitely presented groups, decide their injectivity and surjectivity on finite
universes, study linear automata and subshifts, and measure their information content.

## ADDED Requirements

### Requirement: Automata run on Cayley graphs of groups
Local rules SHALL run on Cayley graphs of finitely presented groups, built on finite quotients or on finite balls, with
memory sets given as group elements.

#### Scenario: Free group majority
- **WHEN** the majority automaton on the free group with two generators runs on a ball of radius 6
- **THEN** every interior cell takes the majority value of itself and its four neighbours given by the generators and
  their inverses

#### Scenario: Orientation is kept
- **WHEN** a rule copies the value at the generator `a` on the Heisenberg group
- **THEN** each cell takes the value of `g a`, not of `a g`

### Requirement: Injectivity and surjectivity are decided on finite universes
The analysis layer SHALL decide injectivity, surjectivity, and pre-injectivity of an automaton on finite quotients and
windows, find Garden of Eden patterns, and find the inverse of a reversible automaton.

#### Scenario: Life has a Garden of Eden
- **WHEN** a user checks the Game of Life on a 6 × 6 torus
- **THEN** it is reported not surjective, with a configuration that has no preimage

#### Scenario: Reversible rule
- **WHEN** a user asks for the inverse of the elementary shift automaton
- **THEN** the result is the shift in the opposite direction, and their composition is the identity

### Requirement: Linear automata are decided algebraically
Linear automata SHALL be represented as group-ring elements whose invertibility and surjectivity are decided
algebraically.

#### Scenario: Laplacian over a finite field
- **WHEN** a user checks the discrete Laplacian on the cycle of length 5 over the field with 2 elements
- **THEN** the answer matches the invertibility of the corresponding circulant matrix

### Requirement: Subshifts constrain configurations
Subshifts of finite type SHALL check configurations against forbidden patterns, count admissible patterns, and sample
admissible configurations.

#### Scenario: Golden mean shift
- **WHEN** a user counts admissible words of length n in the subshift that forbids `11`
- **THEN** the counts follow the Fibonacci numbers, and the entropy estimate approaches the log of the golden ratio

### Requirement: Automata information is measured
The analysis layer SHALL estimate the entropy of configuration sets along windows, track the spatial entropy rate of
runs, and compute difference patterns of single-cell perturbations.

#### Scenario: Entropy decreases
- **WHEN** an elementary automaton runs from a random configuration on a large ring
- **THEN** the spatial entropy rate never increases from one step to the next beyond estimation error
