# Spec Delta

## Purpose

Lets a user declare Boolean decision states under clauses and cardinality constraints, decide satisfiability as
constraints arrive frame by frame, and train estimators by epochs.

## ADDED Requirements

### Requirement: Constraint problems declare Boolean states and constraints
Elements SHALL declare Boolean decision states, clauses over literals, and cardinality constraints that at least or
at most `k` of a set of literals hold. Constraints SHALL be given as a sequence, so a run can reveal them in order.

#### Scenario: Perceptron pattern as a constraint
- **WHEN** a binary perceptron of 21 weights `w_i = ±1` must give the output `sigma` for a pattern `xi`
- **THEN** the pattern becomes the constraint that at least 11 of the literals `w_i = sigma xi_i` hold

### Requirement: Satisfiability is decided incrementally
A run SHALL reveal a set number of further constraints per frame and SHALL decide at every frame whether the
revealed constraints are satisfiable, keeping what the solver learned before. The frame SHALL hold a satisfying
assignment when one exists, and the observable `satisfiable`. A problem SHALL also enumerate all solutions up to a
limit.

#### Scenario: Small perceptron
- **WHEN** the perceptron of Bertin (2021 Ex. 6.4) with `N = 3` and `p = 2` enumerates its solutions
- **THEN** every enumerated weight vector satisfies Eq. 6.60 for both patterns, and no other vector does

#### Scenario: Capacity of the binary perceptron
- **WHEN** random patterns of 21 components are revealed one per frame until the constraints become unsatisfiable,
  over 20 draws
- **THEN** the mean load `p/N` at the first unsatisfiable frame lies between 0.7 and 1.0, near the critical load 0.83

### Requirement: Estimators train by epochs
Elements SHALL wrap an estimator that learns incrementally, with its training data, and SHALL run one epoch per
frame. The frame SHALL hold the learned coefficients and the training error.

#### Scenario: Separable patterns
- **WHEN** a real-weight perceptron with 25 inputs trains on 12 random patterns
- **THEN** its training error reaches 0 within 200 epochs
