# Spec Delta

## Purpose

Lets a user declare games between players, compute their Nash equilibria, and run learning dynamics whose rounds are
time steps.

## ADDED Requirements

### Requirement: Finite games declare strategies and payoffs
Elements SHALL declare a game between two players by their strategies and their payoff tables, given as arrays or as
symbolic expressions of the strategies and parameters.

#### Scenario: Payoffs from an expression
- **WHEN** a pricing game declares a price grid and a payoff expression in the two prices
- **THEN** the game's payoff tables hold the expression at every pair of grid prices

### Requirement: Equilibria are computed
The analysis layer SHALL return the Nash equilibria of a finite two-player game by support enumeration, vertex
enumeration, or the Lemke-Howson algorithm, as mixed strategies of both players.

#### Scenario: Matching pennies
- **WHEN** the equilibria of matching pennies are computed by support enumeration
- **THEN** there is one equilibrium, in which both players mix their two strategies with probability 1/2

### Requirement: Learning dynamics run one round per step
A game SHALL take a learning dynamic, any object that advances the game by one round, and SHALL then run one round per
step of discrete time, with the frame holding each player's current mixed strategy or play counts. The core SHALL ship
no learning dynamic; fictitious play, stochastic fictitious play, and replicator dynamics SHALL be provided by
`skcomplex.models.economics.realizations`.

#### Scenario: Fictitious play converges
- **WHEN** the library's fictitious play runs matching pennies for 1000 rounds
- **THEN** the empirical frequencies of both players are within 0.05 of 1/2 at the last frame

### Requirement: Games declare the time they run under
A game SHALL require static time without a learning dynamic and discrete time with one, and validation SHALL report
any other time with the game's path and a fix.

#### Scenario: Dynamic under static time
- **WHEN** a game with a learning dynamic is placed in a system with `StaticTime()`
- **THEN** validation reports the game and suggests `DiscreteTime()`

### Requirement: Continuous games are solved from best responses
The modelling layer SHALL turn a continuous game, given by each player's payoff in the strategies, into the
first-order conditions of the best responses, solved as algebraic equations.

#### Scenario: Symmetric Cournot duopoly
- **WHEN** two firms with zero cost face the price `1 - q_1 - q_2`
- **THEN** the equilibrium quantities are `1/3` each, exactly
