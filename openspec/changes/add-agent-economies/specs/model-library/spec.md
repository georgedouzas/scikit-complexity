# Spec Delta

## ADDED Requirements

### Requirement: Economic and social agent models
The library SHALL provide the El Farol bar, Brock-Hommes heuristic switching, a double auction, an order-book market
with imitation, fractional-reserve banking, a firm-bank credit network, a job-search labour market, Sugarscape, the
spatial prisoner's dilemma, and Axelrod culture dissemination as registered factories.

#### Scenario: Spatial cooperation
- **WHEN** the spatial prisoner's dilemma runs with imitate-the-best updating and a temptation payoff of 1.6
- **THEN** cooperators survive in clusters for 200 steps instead of dying out
