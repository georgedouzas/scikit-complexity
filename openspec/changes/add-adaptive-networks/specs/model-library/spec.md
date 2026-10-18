# Spec Delta

## ADDED Requirements

### Requirement: Network models
The library SHALL provide Kuramoto on complete graphs and networks, the power-grid swing equation, SIS and SIR on
networks, adaptive SIS with rewiring, Barabási-Albert growth, voter and majority models on networks, random Boolean
networks, coupled logistic maps, Vicsek flocking, and DebtRank interbank contagion as registered factories.

#### Scenario: Random Boolean network phases
- **WHEN** random Boolean networks with in-degree 1, 2, and 3 run from random states
- **THEN** damage from a single flipped node dies out at in-degree 1, stays near constant at 2, and spreads at 3
