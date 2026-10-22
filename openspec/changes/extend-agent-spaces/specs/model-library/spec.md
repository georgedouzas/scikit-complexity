# Spec Delta

## ADDED Requirements

### Requirement: Agent models in three-dimensional and geographic spaces
The library SHALL provide, as registered factories:
- boids in three dimensions under `biology/collective_behaviour` (Think Complexity ch. 10);
- the Kauai population sentiment model on geographic layers under `social/organizations` (Carmichael et al. 2019
  ch. 3), with a small synthetic island dataset in its package.

#### Scenario: Boids align in three dimensions
- **WHEN** 50 boids with random headings fly in a periodic three-sided `ContinuousSpace` with only the alignment rule for 200 steps
- **THEN** the polarization, the norm of the mean unit velocity, rises from below 0.3 to above 0.9, and the depth
  coordinates of the boids still spread over more than half of the box depth

#### Scenario: Population follows the census layer
- **WHEN** the Kauai model is created from the synthetic island's census layer with populations 100, 50, and 0
- **THEN** it has 150 resident agents, each with a home point inside its census region

#### Scenario: Positive event raises sentiment
- **WHEN** the Kauai model runs with a positive event on a topic that half of the residents weight above zero
- **THEN** the mean sentiment of those residents rises and stays within -1 and 1, and the others' mean changes only
  through interaction
