# Spec Delta

## ADDED Requirements

### Requirement: Monte Carlo copies labels between neighbouring sites
`SamplingInteraction` SHALL sample a state of integer labels on a `LatticeSpace` or a `NetworkSpace` with a region
score and the move `'copy'`, which copies the label of a random neighbour into a random site, `hits` times per site and
sweep, accepted by the Metropolis or the Glauber rule. The region score SHALL be the weighted contact energy between
different labels of neighbouring sites plus elastic constraints `a_l (V_l - A_l)^2` and `b_l (S_l - B_l)^2` on the
volume and the surface of every region, whose strengths and targets are functions of the labels. The sampler SHALL keep
the volume and the surface of every label and update the two changed labels on every accepted copy. A network SHALL
weigh contacts by the edge attribute `'weight'` and site volumes by the node attribute `'volume'`. A region score SHALL
refuse moves other than `'copy'` and the heat-bath rule, and the move `'copy'` SHALL refuse other objectives.

#### Scenario: Volume constraint holds on average
- **WHEN** one region of 9 sites in a medium on a periodic 12 × 12 lattice is sampled with unit contacts and a volume
  constraint of target 30
- **THEN** its volume averages within 2 of 30 after 50 sweeps

#### Scenario: Zero temperature never raises the objective
- **WHEN** four regions are sampled at infinite intensity
- **THEN** the objective recomputed from the labels of every frame never grows

#### Scenario: Weighted networks
- **WHEN** a periodic grid graph with node volumes 0.5 and edge weights 2 holds one region with a volume target of 30
- **THEN** the weighted volume of the region averages within 2 of 30

#### Scenario: Refused moves
- **WHEN** a region score is given the move `'flip'` or the heat-bath rule, or a row score the move `'copy'`
- **THEN** preparing the interaction raises a validation error
