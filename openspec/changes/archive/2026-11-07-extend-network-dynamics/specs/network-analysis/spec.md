# Spec Delta

## ADDED Requirements

### Requirement: Autocatalytic sets are found
`AutocatalyticSetAnalysis` SHALL find the largest reflexively autocatalytic and food-generated set of a catalytic
reaction network, read from a network snapshot or chosen frames of a trajectory, with every hyperedge a reaction that
produces its last member and every edge from a catalyst to a product, by the algorithm of Hordijk and Steel. The food
set SHALL default to the nodes that no reaction produces.

#### Scenario: Catalysed reactions
- **WHEN** three reactions reachable from the food set each have a catalyst among their products
- **THEN** the largest RAF holds the three reactions, and leaves out a fourth whose reactant no reaction produces

#### Scenario: Lost catalyst
- **WHEN** one reaction of a chain loses its only catalyst
- **THEN** the reactions that need its product are removed with it, and no RAF exists
