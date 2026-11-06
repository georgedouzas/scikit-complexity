# Spec Delta

## ADDED Requirements

### Requirement: Cellular Potts model sorts cells by differential adhesion
The library SHALL provide `cellular_potts` in `biology/physiology`, after Nemati and de Graaf (2024, arXiv 2404.09055,
Eq. 3) and Graner and Glazier (1992). It SHALL sample a confluent tissue of cells, regions of sites with one label,
whose kinds have a symmetric adhesion matrix, with an area constraint and an optional perimeter constraint, by label
copies under discrete time on a periodic square lattice with the Moore neighbourhood, a triangular lattice, or a
disordered lattice, the periodic Voronoi tessellation of jittered points with contacts weighted by side length over the
mean and sites of their Voronoi area. Its catalog entry SHALL declare the share of homotypic contacts and the mean
relative deviation of the areas from the target as observables.

#### Scenario: Area constraint holds
- **WHEN** the tissue runs 200 sweeps on any of the three lattices
- **THEN** no cell vanishes and the mean relative deviation of the areas after 50 sweeps is below 0.1

#### Scenario: Differential adhesion sorts the kinds
- **WHEN** the tissue runs 1500 sweeps on the square or the disordered lattice with the default adhesion
- **THEN** the share of homotypic contacts rises by more than 0.1, while equal adhesions keep it within 0.1
