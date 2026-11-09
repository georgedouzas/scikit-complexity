## MODIFIED Requirements

### Requirement: Fuzzy oil drop system
The library SHALL provide `fuzzy_oil_drop` in `biology/systems_biology` (Konieczny et al. 2023 §1.3.1, Eqs. 1.1–1.9).
By default it replays the bundled NMR ensemble of the Trp-cage 1L2Y through an `ArrayUpdateLaw`, one system per frame.
With `dynamics='folding'` it samples the dihedral angles of a chain from an extended start with a row score of `RD` and
a clash penalty. Its catalog entry SHALL declare `RD` and `K` as observables in the default install.

#### Scenario: Ensemble of the Trp-cage
- **WHEN** `fuzzy_oil_drop` runs with its defaults
- **THEN** the trajectory has 38 frames, every `RD` lies in `[0, 1]`, and the recorded `RD` of each frame equals the
  `RD` that `fuzzy_oil_drop_measures` gives for that system within 1e-12

#### Scenario: Folding lowers the relative distance
- **WHEN** the system folds the Trp-cage sequence from an extended chain at temperature zero for 300 units of time
- **THEN** `RD` never increases, and the radius of gyration at the end is lower than at the start
