# Spec Delta

## ADDED Requirements

### Requirement: RNA neutral networks model
The library SHALL provide `rna_neutral_networks` in `biology/evolution` (Raman 2021 §13.2–13.3): walkers, one RNA
sequence each, propose one point mutation per unit of time, accepted when the fold keeps the target in the neutral
regime, or when the base-pair distance to the target does not grow in the adaptive regime. It SHALL record each
walker's structure, free energy, distances, robustness, and evolvability. It needs the extra `rna`.

#### Scenario: Neutral walk keeps the fold
- **WHEN** `rna_neutral_networks` runs with its defaults, 8 walkers of length 30 starting on the sequence
  `GGGGAAACCCCAUCGAUAGCGGGAAACCGC` with target `((((...)))).......((((....))))`, for 300 units of time
- **THEN** every walker folds into the target at every frame, the mean Hamming distance from the start exceeds 8 at the
  end, and every genotype robustness lies between 0 and 90

#### Scenario: Robustness against evolvability
- **WHEN** the genotype robustness and evolvability recorded by the same run are pooled over walkers and frames
- **THEN** their Spearman correlation is below −0.4, as Wagner found for genotypes of RNA folds

#### Scenario: Adaptive walk reaches the target
- **WHEN** the model runs in the adaptive regime from 8 random sequences of length 30 toward the default target
- **THEN** the base-pair distance of every walker never increases, and its mean at the end is lower than at the start

### Requirement: Fuzzy oil drop model
The library SHALL provide `fuzzy_oil_drop` in `biology/systems_biology` (Konieczny et al. 2023 §1.3.1, Eqs. 1.1–1.9).
By default it replays the bundled NMR ensemble of the Trp-cage 1L2Y, one model per frame. As an option it folds a chain
from an extended start by Monte Carlo on its dihedral angles with `RD` and a clash penalty as the score. Either way it
SHALL record `T`, `O`, `R`, the divergences, `RD`, and `K` per frame, in the default install.

#### Scenario: Ensemble of the Trp-cage
- **WHEN** `fuzzy_oil_drop` runs with its defaults
- **THEN** the trajectory has 38 frames, one per NMR model, every `RD` lies in `[0, 1]`, and the recorded `RD` of each
  frame equals the `RD` computed by the analysis function for that model within 1e-12

#### Scenario: Folding lowers the relative distance
- **WHEN** the model folds the Trp-cage sequence from an extended chain at temperature zero for 300 units of time
- **THEN** `RD` never increases, and the radius of gyration at the end is lower than at the start

### Requirement: Early-stage folding intermediate model
The library SHALL provide `early_stage_folding_intermediate` in `biology/systems_biology` (Konieczny et al. 2023
§3.7.1). Each residue's dihedral angles lie on an ellipse path on the Ramachandran map. The model SHALL project a native
structure onto the path by the shortest periodic distance, and SHALL sample each residue's cell of the path with
heat-bath moves from per-amino-acid probabilities estimated from bundled structures.

#### Scenario: Projection onto the path
- **WHEN** the native structure of BPTI is projected onto the ellipse path
- **THEN** every projected pair of dihedral angles lies on the path within 1e-9 rad, and no point of a 3600-point
  sampling of the path is closer to the native pair than the projected one

#### Scenario: Path through the secondary structures
- **WHEN** the default ellipse path is drawn on the Ramachandran map
- **THEN** it passes within 20° of the right-handed helix `(−57°, −47°)` and of the β-strand `(−120°, 130°)`

#### Scenario: Sampled cells follow the sequence's probabilities
- **WHEN** the model samples the cells of a sequence for 300 units of time
- **THEN** each residue's frequency of each cell over the last 200 frames matches its amino acid's probability within
  0.15, and the mean information of the sampled cells per residue is at most `log2` of the number of cells

### Requirement: Molecular structure models have gallery examples
Each model of this change SHALL have a gallery example at `docs/examples/biology/<subfield>/plot_<name>.py` that runs
within the gallery budget, shows its Rerun primary view as the thumbnail, and shows at least one analysis plot from its
own run with a caption stating the behaviour shown.

#### Scenario: Neutral network example
- **WHEN** a user opens the `rna_neutral_networks` example page
- **THEN** its thumbnail is the Rerun view of the target structure's plane layout coloured by how often each site
  mutated, and below it a scatter of genotype robustness against evolvability shows the negative correlation

#### Scenario: Fuzzy oil drop example
- **WHEN** a user opens the `fuzzy_oil_drop` example page
- **THEN** its thumbnail is the Rerun view of the Trp-cage chain in three dimensions coloured by `O − T` per residue,
  and below it a plot of `T`, `O`, and `R` per residue, and of `RD` per model, shows how far the ensemble is from a
  micelle

#### Scenario: Early-stage example
- **WHEN** a user opens the `early_stage_folding_intermediate` example page
- **THEN** below the Rerun view of the backbone it shows the Ramachandran map with the native angles, the ellipse
  path, and the projected angles, and a plot of the logarithm of the curvature radius against the angle between
  peptide bond planes for regular chains along the path, beside the curve of Eq. 3.1, both rising from the helix to
  the strand

### Requirement: Molecular structure models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `rna_neutral_networks` in `biology/evolution`, from Raman 2021 §13.2 and §13.3
  - `fuzzy_oil_drop` in `biology/systems_biology`, from Konieczny et al. 2023 §1.3.1
  - `early_stage_folding_intermediate` in `biology/systems_biology`, from Konieczny et al. 2023 §3.7.1

#### Scenario: Model that needs the extra
- **WHEN** a user without the extra `rna` creates `rna_neutral_networks`
- **THEN** a named error names the model and the extra, and `fuzzy_oil_drop` and `early_stage_folding_intermediate`
  still run
