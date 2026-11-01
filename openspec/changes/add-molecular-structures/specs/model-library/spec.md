# Spec Delta

## ADDED Requirements

### Requirement: RNA neutral networks model
The library SHALL provide `rna_neutral_networks` in `biology/evolution` (Raman 2021 §13.2–13.3): walkers, one RNA
sequence per row of a lattice, propose one point mutation per unit of time, accepted at temperature zero when the fold
keeps the target in the neutral regime, or when the base-pair distance to the target does not grow in the adaptive
regime. It SHALL raise `MissingExtraError` without the extra `rna`.

#### Scenario: Neutral walk keeps the fold
- **WHEN** `rna_neutral_networks` runs with its defaults, 8 walkers of length 30 starting on the sequence
  `GGGGAAACCCCAUCGAUAGCGGGAAACCGC` with target `((((...)))).......((((....))))`, for 300 units of time
- **THEN** every walker folds into the target at every frame, and the mean Hamming distance from the start exceeds 8 at
  the end

#### Scenario: Adaptive walk reaches the target
- **WHEN** the model runs in the adaptive regime from 8 random sequences of length 30 toward the default target
- **THEN** the base-pair distance of every walker never increases, and its mean at the end is lower than at the start

#### Scenario: Model that needs the extra
- **WHEN** a user without the extra `rna` creates `rna_neutral_networks`
- **THEN** a `MissingExtraError` names the model and the extra, and `fuzzy_oil_drop` and
  `early_stage_folding_intermediate` still run with the extra `structures`

### Requirement: RNA folds and neutral networks are analysed in the evolution package
The package `skcomplex.models.biology.evolution` SHALL fold an RNA sequence into its minimum free energy secondary
structure with its free energy and the base-pair distance to a target, memoised per sequence, and SHALL provide
`NeutralNetworkAnalysis`, a subclass of `BaseAnalysis` that gives, for any function from sequences of an alphabet to
phenotype keys, the neutral network of a phenotype and the genotype and phenotype robustness and evolvability as Wagner
defines them.

#### Scenario: Hairpin fold
- **WHEN** the sequence `GGGGAAACCCC` is folded at 37 °C with the default energy parameters
- **THEN** the structure is `((((...))))` and its free energy is negative

#### Scenario: Toy map of the first letter
- **WHEN** the binary sequences of length 4 are mapped to their first letter
- **THEN** each phenotype's neutral network has 8 sequences in one component, every genotype robustness is 3, and every
  genotype evolvability is 1

#### Scenario: Robustness against evolvability
- **WHEN** the genotype robustness and evolvability of the walkers of a neutral run are pooled over walkers and frames
- **THEN** their Spearman correlation is below −0.4, as Wagner found for genotypes of RNA folds

### Requirement: Fuzzy oil drop model
The library SHALL provide `fuzzy_oil_drop` in `biology/systems_biology` (Konieczny et al. 2023 §1.3.1, Eqs. 1.1–1.9).
By default it replays the bundled NMR ensemble of the Trp-cage 1L2Y through a `DriveLaw`, one model per frame. With
`dynamics='folding'` it samples the dihedral angles of a chain from an extended start with a row score of `RD` and a
clash penalty. Its catalog entry SHALL declare `RD` and `K` as observables in the default install.

#### Scenario: Ensemble of the Trp-cage
- **WHEN** `fuzzy_oil_drop` runs with its defaults
- **THEN** the trajectory has 38 frames, every `RD` lies in `[0, 1]`, and the recorded `RD` of each frame equals the
  `RD` that `FuzzyOilDropAnalysis` gives for that model within 1e-12

#### Scenario: Folding lowers the relative distance
- **WHEN** the model folds the Trp-cage sequence from an extended chain at temperature zero for 300 units of time
- **THEN** `RD` never increases, and the radius of gyration at the end is lower than at the start


### Requirement: Structures and the fuzzy oil drop live in the systems biology package
The package `skcomplex.models.biology.systems_biology` SHALL read and write PDB and mmCIF files as structures holding
every model of an ensemble with the extra `structures`, read molecular dynamics trajectories with the extra
`trajectories`, rebuild a protein
backbone from its dihedral angles, and provide `FuzzyOilDropAnalysis`, a subclass of `BaseAnalysis` that gives `T`,
`O`, `R`, the divergences, `RD`, and `K` of a structure and a hydrophobicity scale.

#### Scenario: NMR ensemble
- **WHEN** a user reads the bundled structure 1L2Y
- **THEN** it holds 38 models of 20 residues and 304 atoms each

#### Scenario: Round trip
- **WHEN** a structure is written to mmCIF and to PDB and read back
- **THEN** both copies have the same residues, atom names, and coordinates within 0.001 Å

#### Scenario: Round trip through internal coordinates
- **WHEN** the backbone of a read structure is rebuilt from its dihedral angles
- **THEN** the rebuilt positions match the read ones with a deviation below 0.01 Å after superposition

#### Scenario: Ideal micelle and uniform hydrophobicity
- **WHEN** the observed distribution of a structure equals its theoretical distribution, and then is uniform
- **THEN** `RD` is 0 within 1e-12 in the first case and 1 within 1e-12 in the second

#### Scenario: Native against extended
- **WHEN** `RD` is computed for the bundled native ubiquitin and for its chain rebuilt with all `phi = −120°` and
  `psi = 130°`
- **THEN** the native `RD` is lower than the extended `RD`

#### Scenario: Unknown residue
- **WHEN** a structure holds a residue `SEP` that the scale does not cover
- **THEN** the analysis raises an `AnalysisError` that names `SEP`, its chain and number, and the scale

### Requirement: Early-stage folding intermediate model
The library SHALL provide `early_stage_folding_intermediate` in `biology/systems_biology` (Konieczny et al. 2023
§3.7.1). Each residue's dihedral angles lie on an ellipse path on the Ramachandran map. The model SHALL project a native
structure onto the path by the shortest periodic distance, and SHALL sample each residue's cell of the path with the
heat-bath rule of `SamplingInteraction` from per-amino-acid probabilities estimated from the bundled structures.

#### Scenario: Projection onto the path
- **WHEN** the native structure of BPTI is projected onto the ellipse path
- **THEN** every projected pair of dihedral angles lies on the path within 1e-9 rad, and no point of a 3600-point
  sampling of the path is closer to the native pair than the projected one

#### Scenario: Sampled cells follow the sequence's probabilities
- **WHEN** the model samples the cells of a sequence for 300 units of time
- **THEN** each residue's frequency of each cell over the last 200 frames matches its amino acid's probability within
  0.15

### Requirement: Protein models need the structures extra
`fuzzy_oil_drop` and `early_stage_folding_intermediate` SHALL read structures through Biopython from the extra
`structures`, and SHALL raise `MissingExtraError` without it while staying listed in the catalog.

#### Scenario: Protein model that needs the extra
- **WHEN** a user without the extra `structures` creates `fuzzy_oil_drop` or `early_stage_folding_intermediate`
- **THEN** a `MissingExtraError` names the model, the extra `structures`, and the install command

### Requirement: Molecular structure models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL be marked implemented in
the model inventory, with a gallery example under `docs/examples/biology/<subfield>/plot_<name>.py` that runs within
the gallery budget.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `rna_neutral_networks` in `biology/evolution`
  - `fuzzy_oil_drop` in `biology/systems_biology`
  - `early_stage_folding_intermediate` in `biology/systems_biology`
