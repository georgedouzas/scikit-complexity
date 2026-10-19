# Spec Delta

## ADDED Requirements

### Requirement: Tumour growth laws
The library SHALL provide `exponential_linear_growth`, `power_law_growth`, `von_bertalanffy`, `generalized_logistic`,
`gompertz`, and `dynamic_carrying_capacity` as registered factories under `biology/oncology`, as the model inventory
lists them with their sources, each with a gallery example.

#### Scenario: Gompertz growth
- **WHEN** `gompertz` runs from a small tumour
- **THEN** its volume matches the closed-form Gompertz solution within 1e-6 relative and approaches the carrying
  capacity

### Requirement: Cell kill and toxicity
The library SHALL provide `skipper_log_kill`, `norton_simon`, and `drug_toxicity` as registered factories under
`biology/oncology` and `biology/physiology`, as the model inventory lists them with their sources, each with a gallery
example.

#### Scenario: Log kill
- **WHEN** `skipper_log_kill` receives equal impulsive doses on a non-growing tumour
- **THEN** each dose lowers the logarithm of the tumour size by the same amount within 1e-6

### Requirement: Chemotherapy
The library SHALL provide `de_pillis_radunskaya`, `multidrug_resistance`, `drug_resistance_switching`,
`delayed_resting_hunting_chemotherapy`, `delayed_metastasis`, `cell_cycle_chemotherapy`, and
`erythropoiesis_mercaptopurine` as registered factories under `biology/oncology`, as the model inventory lists them with
their sources, each with a gallery example.

#### Scenario: Chemotherapy clears a tumour
- **WHEN** `de_pillis_radunskaya` runs untreated and then with the book's constant chemotherapy infusion
- **THEN** the untreated tumour settles at its high equilibrium, and the treated tumour falls below 1% of its initial
  size while the normal cells stay above their lower bound

### Requirement: Immunotherapy
The library SHALL provide `kirschner_panetta`, `dendritic_cell_vaccine`, `anti_pd1_checkpoint`, `delayed_dc_vaccine`,
and `tumour_immune_suppression` as registered factories under `biology/oncology`, as the model inventory lists them with
their sources, each with a gallery example.

#### Scenario: Tumour-immune oscillations and clearance
- **WHEN** `kirschner_panetta` runs with an antigenicity in the oscillatory range and then with adoptive therapy and
  IL-2 above the book's threshold
- **THEN** the untreated tumour and effector cells oscillate, and the treated tumour is cleared

### Requirement: Anti-angiogenic therapy
The library SHALL provide `hahnfeldt` and `vascular_core_periphery` as registered factories under `biology/oncology`, as
the model inventory lists them with their sources, each with a gallery example.

#### Scenario: Hahnfeldt equilibrium
- **WHEN** `hahnfeldt` runs untreated
- **THEN** the tumour and carrying capacity approach the equilibrium volume `(b/d)^(3/2)` within 1%

### Requirement: Radiotherapy
The library SHALL provide `radiotherapy_competition` and `lq_radiosensitizer` as registered factories under
`biology/oncology`, as the model inventory lists them with their sources, each with a gallery example.

#### Scenario: Linear-quadratic fractions
- **WHEN** `lq_radiosensitizer` delivers a fraction of dose `d` without radiosensitizer
- **THEN** the surviving tumour fraction of that fraction is `exp(-alpha d - beta d^2)` within 1e-6

### Requirement: Hormone therapy
The library SHALL provide `continuous_androgen_suppression`, `ideta_intermittent_androgen`, `prostate_cell_quota`,
`droop_cell_quota`, `prostate_androgen_receptor`, `piecewise_linear_ias`, and `tamoxifen_cell_cycle` as registered
factories under `biology/ecology` and `biology/oncology`, as the model inventory lists them with their sources, each
with a gallery example.

#### Scenario: Intermittent suppression
- **WHEN** `ideta_intermittent_androgen` runs with its PSA thresholds
- **THEN** the therapy switches on above the upper threshold and off below the lower one, and PSA cycles between them
  while treatment is effective

### Requirement: Gene, viral, nanoparticle, and stem-cell therapies
The library SHALL provide `tcr_gene_therapy`, `sirna_tgf_beta`, `oncolytic_virus`, `oncolytic_virus_dendritic`,
`nanoparticle_corona`, `two_compartment_pharmacokinetics`, `stem_cell_differentiation`, `bone_marrow_blood_leukaemia`,
and `leukaemia_stem_cells` as registered factories under `biology/oncology`, `biology/physiology`, and
`chemistry/kinetics`, as the model inventory lists them with their sources, each with a gallery example.

#### Scenario: Two-compartment pharmacokinetics
- **WHEN** `two_compartment_pharmacokinetics` receives one bolus dose
- **THEN** the central concentration matches the closed-form bi-exponential solution within 1e-6 relative, and the total
  drug decays by elimination only

### Requirement: Combination therapies
The library SHALL provide `chemoimmunotherapy`, `clonal_chemoimmunotherapy`, `chemovirotherapy`,
`glioma_chemovirotherapy`, `her2_targeted_therapy`, and `angiogenesis_immunotherapy` as registered factories under
`biology/oncology`, as the model inventory lists them with their sources, each with a gallery example.

#### Scenario: Combination beats each therapy alone
- **WHEN** `chemovirotherapy` runs with the book's chemotherapy alone, the virus alone, and both
- **THEN** the tumour at the end of the run is smallest under the combination

### Requirement: Cellular network models
The library SHALL provide `phosphorylation_oscillator`, `biochemical_systems_theory`, `three_gene_boolean_network`,
`mammalian_cell_cycle`, `boolean_repressilator`, `goodwin_oscillator`, and `generalized_lotka_volterra` as registered
factories under `biology/ecology` and `biology/systems_biology`, as the model inventory lists them with their sources,
each with a gallery example.

#### Scenario: Goodwin oscillations
- **WHEN** `goodwin_oscillator` runs with three variables and Hill coefficients 6 and 12
- **THEN** it settles on its stable fixed point at 6 and on a limit cycle at 12, the threshold for three equal
  degradation rates being 8

### Requirement: Surveyed models are recorded in the inventory
The inventory SHALL record every model of Padmanabhan et al. 2021, Konieczny et al. 2023, and Raman 2021 under the
source keys `Pa`, `Ko`, and `Rm`: the models above as pending this change, models pending other changes, models blocked
on `add-constraint-based-models`, out-of-scope entries, and the new sources of models the library already implements.

#### Scenario: Source of an implemented model
- **WHEN** a contributor reads the inventory entry of `michaelis_menten`
- **THEN** its sources include `Ko 1.1.4` and `Rm 5.3.1`
