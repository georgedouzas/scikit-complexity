# Tasks

## 1. Inventory

- [x] 1.1 Record the sources `Pa`, `Ko`, and `Rm`, every surveyed model with its status, and the new sources of
  implemented models in the inventory, and verify the inventory check passes

## 2. Model groups

For every task in this section:
- implement each model as a factory on shared components of its subfield package, built only from public core names;
- verify its scenario and a behavioural check per model;
- add its gallery example with a Rerun thumbnail and an analysis plot, as `add-model-gallery` requires;
- mark its inventory entry implemented.

- [x] 2.1 Tumour growth laws: exponential_linear_growth, power_law_growth, von_bertalanffy, generalized_logistic,
  gompertz, dynamic_carrying_capacity
- [x] 2.2 Cell kill and toxicity: skipper_log_kill, norton_simon, drug_toxicity
- [x] 2.3 Chemotherapy: de_pillis_radunskaya, multidrug_resistance, drug_resistance_switching,
  delayed_resting_hunting_chemotherapy, delayed_metastasis, cell_cycle_chemotherapy, erythropoiesis_mercaptopurine
- [x] 2.4 Immunotherapy: kirschner_panetta, dendritic_cell_vaccine, anti_pd1_checkpoint, delayed_dc_vaccine,
  tumour_immune_suppression
- [x] 2.5 Anti-angiogenic therapy: hahnfeldt, vascular_core_periphery
- [x] 2.6 Radiotherapy: radiotherapy_competition, lq_radiosensitizer
- [x] 2.7 Hormone therapy: continuous_androgen_suppression, ideta_intermittent_androgen, prostate_cell_quota,
  droop_cell_quota, prostate_androgen_receptor, piecewise_linear_ias, tamoxifen_cell_cycle
- [x] 2.8 Gene, viral, nanoparticle, and stem-cell therapies: tcr_gene_therapy, sirna_tgf_beta, oncolytic_virus,
  oncolytic_virus_dendritic, nanoparticle_corona, two_compartment_pharmacokinetics, stem_cell_differentiation,
  bone_marrow_blood_leukaemia, leukaemia_stem_cells
- [x] 2.9 Combination therapies: chemoimmunotherapy, clonal_chemoimmunotherapy, chemovirotherapy,
  glioma_chemovirotherapy, her2_targeted_therapy, angiogenesis_immunotherapy
- [x] 2.10 Cellular network models: phosphorylation_oscillator, biochemical_systems_theory, three_gene_boolean_network,
  mammalian_cell_cycle, boolean_repressilator, goodwin_oscillator, generalized_lotka_volterra

## 3. Gate

- [x] 3.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
