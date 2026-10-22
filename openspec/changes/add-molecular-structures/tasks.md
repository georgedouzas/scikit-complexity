# Tasks

## 1. Dependencies and errors

- [ ] 1.1 Add `biopython>=1.88` to the runtime dependencies, the extra `rna` with `ViennaRNA>=2.7.2`, and the extra
  `trajectories` with `mdtraj>=1.11.2`. Relock, teach deptry the optional imports, and grow the Project Profile's
  dependency list in the same PR. Verify that the full test suite passes on the new lock
- [ ] 1.2 Add `MissingExtraError` as a `BackendError` unless `add-constraint-based-models` has added it, and verify the
  missing RNA extra and default install scenarios with the extras hidden from imports

## 2. Sequences, chains, scores, and replay in the core

- [ ] 2.1 Add sequences as categorical rows of a lattice site group, the `'mutation'` move kind of `Sample`, and string
  conversion, and verify the point mutation and letter outside the alphabet scenarios
- [ ] 2.2 Add global scores to `Sample` with per-row Metropolis acceptance, temperature zero, and `proposals`, on NumPy
  and Numba, and verify the neutral walk, domain-agnostic score, and positive temperature scenarios
- [ ] 2.3 Add `Chain` to `core.components` with the bead-chain NeRF, geometry objects, and the dihedral rotation move,
  and verify the freely rotating chain scenario
- [ ] 2.4 Add the replay law to `core.laws`, and verify the replay of an ensemble scenario
- [ ] 2.5 Declare the requirements of `Chain`, the global score, and the replay law, extend the `Supports` of the NumPy
  and Numba backends, and add the validation issues; verify the chain on a square lattice, local energy and score,
  Python score on Numba, and replay under continuous time scenarios

## 3. Biology realizations, maps, and analyses

- [ ] 3.1 Add `GenotypeMap` with its cache statistics, and `rna_fold` and `rna_layout` on ViennaRNA, to the
  `biology/evolution` model package, and verify the hairpin fold, memoised map, and base-pair distance scenarios
- [ ] 3.2 Add `neutral_network` and the four robustness and evolvability functions to the `biology/evolution` package,
  and verify the two toy map scenarios and the neighbours of an RNA sequence scenario
- [ ] 3.3 Add `Structure`, `read_structure`, `write_structure`, effective atoms, `read_trajectory` on MDTraj, and the
  protein backbone geometry to `skcomplex.models.biology.realizations`, and bundle 1L2Y, 1UBQ, and 5PTI as gzipped
  mmCIF. Verify the NMR ensemble, round trip, and round trip through internal coordinates scenarios and a DCD
  trajectory read with the extra
- [ ] 3.4 Add radius of gyration, end-to-end distance, and superposed RMSD to `core.analysis`, and backbone dihedral
  and peptide plane angles to the biology realizations, and verify the ideal helix and superposition scenarios
- [ ] 3.5 Add the fuzzy oil drop profile with `RD` and the fitted `K`, and its Numba kernel, to the biology
  realizations, and verify the ideal micelle, uniform hydrophobicity, native against extended, and unknown residue
  scenarios
- [ ] 3.6 Verify the field-term and import-graph tests of the core scenario

## 4. Visuals

- [ ] 4.1 Add `ChainVisual` in three dimensions and in the plane with base pairs, and the categorical image of
  sequences, and verify the protein chain, secondary structure, and sequences as an image scenarios

## 5. Library and gallery

- [ ] 5.1 Change the three inventory entries from out of scope to `pending add-molecular-structures`, and verify that
  the inventory check passes against this change's specs
- [ ] 5.2 Add `rna_neutral_networks` to `biology/evolution` with its catalog observables, and verify the neutral walk
  keeps the fold, robustness against evolvability, and adaptive walk scenarios
- [ ] 5.3 Add `fuzzy_oil_drop` to `biology/systems_biology` with the replay and folding variants and its catalog
  observables, and verify the ensemble of the Trp-cage and folding scenarios
- [ ] 5.4 Transcribe the ellipse parameters of Brylinski et al. 2004 with their reference, add
  `early_stage_folding_intermediate` to `biology/systems_biology` on `Sample` with heat-bath acceptance, and verify the
  projection, path through the secondary structures, and sampled cells scenarios
- [ ] 5.5 Mark the three entries implemented, and verify the inventory models registered and model that needs the
  extra scenarios
- [ ] 5.6 Add the three gallery examples with their thumbnails, analysis plots, and captions, install the extra `rna`
  in the tests, docs, and gallery sessions, and verify the three example scenarios and the gallery coverage check
  within the budget

## 6. Documentation and gate

- [ ] 6.1 Document sequences, chains, scores, replay, the biology realizations, the analyses, the extras, and
  ViennaRNA's licence in the user guide, and verify that the documentation test passes
- [ ] 6.2 Add a CI job on a default install without the extras, and verify the default install stays importable
  scenario
- [ ] 6.3 Run formatting, checks, the docs build, and tests through nox, and verify that all of them pass
