# Tasks

## 1. Dependencies and errors

- [ ] 1.1 Add `biopython>=1.88` to the runtime dependencies, the extra `rna` with `ViennaRNA>=2.7.2`, and the extra
  `trajectories` with `mdtraj>=1.11.2`. Relock, teach deptry the optional imports, and grow the Project Profile's
  dependency list in the same PR. Verify that the full test suite passes on the new lock
- [ ] 1.2 Add `MissingExtraError` as a `BackendError` unless `add-constraint-based-models` has added it, and verify the
  missing RNA extra and default install scenarios with the extras hidden from imports

## 2. Sequences, chains, and scores

- [ ] 2.1 Add sequences as categorical rows with the `'mutation'` move and string conversion, and verify the point
  mutation and letter outside the alphabet scenarios
- [ ] 2.2 Add `Score` with per-row Metropolis acceptance, temperature zero, and `proposals`, on NumPy and Numba, and
  reject a local energy and a score on one state with `IncompatibleLawsError`. Verify the neutral walk,
  domain-agnostic score, and positive temperature scenarios
- [ ] 2.3 Add `Chain` with the bead-chain NeRF and the protein backbone through Biopython's internal coordinates, and
  the dihedral rotation move. Verify the round trip through internal coordinates and freely rotating chain scenarios

## 3. Maps, structures, and analyses

- [ ] 3.1 Add `GenotypeMap` with its cache statistics, and `rna_fold` and `rna_layout` on ViennaRNA, and verify the
  hairpin fold, memoised map, and base-pair distance scenarios
- [ ] 3.2 Add `neutral_network` and the four robustness and evolvability functions, and verify the two toy map
  scenarios and the neighbours of an RNA sequence scenario
- [ ] 3.3 Add `Structure`, `read_structure`, `write_structure`, effective atoms, `Replay`, and `read_trajectory` on
  MDTraj, and bundle 1L2Y, 1UBQ, and 5PTI as gzipped mmCIF. Verify the NMR ensemble, replay, and round trip scenarios
  and a DCD trajectory read with the extra
- [ ] 3.4 Add dihedral angles, radius of gyration, end-to-end distance, superposed RMSD, and peptide plane angles, and
  verify the ideal helix and superposition scenarios
- [ ] 3.5 Add the fuzzy oil drop profile with `RD` and the fitted `K`, and its Numba kernel, and verify the ideal
  micelle, uniform hydrophobicity, native against extended, and unknown residue scenarios

## 4. Visuals

- [ ] 4.1 Add `ChainVisual` in three dimensions and in the plane with base pairs, and the categorical image of
  sequences, and verify the protein chain, secondary structure, and sequences as an image scenarios

## 5. Library and gallery

- [ ] 5.1 Change the three inventory entries from out of scope to `pending add-molecular-structures`, and verify that
  the inventory check passes against this change's specs
- [ ] 5.2 Add `rna_neutral_networks` to `biology/evolution`, and verify the neutral walk keeps the fold, robustness
  against evolvability, and adaptive walk scenarios
- [ ] 5.3 Add `fuzzy_oil_drop` to `biology/systems_biology` with the replay and folding variants, and verify the
  ensemble of the Trp-cage and folding scenarios
- [ ] 5.4 Transcribe the ellipse parameters of Brylinski et al. 2004 with their reference, add
  `early_stage_folding_intermediate` to `biology/systems_biology`, and verify the projection, path through the
  secondary structures, and sampled cells scenarios
- [ ] 5.5 Mark the three entries implemented, and verify the inventory models registered and model that needs the
  extra scenarios
- [ ] 5.6 Add the three gallery examples with their thumbnails, analysis plots, and captions, install the extra `rna`
  in the tests, docs, and gallery sessions, and verify the three example scenarios and the gallery coverage check
  within the budget

## 6. Documentation and gate

- [ ] 6.1 Document sequences, chains, scores, maps, structures, the analyses, the extras, and ViennaRNA's licence in
  the user guide, and verify that the documentation test passes
- [ ] 6.2 Add a CI job on a default install without the extras, and verify the default install stays importable
  scenario
- [ ] 6.3 Run formatting, checks, the docs build, and tests through nox, and verify that all of them pass
