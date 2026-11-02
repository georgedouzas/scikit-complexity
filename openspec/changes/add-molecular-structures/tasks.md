# Tasks

## 1. Dependencies and errors

- [ ] 1.1 Add the extras `structures = ["biopython>=1.88"]`, `rna = ["ViennaRNA>=2.7.2"]`, and
  `trajectories = ["mdtraj>=1.11.2"]` to `pyproject.toml`, teach deptry the optional imports, list them in the Project
  Profile, relock, and verify `pdm install`, `deptry`, and `pip-audit` pass
- [ ] 1.2 Add `MissingExtraError` to `skcomplex.errors` unless `add-constraint-based-models` has added it, and verify
  the default install stays importable scenario with the extras hidden from imports

## 2. Core

- [ ] 2.1 Add `RowScoreSpec` to `skcomplex.specs` and its acceptance per row as a private method of
  `SamplingInteraction`, with the row scores recorded as observables, and verify the neutral walk, domain-agnostic
  score, score at a positive temperature, and point mutation as a function move scenarios
- [ ] 2.2 Widen the intensity domain of `SamplingInteraction` to `(0, oo]` with the zero-temperature Metropolis rule,
  and verify the zero temperature on a local energy scenario
- [ ] 2.3 Pass `time` to a `DriveLaw` function that names it, and verify the replay of an ensemble scenario
- [ ] 2.4 Add `ShapeAnalysis` to `skcomplex.analysis`, and verify the freely rotating chain and superposition scenarios
- [ ] 2.5 Add `strip` to `PointsVisual`, and verify the protein chain in three dimensions scenario with a recorded
  Rerun stream

## 3. Biology packages

- [ ] 3.1 Add the memoised RNA fold, base-pair distance, and plane layout over ViennaRNA and `NeutralNetworkAnalysis`
  to `skcomplex.models.biology.evolution`, and verify the hairpin fold and toy map scenarios
- [ ] 3.2 Add `read_structure`, `write_structure`, `read_trajectory`, `StructureSpec`, effective atoms, and the backbone
  geometry to `skcomplex.models.biology.systems_biology` with 1L2Y, 1UBQ, and 5PTI bundled as gzipped mmCIF, and verify
  the NMR ensemble, round trip, and round trip through internal coordinates scenarios and a DCD read with the extra
- [ ] 3.3 Add `FuzzyOilDropAnalysis` and `BackboneVisual` to the systems biology package, and verify the ideal micelle
  and uniform hydrophobicity, native against extended, and unknown residue scenarios
- [ ] 3.4 Verify the field-term and import contracts scenario

## 4. Library and gallery

- [ ] 4.1 Add `rna_neutral_networks` to `biology/evolution`, and verify the neutral walk, adaptive walk, robustness
  against evolvability, and model that needs the extra scenarios and the conformance suite
- [ ] 4.2 Add `fuzzy_oil_drop` with the replay and folding dynamics, and verify the ensemble of the Trp-cage and folding
  scenarios, the protein model that needs the extra scenario, and the conformance suite
- [ ] 4.3 Transcribe the ellipse parameters of Brylinski et al. 2004 with their reference, add
  `early_stage_folding_intermediate`, and verify the projection, sampled cells, and protein model that needs the extra
  scenarios and the conformance suite
- [ ] 4.4 Mark the three entries implemented in `docs/overview/inventory.toml`, add their gallery examples, install the
  extras `structures` and `rna` in the tests, docs, and gallery sessions, and verify the sequences as an image
  scenario, the inventory check, and the gallery coverage check within the budget

## 5. Documentation and gate

- [ ] 5.1 Document row scores, zero temperature, drives that read the step, `ShapeAnalysis`, the extras, the source
  build of Biopython on Intel Macs, and ViennaRNA's licence in the user guide, add the changelog entry, and verify the
  documentation test and the changelog session pass
- [ ] 5.2 Add a CI job on a default install without the extras, and verify it passes
- [ ] 5.3 Run `nox -s formatting`, `nox -s checks`, `nox -s docs`, and `nox -s tests` in that order, and verify all of
  them pass
