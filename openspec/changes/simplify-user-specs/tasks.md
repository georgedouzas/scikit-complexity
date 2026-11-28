## 1. Treatment

- [ ] 1.1 Check every public spec outside the engine against the rule and record its final treatment in design.md,
      flagging owners that would take more than about a dozen keywords
- [ ] 1.2 Rewrite the constitution rules on public values and variants and raise its version

## 2. Variants to options

- [ ] 2.1 Replace the matching specs with options of the matching interactions in `systems/general/matching`
- [ ] 2.2 Replace the rule specs, `SamplerSpec` and the objective specs with options of their laws
- [ ] 2.3 Replace `WienerProcessSpec`, `FiniteVolumeSpec` and `ScheduleSpec` with options of their owners

## 3. Records to plain values

- [ ] 3.1 Replace `AxisSpec`, `LayerSpec` and `VisualSpec` with parameters of spaces and elements
- [ ] 3.2 Replace the record specs of the laws, `DrawSpec`, `MoveSpec`, `PartnerSpec`, `MatchSpec`, `RequirementSpec`
      and `DatasetSpec`, with parameters or tuples
- [ ] 3.3 Replace `ScenarioSpec` and the record specs of rewriting, biology and physics with parameters, tuples or
      dicts
- [ ] 3.4 Move every check of a removed spec to the `_check` of its owner, with errors that name the owner

## 4. Private and results

- [ ] 4.1 Make `EntrySpec` and `SignatureSpec` private
- [ ] 4.2 Turn `NetworkSnapshotSpec`, `AgentSnapshotSpec` and `StructureStateSpec` into frozen `*Result` classes

## 5. Docs and close

- [ ] 5.1 Update every example, guide page and README section that builds a removed spec
- [ ] 5.2 Add the tests of public specs and of trajectory results, in the tests that mirror the packages
- [ ] 5.3 Run black, ruff, mypy, pydoclint, interrogate, lint-imports and the convention checks, the test groups of
      every touched package, and the full suite at the end
