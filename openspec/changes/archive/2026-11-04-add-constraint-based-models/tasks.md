# Tasks

## 1. Dependencies and errors

- [ ] 1.1 Add `highspy>=1.15.1` and `passagemath-polyhedra>=10.8.13` to the runtime dependencies and the `metabolism`
  extra with `cobra>=0.32.1`, `straindesign>=1.19.1`, and `micom>=0.39.1`, relock, which resolves pandas 2.3, teach
  deptry the optional imports, and grow the Project Profile's dependency list and Sage import rule in the same commit.
  Verify the full test suite passes on the new lock
- [x] 1.2 Add `InfeasibleError` as a `SolverError` and, unless `add-molecular-structures` has added it,
  `MissingExtraError` to `skcomplex.errors`, and verify the missing extra scenario with the extra hidden from imports

## 2. The linear program law

- [x] 2.1 Add `LinearObjectiveSpec` to `skcomplex.specs` and `LinearProgramLaw` to `core.laws` with doctests, requiring
  continuous, discrete, or static time, and verify that a law over a state that is not algebraic raises a named error
- [x] 2.2 Add the pass that merges the laws over shared variables, extracts the constant matrix, and compiles right
  sides, bounds, and activity on NumPy and Numba, with the issues `linear-program-form` and
  `linear-program-objectives`, and verify the nonlinear relation, two objectives, and two bounds scenarios
- [x] 2.3 Add the kernel on `highspy` with warm starts, lexicographic objectives, absolute value objectives, duals, the
  infeasibility policies, and the irreducible infeasible subset, and verify the toy network optimum, domain-agnostic
  flow network, lexicographic objectives, raised infeasibility, and zero policy scenarios
- [x] 2.4 Record variables, objective values, and requested duals in trajectories, and verify the recorded values
  scenario

## 3. Coupling

- [x] 3.1 Solve programs once per drift evaluation beside rate laws and once per step under discrete time, and verify
  the one solve per evaluation and the program with update laws scenarios
- [x] 3.2 Solve a program alone under static time in the static form of `add-static-models`, and verify the static
  program scenario
- [x] 3.3 Run a program's discrete-time partition beside an ODE partition with `CoSimulationSolver`, and verify the
  static optimisation step and agreement with the direct approach scenarios
- [x] 3.4 Solve programs per member of a site group where `active` holds, with per-member warm starts, and verify the
  only active sites and single site scenarios
- [x] 3.5 Set adaptive switches from Boolean nodes by `UpdateLaw`s, and verify the isozyme, live knock-out, and
  regulation gates a reaction scenarios

## 4. Visual and analysis

- [x] 4.1 Add `LinearProgramVisual` to `engine.viz`, and verify the bar chart, graph, and value per site scenarios
- [x] 4.2 Add `ElementaryModeAnalysis` with the kinds `'modes'` and `'pathways'`, the time limit, and
  `EnumerationLimitError` to `skcomplex.analysis`, and verify the toy network modes, optimum as a combination of modes,
  domain-agnostic circulation, knock-out removes modes, non-zero balance, pathways and modes, and genome-scale network
  scenarios

## 5. Metabolism and labelling in the models

- [x] 5.1 Add `biology/common/metabolism` with `MetabolicNetworkElement`, `ReactionSpec`, `from_cobra`, `to_cobra`,
  `load_network`, and `save_network`, and verify the bundled E. coli core and round trip scenarios, with a
  `network`-marked test that downloads iNJ661
- [x] 5.2 Add `CobraAnalysis` and `StrainDesignAnalysis`, and verify the flux variability, parity, adjustment after a
  knock-out, and growth-coupled succinate scenarios
- [x] 5.3 Add `ProductionScanAnalysis` and `CommunityAnalysis`, and verify the scan toward a product, identical members,
  and trade-off scenarios
- [x] 5.4 Add `AtomMapSpec`, `TracerSpec`, and `FragmentSpec` to `biology/systems_biology`, and verify the valid map,
  unbalanced letters, and inconsistent atom count scenarios
- [x] 5.5 Add `LabellingAnalysis` with the simulation kind, and verify the published TCA example, unlabelled tracer, and
  symmetric metabolite scenarios, with the mfapy parity test
- [x] 5.6 Add the estimation kind with seeded starts, linear and profile intervals, and the chi-square test, and verify
  the noise-free data, reproducible starts, data that cannot be fitted, noisy data, and unidentifiable exchange
  scenarios

## 6. Library and gallery

- [x] 6.1 Add `flux_balance`, `metabolic_perturbations`, `regulatory_fba`, `dynamic_fba`, `elementary_flux_modes`, and
  `carbon_13_mfa` to `biology/systems_biology`, and verify their scenarios
- [x] 6.2 Add `dynamic_multispecies_fba`, `comets`, and `community_fba` to `biology/ecology`, and verify their scenarios
- [x] 6.3 Add the law tag `linear-program` to the inventory vocabulary, set the nine inventory entries to implemented
  with the tags of the systems they build, and verify the inventory registered and models that need the extra scenarios
- [ ] 6.4 Add the nine gallery examples, install the extra in the `tests`, `docs`, and `gallery` sessions, and verify
  the dynamic FBA example, carbon-13 flux analysis example, and coverage scenarios within the budget

## 7. Documentation and gate

- [x] 7.1 Document linear program laws, merged programs, switches, the coupling forms, the issues, elementary modes, the
  metabolism family, labelling, and the licence of the extra's environment in the user guide, and verify the
  documentation test passes
- [x] 7.2 Add a CI job on a default install without the extra, and verify the default install stays importable scenario
- [ ] 7.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
