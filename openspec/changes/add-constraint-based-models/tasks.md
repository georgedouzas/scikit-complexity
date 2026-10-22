# Tasks

## 1. Dependencies and errors

- [ ] 1.1 Add `highspy>=1.15.1` to the runtime dependencies and the `metabolism` extra with `cobra>=0.32.1`,
  `straindesign>=1.19.1`, and `micom>=0.39.1`. Relock, which resolves pandas 2.3 under COBRApy's cap, and teach deptry
  the optional imports. Grow the Project Profile's dependency list in the same PR. Verify that the full test suite
  passes on the new lock
- [ ] 1.2 Add `MissingExtraError` as a `BackendError` and `InfeasibleError` as a `SolverError` in the errors module,
  and verify the missing extra scenario with the extra hidden from imports
- [ ] 1.3 Add the `polyhedra` extra with `pplpy>=0.9.0`, raise the SciPy floor to 1.16, add mfapy pinned to a Git
  commit in the test group, and teach deptry the optional import. Check the macOS version of the CI runners against the
  macOS 15 wheels, and verify the missing polyhedra extra scenario with `ppl` hidden from imports

## 2. Networks and the core solve

- [ ] 2.1 Add `ConstraintNetwork`, `Process`, and `Constraint` in `core.components`, `total_flux`, and
  `ConstraintNetwork.from_arrays` with schema validation, with switches as adaptive parameters of the domain `{0, 1}`
  and the network's balances and objectives contributed as `Equation` and `Objective` laws. Verify the stoichiometric
  matrix, inverted bounds, and undeclared switch scenarios
- [ ] 2.2 Add `Bound` in `core.laws` with the law kind `BOUND` of the constraint class, and the lowering to
  `ConstraintProblem` with compiled bound, switch, and rule functions on NumPy and Numba. Verify the isozyme,
  Michaelis-Menten uptake, switch set by an equation, and two bound laws scenarios
- [ ] 2.3 Add `HighsLP` with warm-started bound changes, lexicographic objectives, duals, and the irreducible
  infeasible subset. Verify the toy network optimum, domain-agnostic flow network, lexicographic objectives, raised
  infeasibility, and zero policy scenarios
- [ ] 2.4 Record fluxes, objectives, switches, and requested duals in trajectories, and verify the flux series
  scenario
- [ ] 2.5 Add the `Requires` declarations of `ConstraintNetwork`, `Bound`, `Enumerate`, and `Estimate`, and the
  `Supports` declarations of `HighsLP`, `ConstraintStepper`, `ModeStepper`, `EmuSolver`, and `EstimationStepper`, with
  the validation issues of the design. Verify the bound without a network, network on edges, and enumeration beside
  dynamics scenarios, and that the declaration test passes

## 3. Forms and coupling

- [ ] 3.1 Lower networks under static time to the constraint form with `ConstraintStepper`, and verify the static
  network, live knock-out, and modified form scenarios
- [ ] 3.2 Solve once per drift evaluation for networks beside rate laws, and verify the one solve per evaluation and
  network with rate laws scenarios
- [ ] 3.3 Add constraint partitions of nested systems with held fluxes to co-simulation, and verify the static
  optimisation step and agreement with the direct approach scenarios
- [ ] 3.4 Add networks in site groups with the activity condition and per-site warm starts, and verify the only
  occupied sites and single site scenarios
- [ ] 3.5 Couple switch equations with a map partition of Boolean nodes, and verify the regulation gates a process
  scenario

## 4. Analyses

- [ ] 4.1 Add `skcomplex.models.biology.realizations` with `to_cobra`, `from_cobra`, `read_network`, and
  `write_network`, with the `hybrid` solver pinned, and verify the bundled E. coli core and round trip scenarios. Add a
  `network`-marked test that downloads iNJ661
- [ ] 4.2 Add to the biology realizations `flux_variability`, `parsimonious_fluxes`, `deletions`, `moma`, `room`, and `synthetic_lethals`, and
  verify the flux variability, parity, and adjustment after a knock-out scenarios
- [ ] 4.3 Add `strain_designs` to the biology realizations on StrainDesign with SCIP, and verify the growth-coupled succinate scenario
- [ ] 4.4 Add `join_networks` in `skcomplex.core.analysis`, and `fseof`, `steadycom`, and `cooperative_tradeoff` in
  the biology realizations. Verify the scan toward a product, identical members, and trade-off scenarios, and that the
  core's field-term and import-graph tests still pass

## 5. Enumeration and labelling

- [ ] 5.1 Add `extreme_rays` in `skcomplex.core.analysis._cones` with the `cysignals` time limit and
  `EnumerationLimitError`, and verify the domain-agnostic circulation and genome-scale network scenarios
- [ ] 5.2 Add `elementary_modes` and `extreme_pathways` in `skcomplex.core.analysis`, and verify the toy network modes,
  optimum as a combination of modes, knock-out removes modes, non-zero balance, and pathways and modes scenarios
- [ ] 5.3 Add `FluxModes`, `Enumerate` with the law kind `ENUMERATE`, its lowering under static time, `ModeStepper`,
  and the trajectory `axis` unless `add-static-models` has added it, and verify the modes as frames and enumerated
  network scenarios
- [ ] 5.4 Add `AtomMap`, `Tracer`, and `Fragment` with schema validation, and verify the valid map, unbalanced letters,
  and inconsistent atom count scenarios
- [ ] 5.5 Add the EMU lowering to `LabellingProblem`, `EmuSolver`, and `simulate_labelling`, and verify the published
  TCA example, unlabelled tracer, and symmetric pool scenarios, with the mfapy parity test
- [ ] 5.6 Add `LabellingExperiment`, `Estimate`, `estimate_fluxes` with seeded HiGHS starts, linear and profile
  intervals, and the chi-square test, and verify the noise-free data, reproducible starts, data that cannot be fitted,
  noisy data, and unidentifiable exchange scenarios
- [ ] 5.7 Add `Estimate` with the law kind `ESTIMATE`, its lowering under static time, and `EstimationStepper` on the `least_squares` callback, and verify the iterations as
  frames, labelling experiment, and estimation beside dynamics scenarios

## 6. Visuals

- [ ] 6.1 Add `FluxVisual` with the bar chart, objective series, and seeded graph layout, and make `ImageVisual` the
  default for lattice networks. Verify the flux bar chart, flux graph, and growth per site scenarios
- [ ] 6.2 Log runs indexed by mode or iteration on their own sequence timeline with a matching time panel, and add
  `LabellingVisual`. Verify the modes on a timeline and fit converges in the viewer scenarios

## 7. Library and gallery

- [ ] 7.1 Add `flux_balance` and `metabolic_perturbations` to `biology/systems_biology`, and verify their scenarios
- [ ] 7.2 Add `dynamic_fba` and `regulatory_fba` to `biology/systems_biology`, and verify their scenarios
- [ ] 7.3 Add `dynamic_multispecies_fba`, `comets`, and `community_fba` to `biology/ecology`, and verify their
  scenarios
- [ ] 7.4 Add `elementary_flux_modes` and `carbon_13_mfa` to `biology/systems_biology`, and verify the toy network
  modes and the FBA optimum, yield of a knock-out, recovered fluxes, and published labelling pattern scenarios
- [ ] 7.5 Move the seven inventory entries from pending to implemented, and `elementary_flux_modes` and
  `carbon_13_mfa` from out of scope to implemented, with their sources in Raman 2021. Verify the inventory registered,
  models that need the extra, and models outside the inventory's earlier scope scenarios
- [ ] 7.6 Add the nine gallery examples with their thumbnails, analysis plots, and captions. Install both extras in the
  tests, docs, and gallery sessions, and verify the dynamic FBA example, carbon-13 flux analysis example, and coverage
  scenarios within the budget

## 8. Documentation and gate

- [ ] 8.1 Document networks, switches as adaptive parameters, bound laws, the declarations and validation issues, the coupling forms, the analyses, elementary modes and extreme
  pathways, atom maps, labelling simulation, flux estimation, runs indexed by mode or iteration, the extras, and the
  licences of their environments in the user guide, and verify that the documentation test passes
- [ ] 8.2 Add a CI job on a default install without the extras, and verify the default install stays importable
  scenario
- [ ] 8.3 Run formatting, checks, the docs build, and tests through nox, and verify that all of them pass
