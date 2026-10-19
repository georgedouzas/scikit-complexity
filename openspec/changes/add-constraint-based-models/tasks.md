# Tasks

## 1. Dependencies and errors

- [ ] 1.1 Add `highspy>=1.15.1` to the runtime dependencies and the `metabolism` extra with `cobra>=0.32.1`,
  `straindesign>=1.19.1`, and `micom>=0.39.1`. Relock, which resolves pandas 2.3 under COBRApy's cap, and teach deptry
  the optional imports. Grow the Project Profile's dependency list in the same PR. Verify that the full test suite
  passes on the new lock
- [ ] 1.2 Add `MissingExtraError` as a `BackendError` and `InfeasibleError` as a `SolverError` in the errors module,
  and verify the missing extra scenario with the extra hidden from imports

## 2. Networks and the core solve

- [ ] 2.1 Add `ConstraintNetwork`, `Process`, `Constraint`, `total_flux`, and `ConstraintNetwork.from_arrays` with
  schema validation, and verify the stoichiometric matrix, inverted bounds, and undeclared switch scenarios
- [ ] 2.2 Add `Bound`, `Switch`, and `Solve`, and the lowering to `ConstraintProblem` with compiled bound, switch, and
  rule functions on NumPy and Numba. Verify the isozyme, Michaelis-Menten uptake, and two bound laws scenarios
- [ ] 2.3 Add `HighsLP` with warm-started bound changes, lexicographic objectives, duals, and the irreducible
  infeasible subset. Verify the toy network optimum, domain-agnostic flow network, lexicographic objectives, raised
  infeasibility, and zero policy scenarios
- [ ] 2.4 Record fluxes, objectives, switches, and requested duals in trajectories, and verify the flux series
  scenario

## 3. Forms and coupling

- [ ] 3.1 Extend form inference with the constraint form and `ConstraintStepper`, and verify the static network, live
  knock-out, and modified form scenarios
- [ ] 3.2 Solve once per drift evaluation for networks beside rate laws, and verify the one solve per evaluation and
  network with rate laws scenarios
- [ ] 3.3 Add constraint partitions with held fluxes to co-simulation, and verify the static optimisation step and
  agreement with the direct approach scenarios
- [ ] 3.4 Add per-site networks with the activity condition and per-site warm starts, and verify the only occupied
  sites and single site scenarios
- [ ] 3.5 Couple switch laws with a map partition of Boolean nodes, and verify the regulation gates a process scenario

## 4. Analyses

- [ ] 4.1 Add `to_cobra`, `from_cobra`, `read_network`, and `write_network` with the `hybrid` solver pinned, and verify
  the bundled E. coli core and round trip scenarios. Add a `network`-marked test that downloads iNJ661
- [ ] 4.2 Add `flux_variability`, `parsimonious_fluxes`, `deletions`, `moma`, `room`, and `synthetic_lethals`, and
  verify the flux variability, parity, and adjustment after a knock-out scenarios
- [ ] 4.3 Add `strain_designs` on StrainDesign with SCIP, and verify the growth-coupled succinate scenario
- [ ] 4.4 Add `fseof`, `steadycom`, and `join_networks` in `skcomplex.core.analysis`, and `cooperative_tradeoff` in
  the backends. Verify the scan toward a product, identical members, and trade-off scenarios

## 5. Visuals

- [ ] 5.1 Add `FluxVisual` with the bar chart, objective series, and seeded graph layout, and make `ImageVisual` the
  default for lattice networks. Verify the flux bar chart, flux graph, and growth per site scenarios

## 6. Library and gallery

- [ ] 6.1 Add `flux_balance` and `metabolic_perturbations` to `biology/systems_biology`, and verify their scenarios
- [ ] 6.2 Add `dynamic_fba` and `regulatory_fba` to `biology/systems_biology`, and verify their scenarios
- [ ] 6.3 Add `dynamic_multispecies_fba`, `comets`, and `community_fba` to `biology/ecology`, and verify their
  scenarios
- [ ] 6.4 Move the seven inventory entries from pending to implemented with their sources in Raman 2021. Verify the
  inventory registered and models that need the extra scenarios
- [ ] 6.5 Add the seven gallery examples with their thumbnails, analysis plots, and captions. Install the extra in the
  tests, docs, and gallery sessions, and verify the dynamic FBA example and coverage scenarios within the budget

## 7. Documentation and gate

- [ ] 7.1 Document networks, switches, bound laws, the coupling forms, the analyses, the extra, and the licence of its
  environment in the user guide, and verify that the documentation test passes
- [ ] 7.2 Add a CI job on a default install without the extra, and verify the default install stays importable
  scenario
- [ ] 7.3 Run formatting, checks, the docs build, and tests through nox, and verify that all of them pass
