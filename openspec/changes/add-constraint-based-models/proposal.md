# Proposal

## Why

The survey of Raman (2021) found seven models blocked on one missing feature: constraint-based modelling. A
constraint-based model is a linear system `S v = b` over flow variables `v` with bounds `l ≤ v ≤ u` and an objective,
solved as a linear program at a point in time (Raman 2021 §8.2, Eq. 8.3). Metabolism uses it as flux balance analysis.
The same structure describes flow networks, supply chains, and input-output economies. Raman uses it in three ways:
- statically: flux balance, flux variability, parsimonious fluxes, knock-outs, MoMA, ROOM, OptKnock, and synthetic
  lethals (ch. 8 to 10),
- coupled to dynamics: dynamic FBA with a solve per time step (§9.6, Eqs. 9.8 and 9.9), regulatory FBA alternating with
  Boolean gene rules (§9.4.1, Eq. 9.7), multi-species dynamics, and COMETS on a lattice with diffusing metabolites
  (§11.1.3.4),
- for communities: joint FBA, community FBA, and SteadyCom (§11.1.3, Eq. 11.3).

Two more models of Raman, elementary flux modes (§9.8) and carbon-13 metabolic flux analysis (§9.7), are recorded in
the inventory as blocked on this change. One enumerates the generators of the cone `{v : S v = 0, v_irr ≥ 0}`, which
also finds the simple cycles of a flow network, and the other fits fluxes to labelling data.

The current code already provides the ground these need, so this change no longer adds it:
- algebraic states, which equations alone determine, come from `add-acausal-components`, and the flows of a program
  are such states,
- bounded adaptive parameters (`ParameterSpec(adaptive=True)`) hold switches, and `UpdateLaw`s set them,
- `GroupElement(on='sites')` on a `LatticeSpace` puts one copy of an element on every site, `DiffusionInteraction`
  diffuses metabolites, and `CoSimulationSolver` runs an ODE partition beside a discrete-time partition with held
  inputs,
- `CallSpec` values, the issues of `PassCompiler`, `ImageVisual` for values per site, and the analysis layer
  `skcomplex.analysis`, whose surface is classes, run what the rest needs,
- the Boolean gene networks of `biology/systems_biology` give regulation.

So the plan drops the `ConstraintNetwork` component, the `Bound`, `Objective`, `Enumerate`, and `Estimate` laws and
their law kinds, the solvers `HighsLP`, `ConstraintStepper`, `ModeStepper`, `EmuSolver`, and `EstimationStepper`, frames
indexed by mode or iteration or a shared trajectory axis, and the public functions `join_networks`, `elementary_modes`,
`extreme_pathways`, `simulate_labelling`, and `estimate_fluxes`. Isotope labelling is a field case, so it moves to the
one model package that uses it.

## What Changes

- Add `LinearProgramLaw` in `core.laws`: it sets algebraic states, its variables, to an optimal solution of a linear
  program whose constraints are Sage relations linear in the variables and whose objectives, `LinearObjectiveSpec`s of
  `skcomplex.specs`, are solved in order. The programs of laws that share variables merge, so an interaction adds
  constraints, such as an uptake bound, to the program of an element.
- Lower the programs in `skcomplex.engine`: a compile pass extracts the constant matrix and compiles the right sides and
  bounds, and a kernel solves them with HiGHS through `highspy`, warm-started, once per evaluation of an ODE right side
  or per step of a map, per member of a group, and only where an activity condition holds.
- Add `InfeasibleError` and `MissingExtraError` to `skcomplex.errors`, and the issues `linear-program-form` and
  `linear-program-objectives` to `PassCompiler`.
- Add `LinearProgramVisual` in `skcomplex.engine.viz`: a bar chart of the variables, the objective values, and a graph
  of constraints and variables.
- Add `ElementaryModeAnalysis` in `skcomplex.analysis`, with the kinds `'modes'` and `'pathways'`, which enumerates the
  generators of the steady-state cone of a program exactly with `passagemath-polyhedra`.
- Add the family `skcomplex.models.biology.common.metabolism`, shared by `systems_biology` and `ecology`: the
  metabolic network element on `LinearProgramLaw`, model import and export, and analyses on COBRApy, StrainDesign, and
  MICOM behind the optional extra `metabolism`, with FSEOF and SteadyCom in house.
- Add isotope labelling to `skcomplex.models.biology.systems_biology`: atom maps, tracers, and fragments, and
  `LabellingAnalysis`, which simulates labelling by elementary metabolite units and estimates fluxes with intervals and
  a goodness-of-fit test.
- Add the models `flux_balance`, `metabolic_perturbations`, `regulatory_fba`, `dynamic_fba`, `elementary_flux_modes`,
  and `carbon_13_mfa` in `biology/systems_biology`, and `dynamic_multispecies_fba`, `comets`, and `community_fba` in
  `biology/ecology`, with gallery examples.

## Capabilities

### New Capabilities

- `constraint-based-models`: linear program laws, their merging, solve, coupling to dynamics, members, and validation,
  the elementary mode analysis, and the optional extra.
- `isotope-labelling`: atom maps, labelling simulation, and flux estimation in the model library.

### Modified Capabilities

- `model-compilation`: linear program laws keep the form of the system that holds them.
- `rerun-visualization`: the visual of linear programs.
- `model-library`: the nine constraint-based models and the metabolism family.

## Impact

- `skcomplex.core.laws`: `LinearProgramLaw`. `skcomplex.specs`: `LinearObjectiveSpec`. `skcomplex.errors`:
  `InfeasibleError` and `MissingExtraError`.
- `skcomplex.engine`: the linear program pass and kernel, the issues, and `LinearProgramVisual`.
- `skcomplex.analysis`: `ElementaryModeAnalysis`.
- `skcomplex.models.biology`: `common/metabolism`, the labelling of `systems_biology`, and the nine models, with
  `common/metabolism` among the families of the field.
- Dependencies:
  - `highspy>=1.15.1` and `passagemath-polyhedra>=10.8.13` become runtime dependencies, since the engine and the
    analysis layer use them, and the Project Profile lists them and widens the Sage import rule to
    `sage.all__sagemath_polyhedra`,
  - the new extra is `metabolism = ["cobra>=0.32.1", "straindesign>=1.19.1", "micom>=0.39.1"]`, which only the models
    use,
  - mfapy (MIT), which is not on PyPI, joins the test dependency group from a pinned Git commit as the reference for
    labelling simulation,
  - `cobra` 0.32.1 caps `pandas<3.0`, so a lock with the extra resolves pandas 2.3.
- Depends on `add-acausal-components` for algebraic states, or adds them first if it lands first, and on
  `add-static-models` for the static form `FormKind.STATIC` in which a program alone under `StaticTime` solves. Sweeps
  of a bound are an `Experiment` over a `GridDesign`, so no trajectory axis is shared.
- `MissingExtraError` is added by whichever of `add-molecular-structures` and this change lands first.
