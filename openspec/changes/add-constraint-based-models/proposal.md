# Proposal

## Why

The survey of Raman (2021) found seven models blocked on one missing feature: constraint-based modelling. A
constraint-based model is a linear system `S v = b` over flux variables `v` with bounds `l ≤ v ≤ u` and an objective,
solved as a linear program at a point in time (Raman 2021 §8.2, Eq. 8.3). Metabolism uses it as flux balance analysis.
The same structure describes flow networks, supply chains, and input-output economies. Raman uses it in three ways:
- statically: flux balance, flux variability, parsimonious fluxes, knock-outs, MoMA, ROOM, OptKnock, and synthetic
  lethals (ch. 8–10);
- coupled to dynamics: dynamic FBA with a solve per time step (§9.6, Eqs. 9.8–9.9), regulatory FBA alternating with
  Boolean gene rules (§9.4.1, Eq. 9.7), multi-species dynamics, and COMETS on a lattice with diffusing metabolites
  (§11.1.3.4);
- for communities: joint FBA, community FBA, and SteadyCom (§11.1.3, Eq. 11.3).

Genome-scale models come as SBML and BiGG JSON files. The framework has no decision variable, no linear program, and
no way to read these files.

Two more models of Raman were recorded as out of scope, because they are not integrated in time. They are first-class
models now, and they need two more general capabilities on the same networks:
- elementary flux modes and extreme pathways (§9.8) enumerate the generators of the flux cone `{v : S v = 0, v_irr ≥ 0}`.
  The same enumeration finds the simple cycles of a flow network or the conserved moieties of a reaction network;
- carbon-13 metabolic flux analysis (§9.7) simulates how labelled atoms of a tracer spread through a network with atom
  transitions, and estimates the fluxes by least squares from measured labelling patterns, with confidence intervals.
  The same simulation traces any labelled unit through a network of transformations.

**Rebased on `redesign-system-model`.** The redesign already provides several parts this change first planned to add:
- static time (`StaticTime`) and the `Objective` law kind in the constraint class, so a network alone is a static
  system and its objectives are `Objective` laws rather than a new `Solve` law;
- adaptive parameters (`parameter(adaptive=True)`), so switches are adaptive parameters set by algebraic `Equation`
  laws of interactions, and the separate `Switch` law is dropped;
- groups on sites (`Group(on='sites')` on a `LatticeSpace`), so a per-site network is a site group, not a network with
  its own `space=` argument;
- simulations that take observables, and the declarations and `validate()` pass, which report the invalid mixes the
  compiler used to reject.

This change still adds the constraint network and its linear solve, the `Bound` law, the coupling of solves to rates,
maps, partitions, and site groups, the enumeration of flux cones, labelling simulation and flux estimation, and their
visuals. The metabolism-specific analyses built on COBRApy, StrainDesign, and MICOM, and FSEOF and SteadyCom, move to
the biology realizations of the library, so the core keeps only general primitives.

## What Changes

- Add constraint networks, declared from the framework's symbols: pools, processes with stoichiometric coefficients
  and bounds, extra linear constraints, switches with Boolean rules that gate processes (gene-reaction rules in
  metabolism), and linear objectives, solved in sequence when there are several. A network is an `Element` in
  `core.components`. It contributes its balances as algebraic `Equation` laws, its objectives as `Objective` laws, and
  its switches as adaptive parameters with the domain `{0, 1}`.
- Add the `Bound` law in `core.laws`, a new law kind of the constraint class, so interactions set flux bounds from other
  elements, as a medium limits uptake. Switches are set from other elements, as a Boolean network regulates genes, by
  `Equation` laws on the adaptive switch parameters.
- Add the linear solve of networks under every time kind:
  - under `StaticTime`, a network alone is solved once per frame from its current parameters;
  - under `ContinuousTime`, a network in the same system as rate laws is solved inside the ODE right-hand side;
  - a network in a sibling nested system co-simulates, and its fluxes are held between synchronisation points, which
    is Raman's static optimisation approach;
  - a network element in a `Group(on='sites')` of a `LatticeSpace` is solved per site.
- Solve linear programs with HiGHS through `highspy` (MIT), a new runtime dependency, which keeps the default install
  free of copyleft code. `HighsLP` and `ConstraintStepper` in `core.solvers` declare their `Supports`, and every new law
  and component declares its `Requires`, so `validate()` reports misuse before compilation.
- Add an optional extra `metabolism` with COBRApy, StrainDesign, and MICOM, used by the library realizations in
  `skcomplex.models.biology.realizations`:
  - static analyses: FVA, pFBA, single and double deletions, MoMA, ROOM, and synthetic lethals;
  - strain design: OptKnock and minimal cut sets;
  - community models: cooperative trade-off;
  - import and export of SBML and BiGG JSON.
  These libraries are imported lazily. A missing extra raises a named error that names the extra.
- Add, in house on the core solve and without the extra: FSEOF and SteadyCom in the biology realizations, and joint
  community FBA by composition (`join_networks`) in `core.analysis`.
- Add Rerun visuals for fluxes: a bar chart per frame, a pool-process graph whose edges scale with the flux, and images
  of per-site fluxes.
- Enumerate elementary modes and extreme pathways of a network in exact arithmetic with the Parma Polyhedra Library
  through `pplpy` (GPL-3.0+), behind a new extra `polyhedra`, with a time limit and a named error. A network declared
  for enumeration runs as a static simulation whose frames are its modes.
- Add atom maps on processes, labelling simulation at isotopic steady state by elementary metabolite units, and flux
  estimation by weighted least squares with linearised and profile confidence intervals and a chi-square test. These are
  built in house on SciPy's sparse solvers and `least_squares`, because no maintained library installs from PyPI on
  Python 3.12–3.13. A labelling experiment runs as a static simulation whose frames are the iterations of the fit.
- Stream enumeration and estimation runs on a `mode` or `iteration` timeline instead of `sim_time`, and draw labelling
  fits as measured and simulated distributions with the residual series.
- Add library models with gallery examples: `flux_balance`, `metabolic_perturbations`, `regulatory_fba`,
  `dynamic_fba`, `elementary_flux_modes`, and `carbon_13_mfa` in `biology/systems_biology`, and
  `dynamic_multispecies_fba`, `comets`, and `community_fba` in `biology/ecology`.

## Capabilities

### New Capabilities

- `constraint-based-models`: constraint networks, their solve, the bound law and switches as adaptive parameters,
  dynamic and per-site coupling, the declarations and validation of these primitives, the enumeration of elementary
  modes and extreme pathways, and the optional extras.
- `isotope-labelling`: atom maps, labelling simulation, and flux estimation with intervals and a goodness-of-fit test.

### Modified Capabilities

- `model-compilation`: networks lower to a constraint problem, and decide how a network couples with rate and update
  laws. The enumerate and estimate laws lower to problems indexed by mode and iteration.
- `rerun-visualization`: flux visuals, timelines by mode or iteration, and labelling fit visuals.
- `model-library`: the nine constraint-based models, their gallery examples, and the metabolism realizations on the
  optional extra: static analyses, strain design, community trade-off, model import, FSEOF, and SteadyCom.

## Impact

- New parts of the core:
  - `skcomplex.core.components`: `ConstraintNetwork`, `Process`, `Constraint`, `FluxModes`, `AtomMap`, `Tracer`,
    `Fragment`, and `LabellingExperiment`;
  - `skcomplex.core.laws`: `Bound`, `Enumerate`, and `Estimate`, with the law kinds `BOUND`, `ENUMERATE`, and
    `ESTIMATE` of the constraint class in `core.base`, and the expression `total_flux`;
  - `skcomplex.core.compile`: `ConstraintProblem` and `LabellingProblem`;
  - `skcomplex.core.solvers`: `HighsLP`, `ConstraintStepper`, `ModeStepper`, `EmuSolver`, and `EstimationStepper`;
  - `skcomplex.core.viz`: `FluxVisual` and `LabellingVisual`;
  - `skcomplex.core.analysis`: `join_networks`, `elementary_modes`, `extreme_pathways`, `simulate_labelling`, and
    `estimate_fluxes`, with the `pplpy` adapter as a private module of the analysis package;
  - `skcomplex.core.base`: `MissingExtraError`, `InfeasibleError`, and `EnumerationLimitError`.
- New parts of the library, in `skcomplex.models.biology.realizations`, shared by `systems_biology` and `ecology`: the
  COBRApy, StrainDesign, and MICOM adapters, `fseof`, and `steadycom`.
- Dependencies:
  - `highspy>=1.15.1` becomes a runtime dependency, so the constitution's Project Profile must list it;
  - the new extra is `metabolism = ["cobra>=0.32.1", "straindesign>=1.19.1", "micom>=0.39.1"]`;
  - the second new extra is `polyhedra = ["pplpy>=0.9.0"]`, which brings GPL-3.0 PPL and LGPL-3.0 `gmpy2` and
    `cysignals`;
  - the SciPy floor rises from 1.13 to 1.16, the first release whose `least_squares` takes a `callback`;
  - mfapy (MIT), which is not on PyPI, joins the test dependency group from a pinned Git commit as the reference for
    labelling simulation;
  - `cobra` 0.32.1 caps `pandas<3.0`, so a lock with the extra resolves pandas 2.3.
- Uses co-simulation for held fluxes, site groups on lattices for COMETS, and the Boolean networks of
  `biology/systems_biology` for regulatory FBA. Depends on `redesign-system-model` for static time, adaptive
  parameters, site groups, and validation, and shares the trajectory `axis` and the static form with
  `add-static-models`. The models depend on `add-model-gallery` for their examples.
- The inventory entries `elementary_flux_modes` and `carbon_13_mfa` move from out of scope to implemented.
