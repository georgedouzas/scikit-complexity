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

## What Changes

- Add constraint networks, declared from the framework's symbols: pools, processes with stoichiometric coefficients
  and bounds, extra linear constraints, switches with Boolean rules that gate processes (gene-reaction rules in
  metabolism), and linear objectives, solved in sequence when there are several.
- Add bound and switch laws, so interactions set flux bounds and switch states from other elements. This is how a
  medium limits uptake and a Boolean network regulates genes.
- Add a solve law and a constraint form. A network alone is solved at every frame. A network with rate laws in the same
  subsystem is solved inside the ODE right-hand side. A network in a sibling subsystem co-simulates, and its fluxes are
  held between synchronisation points, which is Raman's static optimisation approach. A network on a lattice is solved
  per site.
- Solve linear programs with HiGHS through `highspy` (MIT), a new runtime dependency, which keeps the default install
  free of copyleft code.
- Add an optional extra `metabolism` with COBRApy, StrainDesign, and MICOM:
  - static analyses: FVA, pFBA, single and double deletions, MoMA, ROOM, and synthetic lethals;
  - strain design: OptKnock and minimal cut sets;
  - community models: cooperative trade-off;
  - import and export of SBML and BiGG JSON.
  These libraries are imported lazily. A missing extra raises a named error that names the extra.
- Add, in house on the default solver: flux scanning with enforced objective flux (FSEOF), SteadyCom, and joint
  community FBA by composition.
- Add Rerun visuals for fluxes: a bar chart per frame, a pool-process graph whose edges scale with the flux, and images
  of per-site fluxes.
- Enumerate elementary modes and extreme pathways of a network in exact arithmetic with the Parma Polyhedra Library
  through `pplpy` (GPL-3.0+), behind a new extra `polyhedra`, with a time limit and a named error. A network declared
  for enumeration runs as a simulation whose frames are its modes.
- Add atom maps on processes, labelling simulation at isotopic steady state by elementary metabolite units, and flux
  estimation by weighted least squares with linearised and profile confidence intervals and a chi-square test. These are
  built in house on SciPy's sparse solvers and `least_squares`, because no maintained library installs from PyPI on
  Python 3.12–3.13. A labelling experiment runs as a simulation whose frames are the iterations of the fit.
- Stream enumeration and estimation runs on a `mode` or `iteration` timeline instead of `sim_time`, and draw labelling
  fits as measured and simulated distributions with the residual series.
- Add library models with gallery examples: `flux_balance`, `metabolic_perturbations`, `regulatory_fba`,
  `dynamic_fba`, `elementary_flux_modes`, and `carbon_13_mfa` in `biology/systems_biology`, and
  `dynamic_multispecies_fba`, `comets`, and `community_fba` in `biology/ecology`.

## Capabilities

### New Capabilities

- `constraint-based-models`: constraint networks, their solve, the bound and switch laws, dynamic and per-site
  coupling, static analyses, strain design, communities, model import, the enumeration of elementary modes and extreme
  pathways, and the optional extras.
- `isotope-labelling`: atom maps, labelling simulation, and flux estimation with intervals and a goodness-of-fit test.

### Modified Capabilities

- `model-compilation`: the solve law gives a constraint form, and decides how a network couples with rate and update
  laws. The enumerate and estimate laws give the enumeration and estimation forms, indexed by mode and iteration.
- `rerun-visualization`: flux visuals, timelines by mode or iteration, and labelling fit visuals.
- `model-library`: the nine constraint-based models and their gallery examples.

## Impact

- New parts of the core:
  - `skcomplex.core.model`: `ConstraintNetwork`, `Process`, `Constraint`, `Solve`, `Bound`, `Switch`, `total_flux`,
    `FluxModes`, `Enumerate`, `AtomMap`, `Tracer`, `Fragment`, `LabellingExperiment`, and `Estimate`;
  - `skcomplex.core.compile`: `ConstraintProblem` and `LabellingProblem`;
  - `skcomplex.core.solvers`: `HighsLP`, `ConstraintStepper`, `ModeStepper`, `EmuSolver`, and `EstimationStepper`;
  - `skcomplex.core.viz`: `FluxVisual` and `LabellingVisual`;
  - `skcomplex.core.analysis`: `fseof`, `steadycom`, `join_networks`, `elementary_modes`, `extreme_pathways`,
    `simulate_labelling`, and `estimate_fluxes`;
  - `skcomplex.core.backends`: the COBRApy, StrainDesign, MICOM, and `pplpy` adapters;
  - `skcomplex.core.base`: `MissingExtraError`, `InfeasibleError`, and `EnumerationLimitError`.
- Dependencies:
  - `highspy>=1.15.1` becomes a runtime dependency, so the constitution's Project Profile must list it;
  - the new extra is `metabolism = ["cobra>=0.32.1", "straindesign>=1.19.1", "micom>=0.39.1"]`;
  - the second new extra is `polyhedra = ["pplpy>=0.9.0"]`, which brings GPL-3.0 PPL and LGPL-3.0 `gmpy2` and
    `cysignals`;
  - the SciPy floor rises from 1.13 to 1.16, the first release whose `least_squares` takes a `callback`;
  - mfapy (MIT), which is not on PyPI, joins the test dependency group from a pinned Git commit as the reference for
    labelling simulation;
  - `cobra` 0.32.1 caps `pandas<3.0`, so a lock with the extra resolves pandas 2.3.
- Uses co-simulation for held fluxes, lattice fields for COMETS, and the Boolean networks of `biology/systems_biology`
  for regulatory FBA. The models depend on `add-model-gallery` for their examples.
- The inventory entries `elementary_flux_modes` and `carbon_13_mfa` move from out of scope to implemented.
