## Context

The subfield packages of `skcomplex.systems` split each system into a module that registers it and several one-class
modules for its elements, interactions, and views. A census on 2026-11-07 found 390 modules that register systems and
970 component modules: 784 serve one system, 39 serve several systems of one subfield, and 147 serve other components
or nothing. Six fields have a `common` package with 59 modules.

The census of the folding script, run on `development` once the open systems changes merged, follows imports through
chains of components, so a component reached only through other components counts for the system that reaches it:

| Private modules of the subfields | Count |
| --- | --- |
| Register at least one system | 408 |
| Serve one system | 856 |
| Serve several systems of one subfield | 45 |
| Serve no system | 6 |

| Modules of the `common` packages | Count |
| --- | --- |
| Serve systems of several subfields | 29 |
| Serve several systems of one subfield | 8 |
| Serve one system | 15 |
| Serve no system | 5 |

No component serves systems of two subfields outside `common`. One module imports across fields:
`physics.synchronization._chaos` imports the Lorenz element of `mathematics.common.flows`.

## Goals / Non-Goals

**Goals:**
- One module per system that holds everything only that system uses.
- Shared components of a field in its `common` package, and nowhere else.
- The same public names exported by every subfield package.
- Tests that mirror the new modules.

**Non-Goals:**
- Changing the behaviour, parameters, or registration of any system.
- Moving code into the core without the user's approval of the list of candidates.

## Decisions

- **Mechanical moves by a script, review by agents.** An AST-based script outside the repository builds the import
  graph of each subfield, folds every component module with one owning system into that system's module, rewrites the
  imports, and deletes the folded modules. Agents then review each touched module so it reads as one example, with
  classes ordered from elements to interactions to the factory, and fix what the script cannot, such as name clashes.
- **Ownership through chains.** A component imported only by other components belongs to the system that reaches it.
  When two systems reach it, it is shared.
- **Shared components.** A component shared by systems of one subfield moves to the field's `common` package when
  sharing makes sense, for example a population element that many ecology systems use. When the systems use it
  differently, each keeps a copy under a name that fits it. The agent decides per component and records the choice.
- **Existing `common` packages.** Their modules stay when several systems use them. A module used by one system moves
  into that system's module.
- **Cross-field duplicates.** After the moves, a script lists classes whose source is identical across fields. The
  list goes to the user with a proposal for each: move to the core under a mathematical name, or keep the copies.
- **Tests.** The test modules of folded component modules merge into the test module of the owning system. The layout
  test gains a check that no subfield module registers nothing while serving one system.
- **Waves.** One agent per field group, at most four at a time, each in its own worktree cut from `development`, in
  the order biology, physics, economics and social, then mathematics, engineering, computer science, and chemistry.

## Risks / Trade-offs

- **Merge conflicts with other work on systems:** the change starts after the open systems changes merge, and no other
  systems change runs while it does.
- **Long modules:** a system with many classes gets a long module. That is the intended trade, one complete example
  per file.
- **Lost history of moved code:** `git log --follow` does not follow a class into another file. The commit messages
  name the folded modules.

## Biology

- **Folded.** 252 component modules folded into the 85 system modules that alone use them.
- **Moved to `common`.** Components that several systems of one subfield share moved into the existing packages of
  `skcomplex.systems.biology.common`, and their subfields re-export them under the same names:
  `PredationInteraction` and `ReplicatorInteraction` to `populations`, `ThetaNeuronElement` and
  `SpikeResetInteraction` to `excitable`, `SpeciesElement` and `RepressionInteraction` to `kinetics`, and
  `HostElement`, `SpreadingInteraction`, `BooleanNodeElement`, and `BooleanFunctionInteraction` to `networks`.
- **Moved out of `common`.** `GompertzGrowthInteraction` and `PowerLawGrowthInteraction` served only the growth laws of
  oncology, so they moved into `oncology._growth`. `skcomplex.systems.biology.oncology` now exports them, and
  `common.populations` no longer does.
- **Kept as shared subfield modules.** The molecular structures of systems biology (`_structure_spec`, `_structures`,
  and `_backbone_geometry`) serve the fuzzy oil drop and the early-stage folding intermediate. In `common` they would
  have to be modules directly in the package, and importing any shared biology component would then load Biopython
  and MDTraj. `oncology._tumour_immune_chemotherapy` holds the element of the de Pillis and Radunskaya model, which its
  chemotherapy and its optimal therapy share, and is too specific to that one model for `common`.
- **Copies.** None.
- **Cross-field duplicates, candidates for the core.** No class or function has identical source in biology and
  another field. Ignoring names and docstrings, two bodies match: `CatalyticLinkElement` of `evolution._networks`
  equals `LinkElement` of `mathematics.networks`, `TieElement` of `social.opinion_dynamics`, and `SupervisorElement`
  of `social.organizations`, an element of an edge without states. `activity_grid` of `oncology._immune_competition`
  equals `wealth_classes` of `economics.econophysics` and `criminality_grid` of `social.crowds`, evenly spaced classes
  on the unit interval.
- **Left.** `common.metabolism._analyses`, `common.metabolism._community`, and `evolution._neutral_network_analysis`
  are analyses that no system imports, so they stay where they are.
