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
- **No system module imports another.** After the fields were folded, five system modules still took a definition from
  a sibling system module, and a layout test now fails on any such import within a subfield. Constants and small
  helpers are copied: the reference `BOOK` of oncology into `_optimal_therapy`, like every other oncology module, the
  image `SPIN_IMAGE` into `_spin_glasses`, and the double-well forcing helpers of `_brownian` into `_diffusion`, whose
  Kramers escape uses them. Classes that two systems use the same way move to the family of `common` they belong to,
  with no new package: `ToyNetworkElement` with its reactions to `biology.common.metabolism`, where the metabolic
  network it specializes lives, and `HarmonicForceInteraction` to `physics.common.particles`, beside the constant
  force and the damping. Each subfield re-exports the moved class, so its `__all__` is unchanged.
- **Waves.** One agent per field group, at most four at a time, each in its own worktree cut from `development`, in
  the order biology, physics, economics and social, then mathematics, engineering, computer science, and chemistry.

## Risks / Trade-offs

- **Merge conflicts with other work on systems:** the change starts after the open systems changes merge, and no other
  systems change runs while it does.
- **Long modules:** a system with many classes gets a long module. That is the intended trade, one complete example
  per file.
- **Lost history of moved code:** `git log --follow` does not follow a class into another file. The commit messages
  name the folded modules.

## Economics

- **Moved to `economics.common`:** `AssetElement` and `GeometricBrownianMotionInteraction`, which
  `geometric_brownian_motion` and `correlated_geometric_brownian_motion` of finance use in the same way,
  `BalanceSheetElement` with its view and status constants, which `default_cascade`, `eisenberg_noe_clearing`, and
  `overlapping_portfolio_fire_sales` of finance share, and `ExchangeInteraction` with `cobb_douglas`, `marginal_rate`,
  and `trade_goods`, which `barter` and `goods_traders` of markets share. Each keeps its own module directly in
  `common`, and finance and markets still export these names.
- **Kept in `common`:** the `forecasting` and `resources` packages, since several subfields use them.
- **Copies:** none.
- **Cross-field duplicates:** none in economics.

## Social

- **Shared modules kept in their subfield**, since social has no `common` package: `cooperation._contributor`, the
  player of the public goods game that `public_goods_game` and `hypergraph_public_goods` use in the same way, and
  whose docstring already names both, and `opinion_dynamics._random_pairs`, the base grouping that the encounters of
  `deffuant` and the naming games of `naming_game` subclass, with its own test module. A copy of either would be two
  identical classes, and for the base grouping two base classes.
- **Copies:** none.
- **Cross-field duplicates:** `_random_cells` of `social.opinion_dynamics._voters` has the same source as the
  `_random_cells` helpers of `computer_science.automata._automata` and `computer_science.automata._variants`, a random
  binary grid with a given density. It is a candidate for the core.

## Mathematics

- **Moved to `mathematics.common`:** `NetworkNodeElement`, which `network_structure_analysis` and the ensembles of
  networks use in the same way, `RegulatedNodeElement` of `triadic_percolation`, whose activity visual the
  dependent nodes of `interdependent_network_cascade` reuse, `VertexElement` with its visual, which the growing networks, the ensembles,
  and the hypergraphs share, and `LinkElement`, which the same systems and `motter_lai_overload` share. Each keeps its
  own module directly in `common`. `RosslerElement`, which `rossler` uses and `DrivenRosslerElement` of
  `ou_driven_rossler` subclasses, moved beside `LorenzElement` in `common.flows`, the existing package of shared flows.
  Networks and dynamical systems still export these names.
- **Kept in `common`:** `common.flows` with `LorenzElement`, which `lorenz` and `fractional_lorenz` use.
- **Kept as a shared helper in a system module:** `_get_bookkeeping` of `networks._growth`, the ageing and degree maps
  that the growing simplicial complex of `networks._hypergraphs` reuses, as before.
- **Copies:** none.
- **Cross-field imports:** `physics.synchronization._chaos` imports `LorenzElement` from `mathematics.common.flows`.
  It is left for the folding of physics, which copies it into physics.
- **Cross-field duplicates:** none in mathematics.

## Engineering

- **Moved to `engineering.common`:** `CarElement` and `OptimalVelocityInteraction`, which `optimal_velocity` of
  `traffic._car_following` and `langevin_traffic` of `traffic._langevin` use in the same way, the second also reading
  the headway helper of the interaction. Each keeps its own module directly in `common`, which now exports both.
- **Kept in `common`:** the `electrical` and `mechanics` packages, since several subfields use them. The capacitor,
  Chua diode, current source, diode, and operational amplifier of `electrical._components` serve no system, but they
  share one module with components that several systems use, so they stay with their tests.
- **Copies:** none.
- **Cross-field duplicates:** none in engineering.

## Computer science

- **Shared modules kept in their subfield**, since computer science has no `common` package: in `automata`,
  `_binary_cell`, the cell of most automata, `_codes`, the `Wolfram` and `life` rule codes, `_groups`, the Cayley balls
  of the group automata, and `_pascal_cell`, which `unbounded_alphabet_automaton` of `_variants` and
  `subtraction_automaton` of `_alphabets` use in the same way. Each is a public name of the package, so a copy would be
  two identical public classes or functions under one name.
- **Copies:** none.
- **Cross-field duplicates:** `_random_cells` of `computer_science.automata._automata` and `._variants` has the same
  source as the one of `social.opinion_dynamics._voters`, a random binary grid with a given density, and
  `_group_automata` and `_line_automata` hold a one-dimensional variant of it. It is a candidate for the core.

## Chemistry

- **Kept in `common`:** the `kinetics` package, whose `BrusselatorReactionElement` and `CubicAutocatalysisInteraction`
  serve both `kinetics` and `pattern_formation`.
- **Copies:** none.
- **Cross-field duplicates:** `_noise` of `chemistry.pattern_formation._reaction_diffusion` has the same source as the
  one of `physics.pattern_formation._patterns`, small uniform noise of a given shape. It is a candidate for the core.

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

## Physics

- **Folded.** Every component module of a physics subfield that serves one system is folded into that system's module,
  and the test modules of the folded components are merged into the system's test module.
- **Moved to `physics.common`.** Components that several system modules of one subfield share, used the same way by
  each: `ActiveParticleElement` with its `POINTS` visual, `SelfPropulsionInteraction`, `PropulsionStepInteraction`,
  `BrownianParticleElement`, `ThermalNoiseInteraction`, and `DoubleWellElement` to `common.particles`,
  `PendulumElement` and `PendulumInteraction` to `common.mechanics`, `IsingSpinElement` to `common.spins`,
  `LinkElement` and the order parameter of the synchronization systems, now the public `order_parameter`, to
  `common.oscillators`, `WilsonSamplingInteraction` to `common.fields`, and `SandpileCellElement` to a module directly
  in `common`, since no family fits it. The subfield packages still export these names.
- **Moved out of `physics.common`.** Modules that one system uses: `LandauEnergyInteraction` and `PhaseFieldElement`
  into `pattern_formation._ordering`, `SwiftHohenbergEnergyInteraction` into `pattern_formation._patterns`,
  `corotational` and `upper_convected` into `active_matter._nematics`, `GravityInteraction` into
  `celestial_mechanics._orbits`, `SoftSphereInteraction` into `active_matter._collective`, and
  `PottsSamplingInteraction` and `XYSamplingInteraction` into `statistical_mechanics._spins`.
- **Kept in `physics.common` although one system uses them.** `ConstantForceInteraction` and
  `LennardJonesInteraction`, since the LAMMPS tables of `common.particles` import them, so the layout test counts them
  as shared.
- **Quantum species.** `SiteElement` and `site_space` are folded into `waves._quantum`, and `SpinElement` and
  `spin_space` into `statistical_mechanics._quantum_spins`. `ModeElement`, `mode_space`, and `QubitElement`, which no
  system uses, stay in `common.quantum`. The core tests that need a spin define their own test spin in the shared
  conftest instead of importing a system module.
- **Kept as modules of their subfield.** Analyses that no system uses: `EarlyWarningAnalysis`,
  `CracklingNoiseAnalysis`, `KineticTemperatureAnalysis`, `MaxwellBoltzmannAnalysis`, and `SusceptibilityAnalysis`.
- **Copied instead of shared.** `LorenzElement` of `mathematics.common.flows`, which
  `synchronization._chaos` imported across fields, is copied into that module.
- **Cross-field duplicates, candidates for the core.** `_noise` of `physics.pattern_formation._patterns` and
  `chemistry.pattern_formation._reaction_diffusion`, uniform noise of a shape, has identical source. `LorenzElement` of
  `physics.synchronization._chaos` and `mathematics.common.flows` has identical code and differs only in its docstring.
