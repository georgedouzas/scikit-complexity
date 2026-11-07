## Context

The subfield packages of `skcomplex.systems` split each system into a module that registers it and several one-class
modules for its elements, interactions, and views. A census on 2026-11-07 found 390 modules that register systems and
970 component modules: 784 serve one system, 39 serve several systems of one subfield, and 147 serve other components
or nothing. Six fields have a `common` package with 59 modules.

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
  `LennardJonesInteraction`, since the LAMMPS tables of `common.particles` and core tests read them. The quantum family,
  `SiteElement`, `SpinElement`, `site_space`, and `spin_space`, since it is one set of Hilbert-space species with
  `ModeElement` and `QubitElement`, which no system uses but core tests do.
- **Kept as modules of their subfield.** Analyses that no system uses: `EarlyWarningAnalysis`,
  `CracklingNoiseAnalysis`, `KineticTemperatureAnalysis`, `MaxwellBoltzmannAnalysis`, and `SusceptibilityAnalysis`.
- **Copied instead of shared.** `LorenzElement` of `mathematics.common.flows`, which
  `synchronization._chaos` imported across fields, is copied into that module.
- **Cross-field duplicates, candidates for the core.** `_noise` of `physics.pattern_formation._patterns` and
  `chemistry.pattern_formation._reaction_diffusion`, uniform noise of a shape, has identical source. `LorenzElement` of
  `physics.synchronization._chaos` and `mathematics.common.flows` has identical code and differs only in its docstring.
