## Why

The elements and spaces of the core hold classes that are only parametrised versions of another core class.
`DensityElement` and `CompartmentElement` are the same nonnegative scalar under two names and two state names, `f` and
`n`. `LatticeElement` is a `GroupElement` on sites. `MooreNeighborhood`, `VonNeumannNeighborhood`, `BoxNeighborhood`, and
`TriangularNeighborhood` are one ball of lattice offsets under four metrics and orders. `GeneratorNeighborhood` is a
`MemoryNeighborhood` with a fixed word list. The core should name each math operation once, and the fields should name
their own quantities. The spec `_SubsystemSpec` is also private but imported by two engine packages.

## What Changes

- **BREAKING** New core element `NonnegativeElement(label, visual=None, *, x=0.0)`, a scalar state `x` in `[0, oo)`.
  `DensityElement` and `CompartmentElement` leave the core, and their states `f` and `n` become `x`.
- **BREAKING** `MassActionSystem` builds `NonnegativeElement` species and keys its reactions by `(species, 'x')`.
  `EncounterInteraction` reads the state `x` by default.
- Field realizations subclass `NonnegativeElement` where the field has its own term.
  - biology: `CompartmentElement` in `biology/common/kinetics`, `PopulationDensityElement` in
    `biology/common/populations`. `PopulationElement` and the biology `SpeciesElement` rebase onto the atom.
  - physics: `PhaseSpaceDensityElement` in `physics/common/fields`.
  - chemistry: `SpeciesElement` in `chemistry/kinetics/_nanoparticles.py`, its one user.
  - economics: `WealthDensityElement` in `economics/econophysics/_welfare.py`, its one user.
  - social: `ActivityDensityElement` in a new `social/common/kinetics` family, shared by opinion dynamics,
    cooperation, and crowds.
  - Every subclass of the atom uses the state `x`, so `'A.n'` and `{'f': ...}` become `'A.x'` and `{'x': ...}`.
- **BREAKING** `LatticeElement` leaves the core. `GroupElement` on `'sites'` flattens values of the lattice shape and
  node mappings, and takes the default visual of its space. `FieldElement` subclasses `GroupElement`. Systems call
  `GroupElement(..., on='sites')` directly, with no field copy.
- **BREAKING** New core `BallNeighborhood(radius=1, *, metric='chebyshev', n_dim=2, center=None)`, the lattice offsets
  within a radius under the Chebyshev, Manhattan, or hexagonal metric. `center` is `None`, `'first'`, or `'inline'`,
  where `'inline'` keeps the site in its lexicographic place, the Wolfram order. `MooreNeighborhood`,
  `VonNeumannNeighborhood`, `BoxNeighborhood`, and `TriangularNeighborhood` leave the core with no field copy.
- **BREAKING** `LatticeSpace` takes `metric` in place of `neighborhood='moore' | 'von_neumann'`, so a metric is named
  one way. A triangular lattice defaults to the hexagonal metric.
- **BREAKING** `GeneratorNeighborhood` leaves the core for `computer_science/automata`, its only users, name kept.
- **BREAKING** `_SubsystemSpec` becomes the public `SubsystemSpec` of `skcomplex.core.components.specs`.
- Nothing meaningful is deleted. The lattice flattening moves into `GroupElement`, the offsets of every neighbourhood
  move into `BallNeighborhood`, and the word list of generators moves to the automata package.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `general-mechanisms`: the general elements become `NonnegativeElement` and `PointElement`, and kinetic systems use
  field realizations of the nonnegative atom.
- `phase-space-fields`: a phase-space field holds members of a nonnegative scalar element.
- `lattice-dynamics`: lattice neighbourhoods are balls of offsets under a named metric.
- `package-layout`: the elements and neighbourhoods of the core hold no parametrised copy of another core class, and
  compile records used across engine packages are public.

## Impact

- Core code: `core/components/elements` (new `_nonnegative.py`, removed `_density.py`, `_compartment.py`,
  `_lattice.py`), `core/components/spaces` (new `_ball.py`, removed `_moore.py`, `_von_neumann.py`, `_box.py`,
  `_triangular.py`, `_generator.py`), `core/components/specs/_subsystem_spec.py`,
  `core/components/systems/_mass_action.py`, defaults in `interactions/_sampling.py`, `interactions/_cascade.py`, and
  `interactions/_encounter.py`, docstrings of interactions, laws, solvers, analysis, and viz.
- Engine: `engine/problems/_operator_lowering.py` and `engine/compile/passes/_operator_compiler.py` import the public
  `SubsystemSpec`.
- Systems: every module that names a removed class or the states `n` and `f` of the atom, in biology, chemistry,
  computer science, economics, mathematics, physics, social, and engineering. New package
  `skcomplex.systems.social.common`.
- Tests, user guide, gallery examples, `docs/generate_parts.py`, and the import contracts in `pyproject.toml`.
- Users: code that imports a removed class, passes `n=` or `f=` to a compartment or density, reads `'A.n'`, or passes
  `neighborhood=` to `LatticeSpace` must change. The project makes breaking changes in 0.x without shims.
- Depends on `add-system-package`, which moves `MassActionSystem` to `core/components/systems/_mass_action.py`.
