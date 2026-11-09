## 1. Core atom

- [x] 1.1 Add `src/skcomplex/core/components/elements/_nonnegative.py` with `NonnegativeElement(label, visual=None, *, x=0.0)`, state `x` in `[0, oo)`, export it from `elements/__init__.py`, and add `tests/core/components/elements/test_nonnegative.py` that checks the state, its domain, and a constant run
- [x] 1.2 Replace `CompartmentElement` and `DensityElement` in the docstrings of `interactions/_reaction.py`, `_rate.py`, `_relaxation.py`, `_gradient.py`, `_jump.py`, `engine/solvers/_gillespie.py`, `_semi_lagrangian.py`, and `_strang_splitting.py` with `NonnegativeElement` and the state `x`, and verify `pytest --doctest-modules src/skcomplex/core` passes

## 2. Core group on sites

- [x] 2.1 Move `_flatten_values` of `LatticeElement` into `GroupElement` for groups on `'sites'`, reading the space the group lives on, own or inherited, and move the default visual of the space into `GroupElement._prepare`, and add tests to `tests/core/components/elements/test_group.py` for lattice-shaped values, node mappings, and the visual
- [x] 2.2 Make `FieldElement` subclass `GroupElement` with `on='sites'`, and verify `tests/core/components/elements/test_field.py` passes
- [x] 2.3 Replace `LatticeElement` in the core docstrings of `analysis/_automata.py`, `interactions/_sampling.py`, and `engine/viz/_sites.py`, and move the tests of `test_lattice.py` that still apply into `test_group.py`

## 3. Core ball neighbourhood

- [x] 3.1 Add `src/skcomplex/core/components/spaces/_ball.py` with `BallNeighborhood(radius=1, *, metric='chebyshev', n_dim=2, center=None, label='neighborhood')`, one private method per metric, and a private check raising `ValidationError`, export it, and add `tests/core/components/spaces/test_ball.py` that checks each former class's offsets for its default and `center` settings, the hexagonal rings for radius 2, and every error
- [x] 3.2 Replace `neighborhood` of `LatticeSpace` with `metric=None`, add its check, and make `_default_neighborhood` and `_ball` return `BallNeighborhood`, keeping the inline Chebyshev ball for `_ball`, and update `tests/core/components/spaces/test_lattice.py`
- [x] 3.3 Replace the defaults in `interactions/_sampling.py` and `interactions/_cascade.py` and every core docstring that names `MooreNeighborhood`, `VonNeumannNeighborhood`, `BoxNeighborhood`, or `TriangularNeighborhood`, or passes `neighborhood=` to `LatticeSpace`, including `specs/_objective_spec.py`, `interactions/_local_rule.py`, `interactions/_movement.py`, `laws/_sample.py`, `laws/_cascade.py`, `laws/_local_rule.py`, `laws/_extremal.py`, and `laws/utils/_reductions.py`, and verify the core tests and doctests pass

## 4. Core spec

- [x] 4.1 Rename `_SubsystemSpec` to `SubsystemSpec`, export it from `core/components/specs/__init__.py`, import it from the package in `engine/problems/_operator_lowering.py` and `engine/compile/passes/_operator_compiler.py`, and verify `grep -rn "_SubsystemSpec" src tests` finds nothing

## 5. Field realizations

The old core classes stay until section 7, so each task below leaves the tree green.

- [ ] 5.1 Add biology `CompartmentElement(NonnegativeElement)` in `systems/biology/common/kinetics/_compartment.py`, export it, rebase `PopulationElement`, `BranchElement`, `RatsElement`, `NeutralSpeciesElement`, and `HabitatPatchElement` on it, rename their state to `x` with every `n=`, `'A.n'`, `('A', 'n')`, and `{'n': ...}` that reads them, and verify `pytest tests/systems/biology` and the biology doctests pass
- [ ] 5.2 Rebase the biology `SpeciesElement` in `common/kinetics/_species.py` on `NonnegativeElement`, keeping its docstring, use `NonnegativeElement` directly for the places of `systems_biology/_logical_networks.py`, and replace `CompartmentElement` in the docstrings of epidemiology, pharmacokinetics, excitable, kinetics, populations, and oncology
- [ ] 5.3 Add chemistry `SpeciesElement(NonnegativeElement)` in `systems/chemistry/kinetics/_nanoparticles.py`, export it from the package, and rename its state reads to `x`
- [ ] 5.4 In one commit, make `MassActionSystem` in `core/components/systems/_mass_action.py` build `NonnegativeElement` species keyed by `(species, 'x')`, and update its users `systems/chemistry/kinetics/_reactions.py`, `systems/biology/epidemiology/_factories.py`, `tests/core/components/systems/test_mass_action.py`, `docs/generate_parts.py`, and the user guide pages `system_classes.md` and `extend_the_library.md`, and verify their tests and doctests pass
- [ ] 5.5 Add `PhaseSpaceDensityElement(NonnegativeElement)` in `systems/physics/common/fields/_phase_space_density.py`, export it, and update the package docstring
- [ ] 5.6 Add `PopulationDensityElement(NonnegativeElement)` in `systems/biology/common/populations/_population_density.py` and export it
- [ ] 5.7 Create `systems/social/common/__init__.py` and the family `systems/social/common/kinetics/` with `ActivityDensityElement(NonnegativeElement)` in `_activity_density.py`, add `skcomplex.systems.social.common` to the forbidden modules of every other field's contract in `pyproject.toml`, and verify `lint-imports` passes with `PYTHONPATH=src`
- [ ] 5.8 Add `WealthDensityElement(NonnegativeElement)` in `systems/economics/econophysics/_welfare.py` and export it from the package
- [ ] 5.9 In one commit, set the default state of `EncounterInteraction` to `'x'`, update its docstring and the message of `engine/compile/passes/_compile_encounters.py` where it names a state, and switch every density user to its realization with `x`: `fluids/_kinetic_equations.py` and `active_matter/_kinetic_flocks.py` to `PhaseSpaceDensityElement`, `oncology/_immune_competition.py` and `ecology/_forest_size_structure.py` to `PopulationDensityElement`, `opinion_dynamics/_kinetic_opinion.py`, `cooperation/_multilevel_selection.py`, `crowds/_criminality.py`, and `crowds/_evacuation.py` to `ActivityDensityElement`, and `econophysics/_welfare.py` to `WealthDensityElement`, with the core tests and docstrings that pass `f`, and verify `pytest tests` passes

## 6. Systems on groups and balls

- [ ] 6.1 Replace `LatticeElement(...)` with `GroupElement(..., on='sites')` in every system module, field by field, and verify `pytest tests/systems` passes after each field
- [ ] 6.2 Replace `MooreNeighborhood`, `VonNeumannNeighborhood`, `BoxNeighborhood`, and `TriangularNeighborhood` with `BallNeighborhood` and the matching `metric`, `center`, and `n_dim` in computer science, chemistry, social, biology, and physics, and replace `neighborhood=` of `LatticeSpace` with `metric=` in every system module
- [ ] 6.3 Move `GeneratorNeighborhood` to `systems/computer_science/automata/_generator_neighborhood.py`, export it from the package, import it in `_free_group_automata.py` and `_group_automata.py`, and move `tests/core/components/spaces/test_generator.py` to `tests/systems/computer_science/automata/test_generator_neighborhood.py`

## 7. Removal

- [ ] 7.1 Delete `elements/_density.py`, `_compartment.py`, `_lattice.py`, and `spaces/_moore.py`, `_von_neumann.py`, `_box.py`, `_triangular.py`, `_generator.py` with their exports and tests, and verify `grep -rnw` for every removed name over `src`, `tests`, `docs`, `README.md`, and `CONTRIBUTING.md` finds only the field realizations that keep a name

## 8. Tests, documents, and gallery

- [ ] 8.1 Add tests for each new realization under the mirrored `tests/systems/<field>/common/...` or system path, and update `tests/systems/biology/common/populations/test_population.py` to check the biology `CompartmentElement`
- [ ] 8.2 Update the user guide pages under `docs/overview/user_guide` that name a removed class or a state `n` or `f` of the atom, `docs/generate_parts.py`, and the gallery examples under `docs/examples`, and verify the markdown doctests pass
- [ ] 8.3 Check `docs/overview/inventory.toml` names no removed class, and verify `python docs/inventory_checks.py` passes

## 9. Verification

- [ ] 9.1 Run the conformance suite and check every library system keeps its trajectory on fixed seeds
- [ ] 9.2 Run `nox -s checks` for `src`, `tests`, and `docs`, including ruff, mypy, the import contracts, the full test session with doctests, and the strict docs build, and fix every finding without a suppression
