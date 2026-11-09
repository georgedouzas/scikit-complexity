## 1. Preconditions

- [ ] 1.1 Confirm `slim-core-elements-and-spaces`, `rename-core-systems`, `slim-core-interactions`,
  `slim-core-laws`, and `move-recipes-to-systems` are merged on `development`. Verify with `pdm run python -c`
  importing `CompositeSystem` from `skcomplex.core.components.systems`, `ReactionNetworkSystem` from
  `skcomplex.systems.general.reactions`, `NonnegativeElement` from `skcomplex.systems.general.elements`,
  `DeclaredElement` from `skcomplex.core.components.elements`, and `BallNeighborhood` from
  `skcomplex.core.components.spaces`.
- [ ] 1.2 Create `src/skcomplex/systems/general/dynamics/__init__.py` with a module docstring and an empty `__all__`,
  and the mirrored test directory `tests/systems/general/dynamics/`. The placement rule puts the four systems
  here by step 3, since several fields use them. Verify with `pdm run python -c` importing the subpackage and with
  `lint-imports`, which must show that no core module imports it.
- [ ] 1.3 Write a scratchpad script, outside the repository, that records the trajectory of a list of catalog systems
  with their defaults, `random_state=0` where they take one, and their tuned run settings, and compares two records with
  `numpy.testing.assert_array_equal` on every name. Verify it reports equal on two records of `BathtubSystem` made
  without any code change.

## 2. DynamicalSystem

- [ ] 2.1 Add `DynamicalSystem` in `src/skcomplex/systems/general/dynamics/_dynamical.py` with the signature and hooks
  of design.md, and export it from `systems/general/dynamics/__init__.py`, without a layers contract. Its docstring
  carries runnable examples of two fields: the Wilson-Cowan populations in continuous time and the
  multiplier-accelerator of economics in discrete time. Verify with the doctests of the module.
- [ ] 2.2 Add `tests/systems/general/dynamics/test_dynamical.py` with the scenarios of `general-systems` for
  `DynamicalSystem`: exponential decay, unknown state, the logistic map, the Ornstein-Uhlenbeck tags, noise in
  discrete time, the Hénon map in both orders of states, a subclass with its own parameters, and an equality test of a
  trajectory against the same equations written as element laws. Verify they pass. If the Hénon test fails, build one
  `MapInteraction` on a vector state as design.md says, and verify again.

## 3. Rebases onto DynamicalSystem in continuous time

For each module below: check that the system still matches the shape and that its element class serves it alone,
record the trajectories with the script of 1.3, rebase the system, move the laws of its element into its hooks, remove
the element class and its export, and compare the trajectories. A system that differs keeps its base, and the task
notes why. Verify each with the comparison and the mirrored tests and doctests of the module.

- [ ] 3.1 biology: `ecology/_cell_quota.py` (`QuotaPhytoplanktonElement`), `ecology/_factories.py`
  `PenaeidShrimpSystem` (`ShrimpElement`), `epidemiology/_concurrency.py` (`PartnershipEpidemicElement`),
  `epidemiology/_vaccination.py` (`VaccinatedPopulationElement`), `neuroscience/_populations.py`
  (`PolyhomeostaticNeuronElement`, `WilsonCowanElement`), `oncology/_antiangiogenic.py`,
  `oncology/_chemotherapy.py`, `oncology/_combination_therapy.py`, `oncology/_hormone_therapy.py`,
  `oncology/_optimal_therapy.py`, `physiology/_regulation.py`, `systems_biology/_networks.py`
  (`GlycolysisElement`, whose declared variables go to `_get_variables`)
- [ ] 3.2 chemistry: `kinetics/_reactors.py` `PulsedElectrochemicalReactorSystem` (`PulsedElectrodeElement`)
- [ ] 3.3 economics: `macroeconomics/_capital.py`, `macroeconomics/_optimal_growth.py`,
  `macroeconomics/_purchasing_power.py`, `markets/_prices.py`, `resources/_optimal_harvesting.py`
- [ ] 3.4 engineering: `circuits/_circuits.py` (four systems), `circuits/_memristive_mixed_mode_oscillator.py`,
  `control/_control.py` (four systems), `mechatronics/_vibrating_table.py`, `power_systems/_electromechanics.py`
- [ ] 3.5 mathematics: `dynamical_systems/_driven_flows.py`, `dynamical_systems/_flows.py` (eight systems, not the
  Lorenz and Rössler systems), `dynamical_systems/_koopman.py`, `dynamical_systems/_normal_forms.py`,
  `dynamical_systems/_subcritical_forms.py`, `fractional_dynamics/_fractional.py` (fractional orders go to
  `_get_variables`)
- [ ] 3.6 physics: `celestial_mechanics/_spin_orbit.py`, `climate/_energy_balance.py` (Daisyworld and the energy
  balance), `climate/_ice_ages.py`, `climate/_low_order.py`, `climate/_pollution.py`, `climate/_tipping_types.py`,
  `climate/_variability.py`, `heat/_cooling.py`, `mechanics/_accelerators.py` `FrictionOscillatorSystem`,
  `mechanics/_rotations.py`, `nonlinear_oscillators/_oscillators.py` (Adler, Duffing, Toda),
  `oceanography/_thermohaline.py`, `statistical_mechanics/_master_equations.py`

## 4. Rebases onto DynamicalSystem in discrete time

Same procedure as group 3, with `time='discrete'`.

- [ ] 4.1 biology: `ecology/_factories.py` `FibonacciRabbitsSystem`, `ecology/_instability.py`,
  `epidemiology/_recurrent.py`, `evolution/_prebiotic.py` `EigenQuasispeciesSystem`, `neuroscience/_synapses.py`
- [ ] 4.2 computer science: `neural_networks/_next_generation.py`
- [ ] 4.3 economics: `finance/_switching_exchange.py`, `innovation/_diversity.py`, `macroeconomics/_growth.py`,
  `resources/_crop_livestock.py`
- [ ] 4.4 engineering: `control/_koopman.py`
- [ ] 4.5 mathematics: `dynamical_systems/_maps.py` (thirteen maps), `fractals/_quadratic.py`
- [ ] 4.6 physics: `active_matter/_flocks.py` `GregoireChateSystem`, `climate/_soil.py`, `criticality/_sandpiles.py`
  `SelfOrganizedBranchingSystem`, `mechanics/_accelerators.py` `BeamBeamMapSystem`, `statistical_mechanics/_urns.py`

## 5. NetworkDynamicalSystem

- [ ] 5.1 Add `NetworkDynamicalSystem(DynamicalSystem)` in
  `src/skcomplex/systems/general/dynamics/_network_dynamical.py` with the signature and hooks of design.md, and export
  it from the subpackage, without a layers contract. Its docstring carries runnable examples of diffusion on a graph and
  of gap-junction exchange between cells. Verify with the doctests of the module.
- [ ] 5.2 Add `tests/systems/general/dynamics/test_network_dynamical.py` with the scenarios of `general-systems`:
  diffusion on a path, the missing graph, initial values per node and broadcast, and the tags. Verify they pass.

## 6. Rebases onto NetworkDynamicalSystem

Same procedure as group 3.

- [ ] 6.1 biology: `physiology/_exchange.py`, `epidemiology/_networks.py` `MetapopulationSystem`,
  `epidemiology/_adaptive.py` `QuenchedMeanFieldSISSystem`
- [ ] 6.2 physics: `climate/_tipping.py` `TippingCascadeNetworkSystem`, and `synchronization/_networks.py`
  `KuramotoOnNetworksSystem` if its edge group holds no state its trajectory reports. `PhaseOscillatorElement` stays,
  as other systems compose it.
- [ ] 6.3 mathematics: `networks/_processes.py` `NetworkDiffusionSystem`. `PagerankSystem` keeps its base.

## 7. ReactionDiffusionSystem

- [ ] 7.1 Add `ReactionDiffusionSystem(DynamicalSystem)` in
  `src/skcomplex/systems/general/dynamics/_reaction_diffusion.py` with the signature and hooks of design.md, and export
  it from the subpackage, without a layers contract. Its docstring carries runnable examples of the
  Gray-Scott model and of the Fisher-Kolmogorov front. Verify with the doctests of the module.
- [ ] 7.2 Add `tests/systems/general/dynamics/test_reaction_diffusion.py` with the scenarios of `general-systems`: the
  front speed, a field with no diffusivity, the boundaries per field, a spectral discretization, noise, and an
  override of `_get_equation_interactions`. Verify they pass.

## 8. Rebases onto ReactionDiffusionSystem

Same procedure as group 3.

- [ ] 8.1 biology: `ecology/_fronts.py`, `physiology/_cardiac.py` (Aliev-Panfilov, Barkley), `physiology/_physiology.py`
  `FitzhughNagumoSystem`, `systems_biology/_min_proteins.py`
- [ ] 8.2 chemistry: `pattern_formation/_reaction_diffusion.py` (Gray-Scott, Brusselator, Schnakenberg, keeping their
  reactions through `_get_equation_interactions`)
- [ ] 8.3 physics: `heat/_heat.py` `HeatDiffusionSystem`, `criticality/_absorbing.py`
  `DirectedPercolationLangevinSystem`

## 9. StockFlowSystem

- [ ] 9.1 Add `StockFlowSystem` in `src/skcomplex/systems/general/dynamics/_stock_flow.py` with the signature and hooks
  of design.md, and export it from the subpackage, without a layers contract. Its docstring carries runnable examples of
  the bathtub and of an arms race. Verify with the doctests of the module.
- [ ] 9.2 Add `tests/systems/general/dynamics/test_stock_flow.py` with the scenarios of `general-systems`: the bathtub,
  the flow of an unknown stock, declared domains, conservation of the total over flows between stocks, and the flow
  names in the trajectory. Verify they pass.

## 10. Rebases onto StockFlowSystem

Same procedure as group 3.

- [ ] 10.1 economics: `innovation/_bass.py`, `markets/_commodities.py`, `markets/_inventory.py`,
  `resources/_extraction.py` (oil economy, fishery capital), `resources/_fisheries.py`, `resources/_food_balance.py`
- [ ] 10.2 engineering: `processes/_grain_storage.py`, `processes/_processes.py` `BathtubSystem`,
  `processes/_systems.py` `FoodReliefSystem`
- [ ] 10.3 social: `opinion_dynamics/_undecided.py`, `organizations/_traps.py` (five systems)
- [ ] 10.4 For `CascadeClassifierSystem`, `RiceMillingSystem`, `SolidWasteSystem`, `IrrigationSystem`, `CocoaSystem`,
  and `EnergyReturnSystem`, try the rates outside their flows as named flows and compare with the script of 1.3. Keep
  each system on its base unless names and values are equal, and record the result in this task.

## 11. Translator

- [ ] 11.1 Make the module writer of `core/engine/translators` write a subclass of `StockFlowSystem` for a file whose
  stocks are not subscripted and change only through flow variables, with the constants as parameters and the hooks
  `_get_stocks` and `_get_flows`, and with the source text importing `StockFlowSystem` from
  `skcomplex.systems.general.dynamics`. The translator itself never imports the subpackage. Other files keep the element
  shape. Verify with `tests/core/engine/translators/test_system_dynamics.py`, extended with the two scenarios of
  `system-dynamics-import`, and the PySD comparisons within 1e-6.
- [ ] 11.2 Update the docstrings of `_system_dynamics.py` and `_module_writer.py` and the stocks and flows page of the
  user guide to show a translated `StockFlowSystem`. Verify with the doctests and `pytest-markdown-docs`.

## 12. Documents

- [ ] 12.1 Add a section per general system to the page of `skcomplex.systems.general` in the systems guide, not to
  `core_components/systems.md`, named after the class, with one plain sentence, examples from two fields, and the link
  to the generated table of the parts page. Create the page in the layout of `rewrite-docstrings-and-docs` if it does
  not exist yet. Verify with
  `pytest-markdown-docs` that every block runs.
- [ ] 12.2 Update `docs/overview/user_guide/system_classes.md` to name the general systems, and replace in the user
  guide and the gallery every use of a removed element class. Verify with `pytest-markdown-docs` and a search for the
  removed names that finds none.
- [ ] 12.3 Add a changelog entry that names the four general systems and the translated `StockFlowSystem` subclasses,
  and lists the removed element classes. Verify the entry renders in `nox -s docs -- build`.

## 13. Inventory

- [ ] 13.1 Update the entries of `docs/overview/inventory.toml` whose components or bindings changed with a rebase.
  Verify with the inventory check that the documentation build runs.

## 14. Integration

- [ ] 14.1 Run `nox -s checks` and `nox -s tests`, with `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, and the BLAS threads set
  to 1 when other agents run tests. Verify both pass.
- [ ] 14.2 Run `nox -s docs -- build` and verify it passes with the generated parts page of systems listing the four
  general systems and their catalog users.
- [ ] 14.3 Run `openspec validate add-general-systems --strict` and verify it passes.
