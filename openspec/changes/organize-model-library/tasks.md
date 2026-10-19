# Tasks

## 1. Core and models layout

- [x] 1.1 Move today's `core` to `skcomplex.core.base` and `model`, `compile`, `solvers`, `runtime`, `viz`,
  `experiments`, `analysis`, and `backends` under `skcomplex.core`, update every import, and verify the framework import
  scenarios and that the old paths fail
- [x] 1.2 Add the layout check that core modules never import `skcomplex.models`, model modules import only public
  names of the core, and no model package imports another, and verify the core-without-models, planted reverse import,
  private core import, and cross-package import scenarios

## 2. Constitution and dependencies

- [x] 2.1 Amend the constitution with the rule to build on popular, mature, or performant libraries, and update the
  Project Profile with `scikit-fem`, the core and models layout, and the layering inside the core
- [x] 2.2 Audit the framework's finite-difference, spectral, finite-volume, stiff-stepper, and Gillespie numerics
  against `py-pde`, Dedalus, PyClaw, SciPy, `diffrax`, and `gillespy2`, record the comparison, and create a follow-up
  change for every replacement the comparison favours

## 3. Taxonomy and migration

- [x] 3.1 Create the field and subfield packages, move every existing model to its new home following the migration
  table, move the domain-agnostic shared components into the core and give every package its own science-specific
  components, use catalog field paths, and remove the technique packages, and verify the conformance suite, the layout
  check, and the removal scenario pass over the moved library
- [x] 3.2 Update the tests, the user guide, the README, and the gallery to the new paths, and verify the documentation
  test passes

## 4. Inventory

- [x] 4.1 Add the inventory data file with every entry of the design's inventory, and the check that compares it with
  the catalog and the OpenSpec changes, and verify the check's scenarios
- [x] 4.2 Render the inventory as the coverage page and generate the per-field pages with each model's laws and
  equations, and verify the docs build and the pattern scenario

## 5. Model collections

For every task in this section:
- implement each `+` entry of the field in the inventory, as factories on shared components;
- mark each entry implemented in the inventory;
- when a model turns out to need a feature the interface lacks, create or extend the change that specifies it, with a
  survey of the libraries that implement it, and mark the model blocked on that change instead of working around the
  interface.

- [x] 5.1 Physics, and verify the Kepler and active Brownian particle scenarios
- [x] 5.2 Chemistry, and verify the Oregonator scenario
- [x] 5.3 Biology, and verify the repressilator and Hodgkin-Huxley scenarios
- [x] 5.4 Earth science, and verify the ice-albedo scenario
- [x] 5.5 Economics, and verify the kinetic wealth exchange scenario
- [x] 5.6 Social science, and verify the bounded confidence scenario
- [x] 5.7 Engineering, and verify the optimal velocity scenario
- [x] 5.8 Mathematics, and verify the standard map scenario
- [x] 5.9 Computer science, and verify the random Boolean network scenario

## 6. Missing features and pending changes

- [x] 6.1 Make sure every blocked entry names an existing change, creating the expected changes of the design that
  models in section 5 needed, and verify the inventory check passes
- [x] 6.2 Update the proposals, specs, designs, and tasks of the nine pending changes to the core paths and the new
  subfields, record their models in the inventory, and verify the pending scenario of the inventory check

## 7. Documentation and gate

- [ ] 7.1 Document the core and models layout, the taxonomy, the model pattern, and the coverage page in the user guide,
  list the moved paths in the changelog, and verify the documentation test passes
- [ ] 7.2 Run formatting, checks, the docs build, and tests through nox, and the benchmarks with `--benchmarks`, and
  verify all of them pass
