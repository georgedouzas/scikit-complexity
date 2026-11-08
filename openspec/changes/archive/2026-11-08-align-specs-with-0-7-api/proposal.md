# Proposal

## Why

The archived change `align-api-with-scikit-learn` moved the library to the 0.7 API, with ready-made systems as classes
found by `all_systems`, analyses as functions that return frozen results, `simulate` and `sweep` as functions, and
private compilers for the library's own forms. Several main-spec requirements still describe the registered factories,
the catalog functions, `Simulation`, fitted analysis attributes, design estimators, and surrogate estimators that the
change removed, so the specs contradict the code.

## What Changes

- model-library: rename "Ready-made systems are registered factories" to "Ready-made systems are classes" and describe
  system classes on `BaseSystem` with model parameters in the constructor and class attributes for field, topics,
  references, and tuned run settings, found by `all_systems()`.
- model-catalog: remove "Listing systems imports no system", superseded by "Systems are discovered with all_systems".
- simulation: "A simulation holds a default end time" describes `simulate` with the tuned `until`, and "A system package
  plugs in its own compiler" describes the `compiler` run setting on the public `BaseCompiler` and the automatic choice
  of the private compilers of the library's forms.
- user-documentation: README and guide import ready-made systems as classes from their subfield package and nothing else
  from field packages.
- package-layout: the public names of `skcomplex.systems` are `all_systems` and `import_extra`, and system classes end in
  `System`.
- analysis-interface: remove "Results are fitted attributes", superseded by "Analyses are functions of results".
- experiments: a design is the `design=` argument of `sweep`.
- base-interfaces: drop the bullet on designs keeping rows in `rows_`.
- data-driven-dynamics: the surrogates are the functions `mode_decomposition`, `sparse_identification`, and `reservoir`,
  whose frozen results hold the fitted `system` and the `forecast`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-library`: ready-made systems are classes.
- `model-catalog`: the listing requirement of the removed catalog function goes.
- `simulation`: the default end time and the compiler hook follow `simulate` and the run settings.
- `user-documentation`: the documentation imports system classes from subfield packages.
- `package-layout`: the public names of `skcomplex.systems`.
- `analysis-interface`: the fitted-attribute requirement goes.
- `experiments`: designs are an argument of `sweep`.
- `base-interfaces`: designs no longer keep `rows_`.
- `data-driven-dynamics`: surrogates are analysis functions.

## Impact

Specs only. No code changes, since the specs follow the 0.7 code.
