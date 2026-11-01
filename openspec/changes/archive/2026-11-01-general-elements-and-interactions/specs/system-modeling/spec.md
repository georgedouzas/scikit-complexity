# Spec Delta

## MODIFIED Requirements

### Requirement: The core is application agnostic
Every public name and module of `skcomplex.core` SHALL be a mathematical, modelling, or engineering term. The core SHALL
hold each primitive in its general form, one class per mathematical operation, configured by the states it binds and by
functions, whose standard variants are named by their mathematics, such as the logit or Fermi rule of a choice, the
Lévy step of a movement, or the stable matching of a matching rule. No core name or argument SHALL name a quantity
of a field. Every specific realization
SHALL live in `skcomplex.models`, and the core SHALL import nothing from `skcomplex.models`. A specific realization is
a variant named by a field or a person's model rather than by its mathematics, such as a named potential like
Lennard-Jones, an order book, a rule code, a quantum species, a forecasting heuristic, a tiling, or a distribution
beyond the default.

#### Scenario: Forbidden field terms
- **WHEN** the test suite checks the public names and modules of `skcomplex.core` against the reviewed list of field
  terms
- **THEN** none of them contains a field term

#### Scenario: No import of models
- **WHEN** the import graph of `skcomplex.core` is computed
- **THEN** it contains no module of `skcomplex.models`

#### Scenario: Mathematical variants in the core
- **WHEN** a user builds a `ChoiceInteraction` with the `'fermi'` rule and a `MatchingInteraction` with
  `StableMatchingSpec`
- **THEN** both interactions come from `skcomplex.core.interactions`, the rule spec from `skcomplex.specs`, and no model
  defines either variant again
