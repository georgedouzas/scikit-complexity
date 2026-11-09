## ADDED Requirements

### Requirement: The core names mathematics and holds no realization
Every public name and module of `skcomplex.core.components` SHALL be a mathematical, modelling, or engineering term,
and no core name or argument SHALL name a quantity of a field. The core SHALL hold each primitive in its general form,
one class per mathematical operation. Every specific realization and every recipe SHALL live in `skcomplex.systems`,
and the core SHALL import nothing from `skcomplex.systems`.

#### Scenario: Forbidden field terms
- **WHEN** the test suite checks the public names and modules of `skcomplex.core.components` against the reviewed list
  of field terms
- **THEN** none of them contains a field term

#### Scenario: No import of systems
- **WHEN** the import graph of `skcomplex.core.components` is computed
- **THEN** it contains no module of `skcomplex.systems`

#### Scenario: Recipes outside the core
- **WHEN** a user builds a `ChoiceInteraction` with the `'fermi'` rule and a `MatchingInteraction` with
  `StableMatchingSpec`
- **THEN** both come from `skcomplex.systems.general`, built on the core `RuleLaw`, `UpdateLaw`, and `MatchLaw`, and no
  field defines either again

### Requirement: The core holds building blocks only
Every component of the core SHALL be a building block: a structural base where laws run, a primitive mechanism that
the engine lowers in its own way, or an operator generated from one function. A recipe, a pattern of strategies that
the primitives express, SHALL live in `skcomplex.systems`. A compile pass, a kernel, a solver, or a backend lowering
SHALL key on a core primitive, never on a class of `skcomplex.systems`.

#### Scenario: A recipe written with primitives
- **WHEN** a reviewer lists the interactions of `skcomplex.core.components.interactions`
- **THEN** none is an agent rule that only returns a `RuleLaw`, and none is a strategy over a law the engine already
  lowers, such as a matching over `MatchLaw` or a reaction over `JumpLaw`

#### Scenario: Engine support stays with its primitive
- **WHEN** the engine compiles a matching, a toppling sandpile, or a binary encounter from the systems
- **THEN** it lowers the core `MatchLaw`, `CascadeLaw`, or `BilinearLaw` that the recipe returns, and names no class of
  `skcomplex.systems`

#### Scenario: Strategy branches leave kept classes
- **WHEN** a system needs the zones of a flock or the row scores of a sequence walk
- **THEN** it subclasses the core `CouplingInteraction` or `SamplingInteraction` in its field, and the core classes take
  no argument for either strategy

## REMOVED Requirements

### Requirement: The core is application agnostic
**Reason**: Its scenario placed choices and matchings in the core, and both are recipes. The rule is restated as "The
core names mathematics and holds no realization", which also keeps recipes out of the core.
**Migration**: Import `ChoiceInteraction` from `skcomplex.systems.general.agents` and `MatchingInteraction` with its
rule specs from `skcomplex.systems.general.matching`.
