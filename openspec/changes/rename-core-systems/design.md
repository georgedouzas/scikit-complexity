## Context

See proposal.md for why. `skcomplex.core.components.systems` holds `_base.py` (`BaseSystem`, `RunSettings`),
`_system.py` (`System`), and `_mass_action.py` (`MassActionSystem`). `System` subclasses `BaseSystem['System']`, and
`MassActionSystem` subclasses `BaseSystem[System]`. Several hundred modules of `skcomplex.systems` import `System`, as
do many tests, the doctests of the core, the user guide pages, and the gallery. `ReactionInteraction` in
`interactions/_reaction.py` computes its default propensity in `_get_mass_action`, which three epidemiology
interactions and one test interaction call.

The word `System` also appears inside other names, such as `BaseSystem`, `SIRSystem`, `MesaSystem`, and
`ReactionNetworkSystem` itself, in generic prose ("System page", "System observables"), in table headers of the
generators, and in the `endswith('System')` checks of the catalog and the source tests. A blind text replacement
would break all of these.

The open changes `slim-core-elements-and-spaces`, `slim-core-interactions`, and `slim-core-laws` edit many of the same
system modules, the user guide, `_mass_action.py`, and the interactions.

## Goals / Non-Goals

**Goals:**

- One name per core system class, after what it is, with no alias.
- Every import, call, `isinstance` check, type hint, error message, and doctest that names the core classes uses the
  new names.
- The core describes the default rate law of reactions in mathematical words.

**Non-Goals:**

- No change of behaviour, arguments, or tags of either class.
- No rename of `BaseSystem`, `RunSettings`, or any ready-made system.
- System modules of chemistry and biology keep "mass action" where it is the term of their field, such as the
  generalized mass action of biochemical systems theory.
- No change to the catalog checks that select classes by the suffix `System`. Both new names keep the suffix, and the
  catalog lists only `skcomplex.systems`.

## Decisions

### Land after the three slim changes, alone

The rename lands after `slim-core-elements-and-spaces`, `slim-core-interactions`, and `slim-core-laws` are applied
and archived, and no other change runs while it is applied.

- The rename touches the import line of nearly every system module. Any change applied in parallel in a worktree would
  conflict with it on those lines. Run alone, it conflicts with nothing.
- The slim changes write their tasks, designs, and code examples with `System` and `MassActionSystem`. If the rename
  landed first, their agents would follow stale artifacts and import names that no longer exist. Landing last, one
  sweep renames everything, including the code those changes add.
- Their delta specs touch other requirements than this change, except `General elements` of `general-mechanisms`,
  which `slim-core-elements-and-spaces` rewrites to name `MassActionSystem`. This change modifies that requirement
  from the text the elements change gives it. Landing after it, this change's delta builds on the archived text.
- `slim-core-elements-and-spaces` rewrites `_mass_action.py` to use `NonnegativeElement` and the state `x`. Renaming
  the module after it avoids a rename and an edit of one file in two changes at once.

Alternative: land between the elements change and the two other slim changes. This saves nothing, since the other
two would still need their artifacts updated for the new names, and it splits the sweep in two.

### Rewrite on exact names, never on the bare word

A throwaway script outside the repository does the mechanical part. It is not committed.

- It selects only files that import `System` or `MassActionSystem` from the systems package, by the absolute path, a
  relative path that ends in `components.systems`, or the private modules, in code or in a doctest line.
- In those files it renames the imported name, and the bare identifier `System` where no word character or dot is next
  to it. So `BaseSystem`, `SIRSystem`, and `system.System` stay.
- `MassActionSystem` is a unique token and is renamed everywhere.
- Markdown pages and gallery examples are selected the same way, by their import lines.
- After the script, a grep for the bare word `System` lists every remaining hit, and each is reviewed by hand: prose
  stays, table headers stay, the `endswith('System')` checks stay, and any reference to the core class is renamed.

Alternative: rename by an AST tool such as `rope`. It misses doctests and Markdown, which hold many uses, so a
reviewed text rewrite is simpler.

### Rename modules and tests with git

`git mv` renames `_system.py` to `_composite.py`, `_mass_action.py` to `_reaction_network.py`, and the mirrored tests
`test_system.py` to `test_composite.py` and `test_mass_action.py` to `test_reaction_network.py`, so history follows the
files. The layers contract on `skcomplex.core.components.systems` lists `_reaction_network`, `_composite`, and `_base`.

### Name the rate law by its mathematics

The default propensity of a reaction is `k` times the product of the reactant and modifier amounts, each raised to its
stoichiometric coefficient. The private method that computes it becomes `_get_product_propensity`, after the product
it forms. The docstrings of `ReactionInteraction` and `ReactionNetworkSystem` state the formula and give the field
terms only as examples: the law of mass action in chemistry, homogeneous mixing in epidemiology, and the encounter term
of Lotka-Volterra in ecology. "Reaction network" stays, as the mathematical name of the formalism.

Alternative: `_get_power_law_propensity`. A power law suggests one variable, while the method forms a product over
several, so "product" is the plainer word.

### Rewrite source text that the translators emit

The module writer of the system-dynamics translator imports `System` by name and emits `System(` in the generated
`_build_system`. Both become `CompositeSystem`. The translator tests that compare the generated source change with it.

## Risks / Trade-offs

- [The bare word `System` is renamed in prose or in an unrelated name] → The script touches only files that import the
  core class, it skips names joined to a word character or a dot, and every remaining hit is reviewed by hand.
- [A use is missed in a file that names the class without importing it, such as a type in a string or an error
  message] → A final grep over src, tests, docs, the README, and CONTRIBUTING for the bare word `System` and for
  `MassActionSystem`, `_get_mass_action`, `_system`, and `_mass_action` must show only reviewed hits.
- [Tests match error messages that name `System`] → The test run catches them, and they are updated with the message.
- [The slim changes are not archived when this change starts] → The first task checks that they are archived and
  rebuilds the `General elements` delta from the archived main spec.
- [Cached files such as `__pycache__` or built docs keep old names] → They are not tracked, and the docs build
  regenerates them.

## Migration Plan

Users replace `System` with `CompositeSystem` and `MassActionSystem` with `ReactionNetworkSystem` in their imports and
calls. Subclasses of `ReactionInteraction` call `_get_product_propensity`. The changelog of the next release lists both
renames. Rollback is a revert of the change's commits.
