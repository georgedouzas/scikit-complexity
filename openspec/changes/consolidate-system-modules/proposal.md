## Why

A system module of `skcomplex.systems` should define one system cleanly and read as an example implementation. Today
the subfield packages hold 970 one-class component modules next to 390 modules that register systems, so reading one
system means opening many files. 784 of these component modules serve a single system. The pattern goes back to the
restructure, and later changes copied it.

## What Changes

- Every system module holds the elements, interactions, views, and helpers that only its system uses, beside its
  factory and its `register_system`. The one-class component modules that serve a single system are folded into the
  module of that system and removed. Components reached only through other components follow the chain to their
  system.
- A component that several systems of one field share lives in the field's `common` package, such as
  `skcomplex.systems.biology.common`, when sharing makes sense. Otherwise each system keeps its own copy, under a name
  that fits it. The existing `common` packages stay, and anything in them that only one system uses moves back into
  that system's module.
- Nothing is shared across fields. A component that two fields copy is reviewed: when it is domain-agnostic
  mathematics it moves to the core under a mathematical name, after the user approves the list. Otherwise the copies
  stay.
- The test modules follow the source, so `tests/systems/<field>/<subfield>/test_<module>.py` tests one self-contained
  system module.
- The names a subfield package exports stay importable from it, so code that imports a component from its subfield
  keeps working.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `package-layout`: system modules are self-contained, and shared components of a field live in its `common` package.

## Impact

Most modules of `skcomplex.systems` change, and so do their tests. Private module paths such as
`skcomplex.systems.biology.collective_behaviour._schooling` disappear. Public names exported by the subfield packages do
not change. The import contracts of the checks session keep their field rules, with `common` as the only shared
package of a field.
