# Design

## Context

`all_systems` lives in `skcomplex.systems._catalog`. Its private reader `_SystemClasses` scans the sources with `ast`
and imports only the modules of the classes it returns. Component classes carry no `field` or `topics`, but the class
rules give each kind a name suffix: `Element`, `Interaction`, `Space`, and `Neighborhood`. Every public component is
listed in the `__all__` of a package `__init__`, and many are exported again by a subfield package from the `common`
package of its field. Public class names repeat across fields, such as `GrowthInteraction` in
`biology.common.populations` and `biology.common.metabolism`, and `CarElement` in `engineering.common.traffic` and
`economics.markets`.

After `redesign-core-api`, elements, interactions, spaces, and neighbourhoods derive from `BaseComponent`, with the
bases `BaseElement`, `BaseInteraction`, `BaseSpace`, and `BaseNeighborhood`. Laws are frozen values of seven kinds that
interactions state in `laws_`, and `BaseTime` and its subclasses left the public API. The docstrings of `Element` and
`Interaction` already follow the scikit-learn layout and hold no paragraph on ports, which the redesign removed, or on
the rules of agent members.

## Goals / Non-Goals

**Goals:**

- One function that answers "which part do I import" for every kind of component, from the core and from the systems
  packages.
- Selecting by kind and by package without importing other fields.

**Non-Goals:**

- No topics or tags for components. They declare neither, and inventing them is out of scope.
- No change to `all_systems`, and no systems in the result.
- No laws in the result. Laws are values, not components, and the core holds only the seven kinds.

## Decisions

- **Home: `skcomplex.systems`, next to `all_systems`.** The finder must see both the core and the systems packages,
  and the core never imports the systems packages, so it cannot live in the core. Keeping it in `_catalog.py` keeps
  the two finders side by side. No new module or package is created.
- **Name and arguments: `all_components(type_filter=None, package=None)`.** The code calls these parts components, as
  in `BaseComponent` and `skcomplex.core.components`, and `type_filter` is the scikit-learn name. Its values are
  `'element'`, `'interaction'`, `'space'`, and `'neighborhood'`, one or several. `package` takes the slash path below
  `skcomplex`, the same form `field` takes in `all_systems`, such as `'core'` or `'systems/physics'`.
- **Static reading by `__all__` and suffix.** A private reader, a sibling of `_SystemClasses`, walks the package
  `__init__` files under `skcomplex/core` and `skcomplex/systems`, reads `__all__` statically, and keeps the names whose
  suffix matches a selected kind. It then imports only the selected packages and checks that each class is a subclass
  of the base of its kind, `BaseElement`, `BaseInteraction`, `BaseSpace`, or `BaseNeighborhood`. A name with the suffix
  that is not such a subclass raises `RegistryError`. Detecting subclasses with `ast` alone was rejected, since bases
  are often imported under other names. The package root is found with `importlib.resources.files`.
- **Import paths, not bare names.** The result is `(import path, class)`. Bare names would collide, and the import
  path is the line the user writes. A class exported by several packages keeps the package whose path is the longest
  prefix of the class's `__module__`, which is the package that defines it.
- **Docstrings of `Element` and `Interaction`.** Each keeps its scikit-learn layout. One sentence of the body tells the
  user to look for an existing part with `all_components` before writing a subclass, and the See Also section names
  `all_components`. The example that finds `MassPointElement` lives in the docstring of `all_components` and in the
  guide, so the Examples of the core docstrings stay three to six lines on the class itself and no core docstring
  imports a systems package.
- **Agent rules in the guide.** The agent solver removes a member and refuses to remove one whose values tagged with a
  quantity are not zero, and it reserves names such as `location` and `alive` for its own use. The page
  `agent_based_systems.md` states both, with the names read from the engine at the time of writing.

## Risks / Trade-offs

- [Scanning every `__init__` is slow on first call] → Cache the static scan, as `_SystemClasses` does, and import only
  what is returned.
- [A public component without the class suffix is missed] → The class rules already require the suffix, and the
  convention check enforces it. A test checks that every exported subclass of a component base is found.

## Open Questions

- Should `all_components` also list the law classes that field packages define for their own solvers, such as the
  rewriting laws of `skcomplex.systems.general.rewriting`, under a kind `'law'`? This design leaves them out, since laws
  are values and not components.
