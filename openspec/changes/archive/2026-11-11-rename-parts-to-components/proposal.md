# Proposal

## Why

One concept has two names. The package is `skcomplex.core.components` and the guide section is
`docs/overview/user_guide/core_components`, while the base class of every element, interaction, law, space, time, and
system is `BasePart`, and the docstrings, the guide, the placement rule, and the specs call these building blocks the
parts of a system. A reader meets "component" in the import path and "part" in the text that explains it. Component
is the standard word of systems theory for a building block of a system, and the package already uses it, so the
vocabulary is unified on it.

The rename is not a find and replace. "Part" also names real and imaginary parts, the symmetric part of a tensor, the
sites, edges, and hyperedges of a space, the parts of a delay, the parts of a word or a patch, the parts of a watch or
a product in two models, and ordinary English. "Component" already names vector, tensor, and connected components and
the `n_components` of decompositions. Those meanings stay, and the design gives the rule that tells them apart.

## What Changes

- **BREAKING** `BasePart` of `skcomplex.core.components.base` is renamed `BaseComponent`. No alias is kept, as for the
  earlier removal of `ArgumentError`. Code that subclasses or imports `BasePart` imports `BaseComponent` instead.
- **BREAKING** The public method `GroupElement.get_declaring_part` is renamed `get_declaring_element`, since what it
  returns is always an element class or an element. No alias is kept.
- Private names where "part" means a building block of a system are renamed, such as `_get_part_specs` to
  `_get_component_specs`, `CompositeSystem._get_parts` to `_get_components`, and `FlattenPass._prepare_part` to
  `_prepare_component`. The parts of a stock and flow file become its variables: `_SystemPartsSpec` becomes
  `_SystemVariablesSpec`, since "component" there already means a definition of a variable in PySD.
- Docstrings, error messages, and parameter names say component wherever they meant a building block of a system,
  such as `Return the components of the system` and `The components 'births' and 'deaths' disagree on the tag 'time'`.
- The README, CONTRIBUTING, the constitution, the user guide, the glossary, the troubleshooting page, and the
  placement rule say component. The guide page `core_components/choosing_parts.md` becomes
  `core_components/choosing_components.md`, and `build_a_custom_system/connected_components.md` becomes
  `build_a_custom_system/ports_and_connections.md`, so its title no longer reads as graph theory.
- The generated reference pages move from `parts/<family>.md` to `components/<family>.md`, built by
  `docs/generate_components.py` in place of `docs/generate_parts.py`, as the maintainer chose. Each former URL
  redirects to the new one with the anchor kept, through the `mkdocs-redirects` plugin.
- The main specs whose requirement text says part for a building block are updated through delta specs, and the
  drafted changes under `openspec/changes` are reworded in place.
- `openspec/scripts/check_conventions.py` gains a check that flags "part" or "parts" used for a building block of a
  system in the identifiers and docstrings of `src`, in the identifiers of `tests`, and in the documentation pages,
  with an allow list for the meanings that stay, and a self-check against planted files.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `base-interfaces`: the requirements on run attributes, preparation, estimators, the system hierarchy, ready-made
  systems, and general systems say component and name `BaseComponent`.
- `package-layout`: the requirements on the base of components, unrelated bases, the public surface, the systems
  package, the base module of systems, and the placement rule say component, and a new requirement names one word for
  the building blocks of a system and the check that keeps it.
- `system-modeling`: the requirements on what a system holds, nested systems, the shared main public methods, and the
  noise processes say component, and the requirement that a system is not a component is reversed, since
  `BaseSystem` derives from `BaseComponent`.
- `capability-tags`: components declare tags and systems derive them from their components.
- `system-validation`: an unknown variant names the component at fault.
- `user-documentation`: the family pages, the docstring requirements, the recipes page, the overview page, the
  decision guide, and the runnable documents say component, the family pages move to `components/`, and moved pages
  redirect.
- `declared-state-elements`: the library generates no component classes.
- `general-systems`: a shape covered by one core component, and tags that follow from the components a dynamical
  system builds.
- `general-realizations`: core features return composites of core components.
- `experiments`: designs sweep dotted initial states or component parameters.
- `model-library`: a ready-made system keeps an element class that is a component others compose.
- `constraint-based-models`: a switch may be set by an `UpdateLaw` of another component.
- `rewriting-systems`: the elements and laws of rewriting are the rewriting components that live in the core.

## Impact

- Code: `core/components/base.py` and every module that imports `BasePart` across `core/components`, `core/engine`,
  and the field packages of `skcomplex.systems`, the docstrings of the system classes, and the translators of
  `core/engine/translators`, with `_system_parts_spec.py` renamed `_system_variables_spec.py`.
- Tests: the classes and the test names of `tests/core/components/test_base.py`, `tests/core/test_base.py`,
  `tests/core/engine/solvers/test_root.py`, `tests/core/components/laws/utils/test_reductions.py`, and the test names
  that say part for a component. No test file is added except the mirror of a renamed module, and no test checks that
  the old names are gone.
- Docs: the user guide, the glossary, the troubleshooting page, the generators in `docs`, `docs/gallery_hooks.py`,
  `properdocs.yml`, the README, and CONTRIBUTING. Published links to `parts/` and to the two renamed guide pages keep
  working through redirects, and `mkdocs-redirects` joins the `docs` group of development dependencies. Links to renamed headings, such as `#where-a-new-part-goes`, land at the
  top of their page.
- Specs and plans: the main specs listed above, the constitution, and the drafted changes. The archive keeps its
  history unchanged, including the old `BaseComponent` that was renamed `BaseElement` in October 2026.
- Users: code that imports `BasePart` or calls `get_declaring_part` breaks. The commit is a breaking change, so the
  release notes of the next release say so.
