## Why

Many docstrings of the core do not say what kind of thing a class is. The user quoted the compartment element as an
example: "Compartment, the amount of units in one named state, as a count or a fraction. The label of the element names
the compartment, such as 'A', 'B', or 'source'." A reader cannot tell from it that the class is an element that holds
one nonnegative number, or where it is used. Other docstrings run to several hundred words of internal detail, such as
those of the coupling, transfer, and sampling interactions. The user guide has the same problem in places. It names a
concept, such as a distribution over activity classes, before it shows it in code. The core is meant to be
cross-domain, and its documents should show that with concrete uses from several fields.

## What Changes

- Rewrite the docstrings of `skcomplex.core`: the parts of `components` (elements, interactions, laws, spaces,
  systems, specs, base, and types), the public classes of `engine` (runtime, viz, backends, solvers, compile, problems,
  translators), `analysis`, `experiments`, `specs`, and the root modules `base`, `errors`, and `types`.
  - The summary line says plainly what kind of thing it is, in at most about 80 characters, such as "An element that
    holds one nonnegative number."
  - A short body in plain words follows, then concrete uses from two or more fields where the part is cross-domain,
    such as people infected in an epidemic, moles of a chemical species, money in an account, and tokens in a place of
    a Petri net.
  - `Args` say what a value means, with its units or range, not how the code stores it.
  - Terms of one field or of the library's internals appear only with a plain definition.
  - Doctest examples stay runnable. Where a cross-domain use fits as code, the example shows it.
- Rewrite module docstrings in the same style. A private module keeps one imperative line, and a package `__init__`
  adds at most one short paragraph.
- Rewrite the prose of the documents in the same style: the README, the user guide under `docs/overview`, the pages
  that `docs/generate_parts.py` and `docs/generate_systems.py` write, `CONTRIBUTING.md`, and the text of the gallery
  examples. Every concept is shown with a concrete code block that runs. The prose is short, precise, academic, and
  simple, neither informal nor dense with jargon.
- Add a core components guide, so the documents describe the core parts, how they differ, what they offer, and how
  they support systems in any field.
  - One hand-written page per family under `docs/overview/user_guide/core_components`: elements, interactions, laws,
    spaces with their neighbourhoods and times, and systems. Each page opens with what the family is for and how it
    fits with the others, then gives every class a section with runnable examples from two or more fields, such as
    `NonnegativeElement` with `RateInteraction` for an epidemic and a bank account, `CouplingInteraction` for fireflies
    and for consensus, and `BallNeighborhood` for a sandpile and a voter model.
  - The docstring of every core component gains a comparison paragraph, "Unlike `X`, it ... Choose it when ...", and
    a uses paragraph, "It appears in ...".
  - The generated parts pages gain one comparison table per family, built from those paragraphs, with links to the
    catalog systems that use each class or subclass it. A new parts page covers spaces, neighbourhoods, and times.
    Explanations are written once, by hand, and tables and links are generated.
- Add a light automated guard to the test session. It checks the length of summary lines, semicolons in docstrings
  and documents, the article and kind word of class summary lines, field terms in summary lines, and the word "verb".
  It is not a style linter.
- Work in a reviewable order: a pilot of the elements package and the coupling interaction for the user to approve
  the style, then the other packages of the core, then a pilot of the elements page of the core components guide and
  two existing guide pages, then the other family pages and documents.
- No public name, signature, or behaviour changes. The change lands after `add-system-package`,
  `slim-core-elements-and-spaces`, `slim-core-interactions`, `slim-core-laws`, and
  `hoist-constants-and-helper-methods`, and is written against the names they leave.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `user-documentation`: docstrings and documents follow testable rules for summary lines, bodies, comparisons,
  cross-domain uses, arguments, jargon, module docstrings, and code examples. The guide describes the core components
  family by family, the parts pages compare them in generated tables, and the test session guards the mechanical
  rules.

## Impact

- Code: docstrings only, in every module of `src/skcomplex/core`. No code line changes.
- Tests: new guard tests in `tests/core/components/test_base.py`, beside the field-term test, and a semicolon check of
  the documents in `tests/test_sources.py`. Every rewritten doctest runs in the test session as before.
- Documents: `README.md`, `CONTRIBUTING.md`, the pages under `docs/overview/user_guide`, a new section
  `docs/overview/user_guide/core_components` with its entries in `properdocs.yml`, the generators
  `docs/generate_parts.py` (comparison tables and a new spaces page) and `docs/generate_systems.py`, and the text
  cells of the gallery examples under `docs/examples`.
- Users: none in code. Readers get plainer docstrings and API pages.
- Dependencies: lands after the five changes named above, which rename and remove classes such as
  `CompartmentElement`, `DensityElement`, `DriveLaw`, and `LifeCycleInteraction`.
