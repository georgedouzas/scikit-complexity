## Why

Many docstrings of the core do not say what kind of thing a class is. The user quoted the compartment element as an
example: "Compartment, the amount of units in one named state, as a count or a fraction. The label of the element names
the compartment, such as 'A', 'B', or 'source'." A reader cannot tell from it that the class is an element that holds
one nonnegative number, or where it is used. Other docstrings run to several hundred words of internal detail, such as
those of the coupling, transfer, and sampling interactions. The user guide has the same problem in places. It names a
concept, such as a distribution over activity classes, before it shows it in code. The core is meant to be
cross-domain, and its documents should show that with concrete uses from several fields.

Builders of systems also need more than plain docstrings. They need a full reference of every core component, of
what it models and how the engine runs it, and tutorials that take a system from building to sharing. Today the guide
explains the parts by topic, with no reference per class, and some workflows, such as evaluating and calibrating a
system, have no page at all. The prose should also read as written by a careful person, not by a machine.

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
- Write natural prose. No semicolons and no dashes as punctuation, no stock openers, closers, or filler words such as
  "it is worth noting", "delve", "leverage", "seamless", "robust", or "powerful", no padding in threes, paragraphs
  where a list adds nothing, varied sentences, precise verbs, and every claim backed by code or a reference. Each
  page is read through for flow before review.
- Add a reference of the core components for builders of systems.
  - A page on how elements, interactions, laws, spaces, times, and systems fit together, with Mermaid diagrams and a
    worked example built step by step.
  - One hand-written page per family under `docs/overview/user_guide/core_components`: elements, interactions, laws,
    spaces with their neighbourhoods and times, and systems. Every class gets a section on what it models, how the
    engine runs it, how it combines with other parts, its limits and errors, and how to subclass it into a field
    realization, with runnable examples from two or more fields, such as `NonnegativeElement` with `RateInteraction`
    for an epidemic and a bank account, `CouplingInteraction` for fireflies and for consensus, and `BallNeighborhood`
    for a sandpile and a voter model.
  - The docstring of every core component gains a comparison paragraph, "Unlike `X`, it ... Choose it when ...", and
    a uses paragraph, "It appears in ...".
  - The generated parts pages gain one comparison table per family, built from those paragraphs, and one fact sheet
    per class with its states, roles, parameters, law kinds, the times, solvers, and backends its tags accept, its
    errors, and links to its reference section, its API entry, and the catalog systems that use or subclass it. A new
    parts page covers spaces, neighbourhoods, and times. Catalog pages link back to the core classes they use.
    Explanations are written once, by hand, and facts, tables, and links are generated.
- Add a tutorial for every workflow of a system, each with one running example, the hare and lynx pelt records:
  build, check, simulate, analyse, evaluate, calibrate (every method), experiment, visualise, and share.
- Add three case studies that run the whole workflow in different fields: an epidemic calibrated to outbreak counts,
  an opinion model swept and analysed, and a lattice spin model with its phase transition.
- Add a glossary of mathematical and library terms, a decision guide from the features of a problem to the parts that
  fit it, and a troubleshooting page with one section per error class of the core.
- Add a light automated guard to the test session. It checks the length of summary lines, semicolons, dashes used as
  punctuation, and banned phrases in docstrings and documents, the article and kind word of class summary lines,
  field terms in summary lines, the word "verb", and that the reference and troubleshooting pages cover every class.
  It is not a style linter.
- Work in a reviewable order: a pilot of the elements package and the coupling interaction for the user to approve
  the style, then the other packages of the core, then a documents pilot of the elements reference page, the page on
  how the parts fit, and the build tutorial, then the other reference pages, tutorials, case studies, and extras.
- No public name, signature, or behaviour changes. The change lands after `add-system-package`,
  `slim-core-elements-and-spaces`, `slim-core-interactions`, `slim-core-laws`, and
  `hoist-constants-and-helper-methods`, and is written against the names they leave. The documents also wait for
  `add-identifiability-analysis`, whose `identifiability` the evaluate tutorial uses.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `user-documentation`: docstrings and documents follow testable rules for summary lines, bodies, comparisons,
  cross-domain uses, arguments, jargon, module docstrings, natural prose, and code examples. The guide holds a
  reference of the core components, workflow tutorials, case studies, a glossary, a decision guide, and a
  troubleshooting page, in a task-led order that the modified requirement fixes. The parts pages hold generated fact
  sheets and comparison tables, and the test session guards the mechanical rules.

## Impact

- Code: docstrings only, in every module of `src/skcomplex/core`. No code line changes.
- Tests: new guard tests in `tests/core/components/test_base.py`, beside the field-term test, and checks of the
  documents and of page coverage in `tests/test_sources.py`. Every rewritten doctest runs in the test session as before.
- Documents: `README.md`, `CONTRIBUTING.md`, the pages under `docs/overview/user_guide` with the new sections
  `core_components` and `case_studies` and the new pages `check.md`, `evaluate.md`, `calibrate.md`, `share.md`,
  `glossary.md`, and `troubleshooting.md`, the navigation and a Mermaid fence in `properdocs.yml`, the generators
  `docs/generate_parts.py` (fact sheets, comparison tables, and a spaces page) and `docs/generate_systems.py` (links
  to core classes), and the text cells of the gallery examples under `docs/examples`. `concepts.md` and
  `choosing_compatible_components.md` merge into the new pages.
- Users: none in code. Readers get plainer docstrings and API pages.
- Dependencies: no new package. Mermaid renders through the Material theme already in use, and griffe, which parses
  the docstrings for the fact sheets, is installed with mkdocstrings as `griffelib`. The docs group of
  `pyproject.toml` names it, since the generator imports it directly. The change lands after the five changes named above,
  which rename and remove classes such as `CompartmentElement`, `DensityElement`, `DriveLaw`, and
  `LifeCycleInteraction`, and its documents after `add-identifiability-analysis`.
