# model-inventory Specification

## Purpose
Records every model surveyed from the reference books and articles, with its source, scientific field, and status, so
the library's coverage and the interface's missing features are visible and tracked as specified changes.

## Requirements

### Requirement: The inventory records surveyed models
The project SHALL keep an inventory of surveyed models. Each entry SHALL give the model's name, its field path, its
sources by book section or arXiv identifier, and its status: implemented, pending a planned change, or blocked by a
missing feature. No surveyed model SHALL be out of scope: a model the core cannot express is blocked on the change that
specifies the general capability it needs.

#### Scenario: Entry of a book model
- **WHEN** a contributor reads the inventory entry of the Bak-Sneppen model
- **THEN** it names Gros 2015 §6.6, Thurner et al. 2018 §5.4.4, and Jensen §4.2 as sources, `biology/evolution` as its
  field, and its status

### Requirement: The inventory agrees with the catalog
Every inventory entry marked implemented SHALL name a registered model of the same field path, and every registered
model SHALL have an inventory entry.

#### Scenario: Missing registration
- **WHEN** an inventory entry is marked implemented and no model of that name is registered
- **THEN** the inventory check fails and names the entry

#### Scenario: Unrecorded model
- **WHEN** a model is registered without an inventory entry
- **THEN** the inventory check fails and names the model

### Requirement: Missing features are specified, not worked around
A model that needs a feature the interface lacks SHALL be recorded as blocked, naming the feature and the OpenSpec
change that specifies it, and the library SHALL NOT ship a version of the model that works around the interface.

#### Scenario: Blocked model
- **WHEN** an inventory entry is marked blocked
- **THEN** it names the missing feature and an OpenSpec change that exists, active or archived, which specifies it

#### Scenario: Pending model
- **WHEN** an inventory entry is marked pending
- **THEN** it names an active OpenSpec change whose specs list the model

### Requirement: Features build on mature libraries
A change that specifies a missing feature SHALL survey the popular, mature, and performant libraries that implement
it, and SHALL build the feature on the chosen library unless its design states why none fits.

#### Scenario: Library survey in a gap change
- **WHEN** a contributor reads the design of a change that specifies a missing feature
- **THEN** it names the libraries it considered and the one the feature wraps, or why it implements the feature itself

### Requirement: The inventory is published
The documentation SHALL publish the inventory as a coverage page grouped by field and subfield, with every entry's
sources and status and, for implemented models, a link to the model's documentation.

#### Scenario: Coverage page
- **WHEN** a user opens the coverage page and the section `physics/climate`
- **THEN** it lists every surveyed climate model with its sources, and links each implemented one to its documentation

### Requirement: Models exist to exercise the core
The library's models SHALL serve first to reveal what the core needs and to exercise its general primitives, and only
second as ready-made examples. Every inventory entry SHALL record the core primitives it exercises, named from a
vocabulary the inventory defines, and the inventory check SHALL reject a primitive outside that vocabulary.

#### Scenario: Primitives of an entry
- **WHEN** a contributor reads the inventory entry of `bak_sneppen`
- **THEN** it lists the lattice and extremal-update primitives it exercises

#### Scenario: Unknown primitive
- **WHEN** an entry names a primitive that the vocabulary does not define
- **THEN** the inventory check fails and names the entry and the primitive

### Requirement: Coverage is reported by primitive
The coverage page SHALL show, for every primitive of the vocabulary, the implemented models that exercise it, and SHALL
flag every primitive that no implemented model exercises and every primitive that a pending or blocked change adds.

#### Scenario: Uncovered primitive
- **WHEN** a primitive of the vocabulary is exercised only by models blocked on a change
- **THEN** the coverage page lists it as uncovered and names the change that adds it

### Requirement: Surveys prioritise new primitives
Every inventory entry SHALL carry a priority: `core` when it exercises a primitive or a combination of primitives that
no implemented model exercises, or needs a missing primitive, and `example` when it only repeats covered patterns. A
change that implements models SHALL implement its `core` entries first, and MAY leave `example` entries pending.

#### Scenario: Priority of a repeated pattern
- **WHEN** a survey finds a one-population ODE growth law whose primitives every growth model already exercises
- **THEN** its entry has priority `example`, and a change that implements models reaches it after the `core` entries

#### Scenario: Priority of a new primitive
- **WHEN** a survey finds a model that needs states on the links of a lattice
- **THEN** its entry has priority `core` and is blocked on the change that adds link fields

### Requirement: Entries map to the system vocabulary
Every inventory entry SHALL declare its time kind, its space kind, its component kinds, its interaction bindings, and
its law classes from the vocabulary of `system-modeling`, implemented and pending entries alike. A check SHALL fail on any
entry that does not map. Coverage SHALL be reported per category of the vocabulary.

#### Scenario: Unmapped entry
- **WHEN** an entry lists a law class that the vocabulary does not define
- **THEN** the inventory check fails, names the entry, and asks for the general primitive to be specified

#### Scenario: Coverage by category
- **WHEN** the inventory is published
- **THEN** its coverage table counts implemented and pending entries for every time kind, space kind, component kind,
  binding, and law class

### Requirement: Every reference source is surveyed section by section
Every textbook and paper in `references` SHALL be surveyed section by section, and the survey SHALL be recorded with the
sections read, the models found, and the tags of each in the system vocabulary, so a source with few inventory entries is a source
with few models and not an unread one.

#### Scenario: Thinly cited textbook
- **WHEN** a contributor reads the survey record of a textbook cited by four inventory entries or fewer
- **THEN** it lists every chapter read and, for each, the models recorded or the reason none was

#### Scenario: Paper in the references
- **WHEN** a paper is added to `references/papers`
- **THEN** its models are in the inventory under its arXiv identifier, or its survey record says why it has none

### Requirement: The docs build runs the inventory check
The documentation build SHALL run the inventory check before it writes the model pages, and SHALL fail with every issue
the check finds:
- entries marked implemented with no registered model;
- registered models with no entry;
- unknown primitives;
- entries that do not map to the system vocabulary;
- statuses that name changes which do not exist.

#### Scenario: Inventory disagrees with the catalog
- **WHEN** an inventory entry is marked implemented and no model of that name is registered, and the documentation is
  built
- **THEN** the build fails and its message names the entry

#### Scenario: Consistent inventory
- **WHEN** the inventory agrees with the catalog and follows its rules, and the documentation is built
- **THEN** the build writes the model pages and the coverage page
