# system-inventory Specification

## Purpose
Records every system surveyed from the reference books and articles, with its source, scientific field, and status, so
the library's coverage and the interface's missing features are visible and tracked as specified changes.

## Requirements

### Requirement: The inventory records surveyed systems
The project SHALL keep an inventory of surveyed systems. Each entry SHALL give the system's name, its field path, its
sources by book section or arXiv identifier, and its status: implemented, pending a planned change, or blocked by a
missing feature. No surveyed system SHALL be out of scope: a system the core cannot express is blocked on the change that
specifies the general capability it needs. The sources of an implemented entry SHALL name sections that hold the rules
the system implements, and when no surveyed section holds them, the primary source of those rules. The docstring and
the registered references of the system SHALL agree with these sources.

#### Scenario: Entry of a book system
- **WHEN** a contributor reads the inventory entry of the Bak-Sneppen system
- **THEN** it names Gros 2015 §6.6, Thurner et al. 2018 §5.4.4, and Jensen §4.2 as sources, `biology/evolution` as its
  field, and its status

#### Scenario: System whose rules differ from a surveyed chapter
- **WHEN** a catalog system implements rules that a surveyed chapter on the same topic does not hold, such as a
  zero-intelligence double auction next to a chapter on adaptive bidding
- **THEN** its entry cites the primary source of its rules and not that chapter, and its docstring says which model it
  implements

### Requirement: The inventory agrees with the catalog
Every inventory entry marked implemented SHALL name a registered system of the same field path, and every registered
system SHALL have an inventory entry.

#### Scenario: Missing registration
- **WHEN** an inventory entry is marked implemented and no system of that name is registered
- **THEN** the inventory check fails and names the entry

#### Scenario: Unrecorded system
- **WHEN** a system is registered without an inventory entry
- **THEN** the inventory check fails and names the system

### Requirement: Missing features are specified, not worked around
A system that needs a feature the interface lacks SHALL be recorded as blocked, naming the feature and the OpenSpec
change that specifies it, and the library SHALL NOT ship a version of the system that works around the interface.

#### Scenario: Blocked system
- **WHEN** an inventory entry is marked blocked
- **THEN** it names the missing feature and an OpenSpec change that exists, active or archived, which specifies it

#### Scenario: Pending system
- **WHEN** an inventory entry is marked pending
- **THEN** it names an active OpenSpec change whose specs list the system

### Requirement: Features build on mature libraries
A change that specifies a missing feature SHALL survey the popular, mature, and performant libraries that implement
it, and SHALL build the feature on the chosen library unless its design states why none fits.

#### Scenario: Library survey in a gap change
- **WHEN** a contributor reads the design of a change that specifies a missing feature
- **THEN** it names the libraries it considered and the one the feature wraps, or why it implements the feature itself

### Requirement: Every registered system has a page
The documentation SHALL give every registered system its own page, reached from a navigation grouped by field and
subfield, and every field an index page that links its systems. The inventory SHALL stay a checked source file, and the
documentation SHALL NOT publish a page of its entries, statuses, or coverage.

#### Scenario: System page
- **WHEN** a user opens the section `physics/climate` under Systems in the navigation
- **THEN** it lists every registered climate system, and each entry opens the page of that system

### Requirement: Systems exist to exercise the core
The library's systems SHALL serve first to reveal what the core needs and to exercise its general primitives, and only
second as ready-made examples. Every inventory entry SHALL record the core primitives it exercises, named from a
vocabulary the inventory defines, and the inventory check SHALL reject a primitive outside that vocabulary.

#### Scenario: Primitives of an entry
- **WHEN** a contributor reads the inventory entry of `bak_sneppen`
- **THEN** it lists the lattice and extremal-update primitives it exercises

#### Scenario: Unknown primitive
- **WHEN** an entry names a primitive that the vocabulary does not define
- **THEN** the inventory check fails and names the entry and the primitive

### Requirement: Surveys prioritise new primitives
Every inventory entry SHALL carry a priority: `core` when it exercises a primitive or a combination of primitives that
no implemented system exercises, or needs a missing primitive, and `example` when it only repeats covered patterns. A
change that implements systems SHALL implement its `core` entries first, and MAY leave `example` entries pending.

#### Scenario: Priority of a repeated pattern
- **WHEN** a survey finds a one-population ODE growth law whose primitives every growth system already exercises
- **THEN** its entry has priority `example`, and a change that implements systems reaches it after the `core` entries

#### Scenario: Priority of a new primitive
- **WHEN** a survey finds a system that needs states on the links of a lattice
- **THEN** its entry has priority `core` and is blocked on the change that adds link fields

### Requirement: Entries map to the system vocabulary
Every inventory entry SHALL declare its time kind, its space kind, its component kinds, its interaction bindings, and
its law classes from the vocabulary of `system-modeling`, implemented and pending entries alike. A check SHALL fail on any
entry that does not map.

#### Scenario: Unmapped entry
- **WHEN** an entry lists a law class that the vocabulary does not define
- **THEN** the inventory check fails, names the entry, and asks for the general primitive to be specified

### Requirement: Every reference source is surveyed section by section
Every textbook and paper in `references` SHALL be surveyed section by section, and the survey SHALL be recorded with the
sections read, the systems found, and the tags of each in the system vocabulary, so a source with few inventory entries is a source
with few systems and not an unread one.

#### Scenario: Thinly cited textbook
- **WHEN** a contributor reads the survey record of a textbook cited by four inventory entries or fewer
- **THEN** it lists every chapter read and, for each, the systems recorded or the reason none was

#### Scenario: Paper in the references
- **WHEN** a paper is added to `references/papers`
- **THEN** its systems are in the inventory under its arXiv identifier, or its survey record says why it has none

### Requirement: The docs build runs the inventory check
The documentation build SHALL run the inventory check before it writes the system pages, and SHALL fail with every issue
the check finds:
- entries marked implemented with no registered system;
- registered systems with no entry;
- unknown primitives;
- entries that do not map to the system vocabulary;
- statuses that name changes which do not exist.

#### Scenario: Inventory disagrees with the catalog
- **WHEN** an inventory entry is marked implemented and no system of that name is registered, and the documentation is
  built
- **THEN** the build fails and its message names the entry

#### Scenario: Consistent inventory
- **WHEN** the inventory agrees with the catalog and follows its rules, and the documentation is built
- **THEN** the build writes the system pages
