# Spec Delta

## Purpose

Records every model surveyed from the reference books and articles, with its source, scientific field, and status, so
the library's coverage and the interface's missing features are visible and tracked as specified changes.

## ADDED Requirements

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
- **WHEN** a user opens the coverage page and the section `earth/climate`
- **THEN** it lists every surveyed climate model with its sources, and links each implemented one to its documentation
