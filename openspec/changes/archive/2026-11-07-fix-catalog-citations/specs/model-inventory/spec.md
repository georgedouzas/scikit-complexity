## MODIFIED Requirements

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
