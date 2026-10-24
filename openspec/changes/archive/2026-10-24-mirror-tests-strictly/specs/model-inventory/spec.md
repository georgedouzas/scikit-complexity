# Spec Delta

## ADDED Requirements

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
