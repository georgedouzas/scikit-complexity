# Spec Delta

## MODIFIED Requirements

### Requirement: Listing models imports no model
Importing `skcomplex.models` and listing the catalog SHALL import no model module and no formalism. Each listed entry
SHALL give the name, field path, description, and tags that the model's registration declares.

#### Scenario: Catalog without imports
- **WHEN** a user imports `skcomplex.models` and calls `catalog()` in a fresh interpreter
- **THEN** every registered model is listed, and no module of `skcomplex.models.<field>` and no module of
  `skcomplex.formalisms` is loaded

#### Scenario: Filter by field
- **WHEN** a user calls `catalog(field='physics/mechanics')` in a fresh interpreter
- **THEN** the models of that field path are listed and no model module is imported
