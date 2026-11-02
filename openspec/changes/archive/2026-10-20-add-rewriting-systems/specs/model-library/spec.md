# Spec Delta

## ADDED Requirements

### Requirement: Substitution fractals
The library SHALL provide the Koch curve as an L-system with a turtle, the Cantor set as a segment substitution, and
the Sierpiński triangle and carpet as polygon substitutions, as registered factories whose generations are the
simulation's steps (Thurner et al. 2018 §3.5.2, Trzaska 2023 §3.5).

#### Scenario: Koch curve
- **WHEN** `run('koch_curve', until=5)` is called
- **THEN** the word at the last frame has 1024 symbols `F`, the curve's total length is `(4/3)^5` within 1e-12, and
  the box-counting dimension of its raster at resolution 2048, over boxes of 16 to 256 cells, is
  `log 4 / log 3 ≈ 1.262` within 0.05

#### Scenario: Cantor set
- **WHEN** `run('cantor_set', until=8)` is called
- **THEN** `trajectory['cantor.size']` is `2^n` at generation `n`, and the total length is `(2/3)^n` within 1e-12 at
  every frame

#### Scenario: Sierpiński carpet
- **WHEN** `run('sierpinski_substitution', until=4)` is called with `shape='carpet'`
- **THEN** the patch holds 4096 squares and their area is `(8/9)^4` of the unit square within 1e-12

#### Scenario: Gallery analysis
- **WHEN** the gallery example of any of the three models runs
- **THEN** its analysis figure plots the log measure against the generation with the exact line, and the log box count
  against the log box size with the slope of the similarity dimension

### Requirement: Rewriting models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `koch_curve` in `mathematics/fractals` (Th 3.5.2, Tr 3.5)
  - `cantor_set` in `mathematics/fractals` (Th 3.5.2, Tr 3.5)
  - `sierpinski_substitution` in `mathematics/fractals` (Th 3.5.2, Tr 3.5)

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
