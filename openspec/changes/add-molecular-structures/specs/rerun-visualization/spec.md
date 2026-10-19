# Spec Delta

## ADDED Requirements

### Requirement: Chains are drawn as points and bonds
A chain SHALL stream each frame as points at its sites and line strips along its bonds, in three dimensions for
Cartesian positions and in the plane for a layout such as an RNA secondary structure, with base pairs drawn as extra
line strips. The points SHALL be coloured by a chosen site value with a fixed colour range. A sequence SHALL also stream
as an image with one row per copy and one colour per letter, and every score SHALL stream as a time series.

#### Scenario: Protein chain in three dimensions
- **WHEN** the `fuzzy_oil_drop` model streams with a Rerun observer
- **THEN** the entity `<chain>/sites` holds 20 three-dimensional points at every frame, `<chain>/bonds` holds the line
  strip through them, and the points are coloured by `O − T`

#### Scenario: Secondary structure in the plane
- **WHEN** the `rna_neutral_networks` model streams its target structure
- **THEN** the viewer shows 30 points in the plane joined by the backbone, and one extra segment per base pair of the
  target

#### Scenario: Sequences as an image
- **WHEN** 8 walkers of length 30 stream
- **THEN** the entity `<population>/sequences` holds an 8 by 30 image at every frame, with four colours
