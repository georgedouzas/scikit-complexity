# Spec Delta

## ADDED Requirements

### Requirement: Points are joined by a strip in member order
`PointsVisual` SHALL log, with `strip=True`, one line strip through the positions of a group in member order, in two or
three dimensions, beside the points, colored by the same state as the points with a fixed color range.

#### Scenario: Protein chain in three dimensions
- **WHEN** the `fuzzy_oil_drop` model streams with a `RerunObserver`
- **THEN** the recording holds 20 three-dimensional points of the chain at every frame and one line strip through them

#### Scenario: Sequences as an image
- **WHEN** 8 walkers of length 30 of `rna_neutral_networks` stream with `ImageVisual`
- **THEN** the recording holds an 8 by 30 image at every frame, with four colors
