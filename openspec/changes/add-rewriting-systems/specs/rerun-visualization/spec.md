# Spec Delta

## ADDED Requirements

### Requirement: Rewritten geometry is drawn in a fixed frame
A geometry visual SHALL log the turtle segments and the segment tiles of a rewriting element as one batch of
two-dimensional or three-dimensional line strips per frame, and the polygon tiles as outlines or, when filled, as an
image of their raster. Every frame SHALL use the bounding box of the run's largest geometry, so successive generations
are drawn at the same scale, and colours SHALL follow the prototile or the depth of the branch.

#### Scenario: Koch curve in Rerun
- **WHEN** the Koch curve runs 5 generations with a Rerun observer
- **THEN** one `LineStrips2D` per frame is logged under `koch`, the last one holding 1024 segments, and the view
  shows the whole curve at every frame

#### Scenario: Filled gasket
- **WHEN** the Sierpiński triangle is drawn with the filled option at generation 6
- **THEN** one image is logged per frame, and its filled pixels cover `(3/4)^6` of the triangle's pixels within 2%
