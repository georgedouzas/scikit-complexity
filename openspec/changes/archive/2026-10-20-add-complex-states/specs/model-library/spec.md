# Spec Delta

## ADDED Requirements

### Requirement: Mandelbrot and Julia sets
The library SHALL provide the Mandelbrot set and the filled Julia sets of the complex quadratic map `z → z² + c`
(Dilão 2023 §5.5) as registered factories under the field path `mathematics/fractals`. Each SHALL iterate the map at
every site of a lattice of complex points and record the step at which the modulus first exceeds an escape radius.

#### Scenario: Mandelbrot escape times
- **WHEN** the Mandelbrot model runs for 100 steps on a lattice whose sites include `c = 1`, `c = -1`, `c = -2`, and
  `c = i`
- **THEN** the site at `c = 1` escapes at step 3, since `0 → 1 → 2 → 5`, and the sites at `c = -1`, `c = -2`, and
  `c = i` never escape

#### Scenario: Julia set of the squaring map
- **WHEN** the Julia model with `c = 0` runs for 100 steps on a 201 × 201 lattice over the square `[-2, 2]²`
- **THEN** every site with `|z₀| < 0.99` never escapes, and every site with `|z₀| > 1.01` has escaped

#### Scenario: Mandelbrot symmetry
- **WHEN** the Mandelbrot model runs on a window symmetric about the real axis
- **THEN** the escape-time image equals its mirror image across the real axis
