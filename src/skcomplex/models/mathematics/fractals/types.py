"""Define the types of the fractals: the affine maps of a chaos game and the windows of a quadratic map."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

type AffineMap = Sequence[float]
type Window = tuple[float, float, float, float]
