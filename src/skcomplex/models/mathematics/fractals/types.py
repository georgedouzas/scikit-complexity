"""Define the types of the fractals: the arrays of a rewritten word or tiling."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....types import AnyArray

type Arrays = list[AnyArray]
type AffineMap = Sequence[float]
type Window = tuple[float, float, float, float]
