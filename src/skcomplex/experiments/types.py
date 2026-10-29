"""Define the types of the experiments: the rows of a design and the bounds of its parameters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

type Row = dict[str, object]
type Bounds = Mapping[str, object]
