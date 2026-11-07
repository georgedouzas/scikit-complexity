"""Define the types of the experiments: the rows of a design and the bounds of its parameters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping

from ..components.system import BaseSystem, System
from ..engine.runtime import Trajectory

type Row = dict[str, object]
type Bounds = Mapping[str, object]
type Source = Callable[..., BaseSystem] | System
type Output = str | Callable[[Trajectory], object]
