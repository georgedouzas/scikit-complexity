"""Define the types of experiments, the systems they run, rows, and outputs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping

from ..components.systems import BaseSystem
from ..engine.runtime import Trajectory

type Row = dict[str, object]
type Bounds = Mapping[str, object]
type Source = BaseSystem | Callable[..., BaseSystem]
type Output = str | Callable[[Trajectory], object]
