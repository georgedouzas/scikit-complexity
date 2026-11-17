"""Define the types of experiments, the systems they run, rows, and outputs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping

from ..components.systems import BaseSystem as CompositeBaseSystem
from ..engine.runtime import Trajectory
from ..systems import BaseSystem

type Row = dict[str, object]
type Bounds = Mapping[str, object]
type AnySystem = BaseSystem | CompositeBaseSystem
type Source = AnySystem | Callable[..., AnySystem]
type Output = str | Callable[[Trajectory], object]
