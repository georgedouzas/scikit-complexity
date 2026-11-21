"""Define the types of experiments, the systems they run, rows, and outputs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping

from ..engine.problems._composite_system import _CompositeSystem
from ..engine.runtime import Trajectory
from ..systems import BaseSystem

type Row = dict[str, object]
type Bounds = Mapping[str, object]
type AnySystem = BaseSystem | _CompositeSystem
type SystemSource = AnySystem | Callable[..., AnySystem]
type Output = str | Callable[[Trajectory], object]
