"""Define the types of the experiments.

They are the system they run, the rows of values, and the outputs they keep.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping

from ..components.systems import BaseSystem
from ..engine.runtime import Trajectory

type Row = dict[str, object]
type Bounds = Mapping[str, object]
type Source = BaseSystem | Callable[..., BaseSystem]
type Output = str | Callable[[Trajectory], object]
