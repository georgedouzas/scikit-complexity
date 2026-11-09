"""Define the types of the dynamical systems.

The types are the equations of the states, their noise amplitudes, and the table of the flows of a stock-flow system.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from types import SimpleNamespace

from ....core.components.laws import Sink, Source

type Equations = Mapping[str, Callable[[SimpleNamespace], object]]
type Noises = Mapping[str, Callable[[SimpleNamespace], object] | float]
type FlowTable = Mapping[str, tuple[str | Source | None, str | Sink | None, Callable[[SimpleNamespace], object]]]
