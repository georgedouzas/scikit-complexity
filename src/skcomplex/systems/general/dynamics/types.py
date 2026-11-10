"""Define the types of dynamical systems, their equations, noises, and flow tables."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from types import SimpleNamespace

from ....core.components.laws import Sink, Source

type Equations = Mapping[str, Callable[[SimpleNamespace], object]]
type Noises = Mapping[str, Callable[[SimpleNamespace], object] | float]
type FlowTable = Mapping[str, tuple[str | Source | None, str | Sink | None, Callable[[SimpleNamespace], object]]]
