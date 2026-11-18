"""Define the types of dynamical systems, their equations, noises, and flow tables."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from types import SimpleNamespace

type Equations = Mapping[str, Callable[[SimpleNamespace], object]]
type Noises = Mapping[str, Callable[[SimpleNamespace], object] | float]
type FlowTable = Mapping[str, tuple[str | None, str | None, Callable[[SimpleNamespace], object]]]
