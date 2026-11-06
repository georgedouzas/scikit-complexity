"""Define the types of the systems: the functions that build a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

from ..core.components.system import System

type Factory = Callable[..., System]
