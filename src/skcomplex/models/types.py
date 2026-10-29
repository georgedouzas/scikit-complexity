"""Define the types of the models: the functions that build a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

from ..core.system import System

type Factory = Callable[..., System]
