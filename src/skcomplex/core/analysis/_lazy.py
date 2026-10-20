"""Import heavy third-party libraries on first use, so importing the analysis package stays fast."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import cache
from importlib import import_module
from types import ModuleType


@cache
def module(name: str) -> ModuleType:
    """Return a module, importing it on the first call.

    Args:
        name:
            The dotted name of the module.

    Returns:
        module:
            The imported module.

    Examples:
        >>> from skcomplex.core.analysis._lazy import module
        >>> module('math').sqrt(4.0)
        2.0
    """
    return import_module(name)
