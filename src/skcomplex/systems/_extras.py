"""Import the optional libraries that some systems need."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from importlib import import_module
from types import ModuleType

from ..core.errors import MissingExtraError


def import_extra(module: str, system: str, extra: str) -> ModuleType:
    """Return a module of an optional extra.

    When the extra is missing, it raises an error naming the system, the extra, and the install command.

    Args:
        module:
            The dotted name of the module.

        system:
            The name of the system that needs the module.

        extra:
            The name of the optional extra that holds the module.

    Returns:
        module:
            The imported module.

    Raises:
        MissingExtraError:
            The module is not installed.

    Examples:
        >>> from skcomplex.systems import import_extra
        >>> import_extra('math', 'pendulum', 'none').sqrt(4.0)
        2.0
    """
    try:
        return import_module(module)
    except ImportError as error:
        error_msg = (
            f"The system {system} needs the extra '{extra}'. Install it with "
            f"`pip install 'scikit-complexity[{extra}]'`."
        )
        raise MissingExtraError(error_msg) from error
