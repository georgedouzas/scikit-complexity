"""Import the optional libraries that some models need."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from importlib import import_module
from types import ModuleType

from ..errors import MissingExtraError


def import_extra(module: str, model: str, extra: str) -> ModuleType:
    """Return a module of an optional extra, or raise an error that names the model, the extra, and the install command.

    Args:
        module:
            The dotted name of the module.

        model:
            The name of the model that needs the module.

        extra:
            The name of the optional extra that holds the module.

    Returns:
        module:
            The imported module.

    Raises:
        MissingExtraError:
            The module is not installed.

    Examples:
        >>> from skcomplex.models import import_extra
        >>> import_extra('math', 'pendulum', 'none').sqrt(4.0)
        2.0
    """
    try:
        return import_module(module)
    except ImportError as error:
        error_msg = (
            f"The model {model} needs the extra '{extra}'. Install it with `pip install 'scikit-complexity[{extra}]'`."
        )
        raise MissingExtraError(error_msg) from error
