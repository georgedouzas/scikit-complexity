"""Define the helpers of agent interactions: the keys of the cells and nodes agents stand on."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Hashable

import numpy as np


def get_location_key(location: object) -> Hashable:
    """Return a location as a key, a tuple of the coordinates of a cell or position, or the node of a network.

    Args:
        location:
            The coordinates of a cell, a position, or a network node.

    Returns:
        key:
            A tuple of integers for a cell, a tuple of floats for a position, or the node itself.
    """
    if isinstance(location, tuple | list | np.ndarray):
        values = np.asarray(location)
        if np.issubdtype(values.dtype, np.integer) or np.all(values == np.round(values)):
            return tuple(int(item) for item in values.ravel())
        return tuple(float(item) for item in values.ravel())
    return location if isinstance(location, Hashable) else repr(location)
