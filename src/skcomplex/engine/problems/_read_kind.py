"""How a symbol is read for each evaluation index of a kernel."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from enum import Enum


class ReadKind(Enum):
    """How a symbol is read for each evaluation index of a kernel."""

    MEMBER = 'member'
    SINGLE = 'single'
    FIRST = 'first'
    SECOND = 'second'
    NEIGHBOR = 'neighbor'
    DERIVED = 'derived'
    REDUCED = 'reduced'
    OTHER = 'other'
