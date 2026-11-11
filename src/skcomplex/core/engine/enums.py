"""Define the kinds that only the engine reads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from enum import IntEnum, StrEnum


class LinearProgramBound(IntEnum):
    """A bound of a column or a row of a linear program that a compiled value gives."""

    COLUMN_LOWER = 0
    COLUMN_UPPER = 1
    ROW_LOWER = 2
    ROW_UPPER = 3


class ReadKind(StrEnum):
    """A kind of read of a symbol for each evaluation index of a kernel."""

    MEMBER = 'member'
    SINGLE = 'single'
    FIRST = 'first'
    SECOND = 'second'
    NEIGHBOR = 'neighbor'
    DERIVED = 'derived'
    REDUCED = 'reduced'
    OTHER = 'other'
    SOURCE = 'source'
