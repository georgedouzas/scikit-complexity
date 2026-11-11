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


class LinearOperatorKind(StrEnum):
    """A kind of linear operator that a term of a field rate applies to a state."""

    LAPLACIAN = 'laplacian'
    PARTIAL = 'partial'
    INVERSE_HELMHOLTZ = 'inverse_helmholtz'


class ComplexPartKind(StrEnum):
    """A kind of part of a complex value that a visual draws, such as its modulus.

    Examples:
        >>> from skcomplex.core.engine.enums import ComplexPartKind
        >>> f'state/{ComplexPartKind.PHASE}'
        'state/phase'
    """

    REAL = 'real'
    IMAG = 'imag'
    MODULUS = 'modulus'
    PHASE = 'phase'
