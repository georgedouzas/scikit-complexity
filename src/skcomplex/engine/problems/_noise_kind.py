"""Structure of the Wiener processes that the noise laws of a system draw."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from enum import StrEnum


class NoiseKind(StrEnum):
    """Structure of the Wiener processes that the noise laws of a system draw."""

    DIAGONAL = 'diagonal'
    SHARED = 'shared'
    CORRELATED = 'correlated'
    CONSERVED = 'conserved'
