"""Role a declared argument plays in a part."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from enum import Enum


class VariableKind(Enum):
    """Role a declared argument plays in a part."""

    STATE = 'state'
    PARAMETER = 'parameter'
    ROLE = 'role'
