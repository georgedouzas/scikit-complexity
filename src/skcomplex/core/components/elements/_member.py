"""Define the group whose members a value of an element names."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...errors import ValidationError
from ...specs._base_spec import BaseSpec


@dataclass(frozen=True)
class Member(BaseSpec):
    """A group whose members a value of an element names, such as the firms a worker works for.

    It is written in `Annotated` beside the allowed values, such as
    `employer: Annotated[int, Member('firms')] = -1`. The value is the index of a member of that group, and -1 names
    none. When a member of the group is removed, every value that names it becomes -1, so a worker whose firm closes has
    no employer. A rule reads the member a value names with `agent.get_referenced('employer')`.

    Unlike the allowed values in `Annotated`, which say what a value may be, it says what a value points to. Choose it
    when a value links a member to a member of another group, such as a customer to its bank.

    It appears in many fields. The employer of a worker, the bank of a customer, the owner of a cow, and the parent of a
    child each name a member of another group.

    Args:
        group:
            The label of the group whose members the value names, such as `'firms'`.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element, Member
        >>> class Worker(Element):
        ...     def __init__(self, label, employer: Annotated[int, Member('firms')] = -1, size=None):
        ...         super().__init__(label, size=size)
        ...         self.employer = employer
        >>> Member('firms').group
        'firms'
    """

    group: str

    def _check(self) -> None:
        """Raise an error for a group that is not a nonempty string.

        Raises:
            ValidationError:
                The group is not a nonempty string.
        """
        if not isinstance(self.group, str) or not self.group:
            error_msg = f"A member names a group by its label, such as Member('firms'), but it is {self.group!r}."
            raise ValidationError(error_msg)
