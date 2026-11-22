"""Define the group whose members a value of an element names."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec
from ...errors import ValidationError


@dataclass(frozen=True)
class Member(BaseSpec):
    """A spec of a value that names a member of another group, such as a parent.

    It is written in `Annotated` beside the allowed values, such as `employer: Annotated[int, Member('firms')] = -1`.
    The value is the index of a member of that group, and -1 names none. When a member of the group is removed, every
    value that names it becomes -1, so a worker whose firm closes has no employer. A rule reads the member a value names
    with `agent.get_referenced('employer')`. It links a worker to a firm, a customer to a bank, and a child to a parent.

    Read more in the user guide, under Elements.

    Args:
        group:
            The label of the group whose members the value names, such as `'firms'`.

    See Also:
        Quantity: Marks a value whose total only flows change.
        Period: Marks a value that wraps around, such as an angle.

    Examples:
        >>> from skcomplex.core.components.elements import Member
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
