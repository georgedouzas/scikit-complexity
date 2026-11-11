"""Define the issue spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ..enums import SeverityKind
from ..errors import SkcomplexError, ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class IssueSpec(BaseSpec):
    """A spec of a problem that validation found, with where it is and the fix.

    Args:
        code:
            The stable code of the problem, such as `'law-domain'`.

        path:
            The slash path of the component or system the problem is in.

        message:
            What is wrong and why.

        fix:
            The change that makes the system valid.

        severity:
            `'error'` for problems that stop compilation, `'warning'` otherwise.

        error:
            The class of the error raised for the problem: `ValidationError` or one of its subclasses for the problems
            validation finds, and the class of the original error for a component that fails to declare.
    """

    code: str
    path: str
    message: str
    fix: str
    severity: str = SeverityKind.ERROR
    error: type[SkcomplexError] = field(default=ValidationError, repr=False, compare=False)

    @property
    def text(self) -> str:
        """IssueSpec as one line.

        The line holds its path, code, message, and fix.
        """
        return f'{self.path}: {self.message} {self.fix} [{self.code}]'
