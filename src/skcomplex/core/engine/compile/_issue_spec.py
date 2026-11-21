"""Define the spec of one problem that checking a system found."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ...base import BaseSpec
from ...enums import SeverityKind
from ...errors import SkcomplexError, ValidationError


@dataclass(frozen=True)
class IssueSpec(BaseSpec):
    """A spec of one problem found in a system, with where it is and how to fix it.

    When `prepare` checks a system, it collects every problem it finds as an issue, such as a law that reads a value its
    element does not hold, with a short code, the path of the component, what is wrong, and how to fix it. The issues go
    into one report, so a model of an ecosystem, a circuit, or a market shows all its mistakes at once.

    Read more in the user guide, under Troubleshooting.

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

    See Also:
        ValidationReportSpec: Holds every issue of a system.

    Examples:
        >>> from skcomplex.core.engine.compile import IssueSpec
        >>> issue = IssueSpec('empty', 'system', 'It has no element.', 'Add one.')
        >>> issue.severity.value
        'error'
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
