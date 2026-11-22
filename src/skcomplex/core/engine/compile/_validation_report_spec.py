"""Define the spec of the report of every problem a compiler found."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from dataclasses import dataclass

from ...base import BaseSpec
from ...enums import SeverityKind
from ...errors import ValidationError, ValidationWarning
from ._issue_spec import IssueSpec


@dataclass(frozen=True)
class ValidationReportSpec(BaseSpec):
    """A spec of the report of every problem a compiler found in a system.

    `prepare` checks the system, its solver, and its backend, and gathers every problem it finds in this report, such as
    a law that reads a value its element does not hold. Its check shows the warnings and raises one error that lists all
    the errors, so a model of an ecosystem, a circuit, or a market shows every mistake at once and a valid report
    passes.

    Read more in the user guide, under Troubleshooting.

    Args:
        issues:
            The issues, in the order they were found.

    See Also:
        IssueSpec: One problem of the report.

    Examples:
        >>> from skcomplex.core.engine.compile import IssueSpec, ValidationReportSpec
        >>> report = ValidationReportSpec((IssueSpec('empty', 'system', 'It has no element.', 'Add one.'),))
        >>> ValidationReportSpec().ok, report.ok
        (True, False)
    """

    issues: tuple[IssueSpec, ...] = ()

    @property
    def errors(self) -> list[IssueSpec]:
        """Return the issues that stop compilation."""
        return [issue for issue in self.issues if issue.severity == SeverityKind.ERROR]

    @property
    def warnings(self) -> list[IssueSpec]:
        """Return the issues that do not stop compilation."""
        return [issue for issue in self.issues if issue.severity == SeverityKind.WARNING]

    @property
    def ok(self) -> bool:
        """Check whether the report holds no error."""
        return not self.errors

    def _check(self) -> None:
        """Emit the warnings of the report and raise one error that lists its errors.

        Raises:
            ValidationError:
                The report holds an error, of the class of its errors when they share one.
        """
        for issue in self.warnings:
            warnings.warn(issue.text, ValidationWarning, stacklevel=3)
        if self.errors:
            lines = '\n'.join(f'- {issue.text}' for issue in self.errors)
            error_msg = f'The system is not valid, with {len(self.errors)} error(s):\n{lines}'
            classes = {issue.error for issue in self.errors}
            raise (classes.pop() if len(classes) == 1 else ValidationError)(error_msg)
