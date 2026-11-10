"""Define the validation report spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from dataclasses import dataclass

from ..errors import ValidationError, ValidationWarning
from ._base_spec import BaseSpec
from ._issue_spec import ERROR, WARNING, IssueSpec


@dataclass(frozen=True)
class ValidationReportSpec(BaseSpec):
    """A spec of every issue a compiler found in a system, solver, and backend.

    Its check emits the warnings of the report and raises one error that lists its errors, so a valid report passes
    its check.

    Args:
        issues:
            The issues, in the order they were found.

    Examples:
        >>> from skcomplex.core.specs import IssueSpec, ValidationReportSpec
        >>> ValidationReportSpec().ok
        True
        >>> ValidationReportSpec((IssueSpec('empty', 'system', 'It has no element.', 'Add one.'),)).ok
        False
    """

    issues: tuple[IssueSpec, ...] = ()

    @property
    def errors(self) -> list[IssueSpec]:
        """Return the issues that stop compilation."""
        return [issue for issue in self.issues if issue.severity == ERROR]

    @property
    def warnings(self) -> list[IssueSpec]:
        """Return the issues that do not stop compilation."""
        return [issue for issue in self.issues if issue.severity == WARNING]

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
