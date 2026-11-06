"""Define the validation report spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import warnings
from dataclasses import dataclass

from ..errors import ValidationError, ValidationWarning
from ._base_spec import BaseSpec
from ._issue_spec import _ERROR, _WARNING, IssueSpec


@dataclass(frozen=True)
class ValidationReportSpec(BaseSpec):
    """Every issue that a compiler found in a system, its solver, and its backend.

    `prepare` emits the warnings of the report and raises one error that lists its errors, so a valid report passes
    its check.

    Args:
        issues:
            The issues, in the order they were found.

    Examples:
        >>> from skcomplex.core.engine.compile import PassCompiler
        >>> from skcomplex.core.components.system import System
        >>> PassCompiler().prepare(System()).report_.ok
        True
    """

    issues: tuple[IssueSpec, ...] = ()

    @property
    def errors(self) -> list[IssueSpec]:
        """Return the issues that stop compilation."""
        return [issue for issue in self.issues if issue.severity == _ERROR]

    @property
    def warnings(self) -> list[IssueSpec]:
        """Return the issues that do not stop compilation."""
        return [issue for issue in self.issues if issue.severity == _WARNING]

    @property
    def ok(self) -> bool:
        """Check whether the report holds no error."""
        return not self.errors

    def _check(self) -> None:
        """Emit the warnings of the report and raise one error that lists every error of the report.

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
