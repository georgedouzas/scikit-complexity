"""Describe the problems that validation finds in a system, each with where it is and how to fix it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..base import SkcomplexError, ValidationError

ERROR = 'error'
WARNING = 'warning'


def names_of(terms: Iterable[object]) -> str:
    """Return terms as a sorted, comma-separated list, as issues name them.

    Args:
        terms:
            The terms.

    Returns:
        names:
            The names.
    """
    return ', '.join(sorted(str(term) for term in terms))


@dataclass(frozen=True)
class Issue:
    """Problem that validation found in a system, with where it is and how to fix it.

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
    severity: str = ERROR
    error: type[SkcomplexError] = field(default=ValidationError, repr=False, compare=False)

    def __str__(self) -> str:
        """Return the issue as one line with its path, code, message, and fix."""
        return f'{self.path}: {self.message} {self.fix} [{self.code}]'


class ValidationWarning(UserWarning):
    """Warning of a problem that does not stop compilation."""
