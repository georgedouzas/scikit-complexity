"""Test the `_validation_report_spec` module of `skcomplex.core.specs`."""

import pytest

from skcomplex.core.errors import ValidationError, ValidationWarning
from skcomplex.core.specs import (
    IssueSpec,
    ValidationReportSpec,
)


def test_report_raises_every_error():
    """Test that checking a report raises every error with its path and code, and emits its warnings."""
    report = ValidationReportSpec(
        (
            IssueSpec('role', 'growth', 'The role is unbound.', 'Bind it.'),
            IssueSpec('law-domain', 'drift', 'The law targets a categorical state.', 'Use a jump law.'),
            IssueSpec('note', 'system', 'Something to know.', 'Nothing to do.', severity='warning'),
        ),
    )
    with pytest.warns(ValidationWarning, match='Something to know'), pytest.raises(ValidationError) as error:
        report._check()
    assert '2 error(s)' in str(error.value)
    assert 'growth: The role is unbound. Bind it. [role]' in str(error.value)
    assert '[law-domain]' in str(error.value)
