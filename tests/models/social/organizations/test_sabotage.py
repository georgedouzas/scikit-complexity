"""Test the `_sabotage` module of `skcomplex.models.social.organizations`."""

from skcomplex.models.social.organizations import (
    insider_sabotage,
)

from ..conftest import _run, _values


def test_insider_risk_falls_with_turnover():
    """Test that a higher turnover keeps the mean risk of sabotage lower after one year."""
    risks = [
        _values(_run(insider_sabotage(turnover=turnover, random_state=0), 365), 'employees', 'risk').mean()
        for turnover in (0.1, 3.0)
    ]
    assert risks[1] < risks[0]
