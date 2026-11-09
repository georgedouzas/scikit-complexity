"""Test the `_coalitions` module of `skcomplex.systems.economics.econophysics`."""

from skcomplex.systems.economics.econophysics import ElFarolCoalitionsSystem

from ..conftest import _run


def test_el_farol_coalitions_lower_the_returns_of_all_patrons():
    """Test Table 1 of Collins: attendance near capacity alone, lower returns and wilder swings in coalitions."""
    alone = _run(ElFarolCoalitionsSystem(coalitions=False), 500)
    grouped = _run(ElFarolCoalitionsSystem(), 500)
    attendance, swings = alone.get_values('bar.attendance')[1:], grouped.get_values('bar.attendance')[1:]
    assert 52 < attendance.mean() < 62
    assert alone.get_values('bar.mean_score')[-1] > 0.4
    assert grouped.get_values('bar.mean_score')[100:].mean() < alone.get_values('bar.mean_score')[100:].mean() - 0.1
    assert swings.std() > 2 * attendance.std()
    assert grouped.get_values('bar.coalitions')[-1] < 50
