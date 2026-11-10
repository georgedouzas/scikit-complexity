"""Test the `_technology` module of `skcomplex.systems.economics.innovation`."""

import numpy as np

from skcomplex.systems.economics.innovation import (
    ArthurPolakSystem,
    NkTechnologySystem,
)

from ..conftest import _agents, _run


def _truth_table(arity, function):
    """Return the truth table of a Boolean function, bit `r` holding its value on the input bits of row `r`."""
    return sum(int(function(*(bool(row >> pin & 1) for pin in range(arity)))) << row for row in range(2**arity))


def test_nk_technology_costs_only_fall_and_fall_faster_for_modular_designs():
    """Test that the total cost never rises, and that it falls further when redesigns change fewer components."""
    modular = _run(
        NkTechnologySystem(10, 1).prepare().system_,
        2000,
        observables=NkTechnologySystem().get_run_settings()['observables'],
    ).get_values('total_cost')
    coupled = _run(
        NkTechnologySystem(10, 4).prepare().system_,
        2000,
        observables=NkTechnologySystem().get_run_settings()['observables'],
    ).get_values('total_cost')
    for costs in (modular, coupled):
        assert np.all(np.diff(costs) <= 1e-12)
    assert modular[-1] < coupled[-1]


def test_arthur_polak_invents_only_wanted_functions():
    """Test that the pool grows from the primitives and holds only correct circuits of wanted functions."""
    trajectory = _run(ArthurPolakSystem(), 1000)
    pool = _agents(trajectory, 'technologies', 1000)
    wanted = {
        (1, _truth_table(1, lambda a: not a)),
        *(
            (2, _truth_table(2, function))
            for function in (
                lambda a, b: a and b,
                lambda a, b: a or b,
                lambda a, b: a != b,
                lambda a, b: not (a or b),
                lambda a, b: a == b,
                lambda a, b: (not a) or b,
            )
        ),
        *(
            (3, _truth_table(3, function))
            for function in (
                lambda a, b, c: a and b and c,
                lambda a, b, c: a or b or c,
                lambda a, b, c: (a != b) != c,
                lambda a, b, c: (a and b) or (a and c) or (b and c),
            )
        ),
    }
    inventions = set(zip(pool['arity'][3:].astype(int).tolist(), pool['table'][3:].astype(int).tolist(), strict=True))
    assert len(inventions) == len(pool['arity']) - 3 > 0
    assert inventions <= wanted
    assert np.all(np.diff(trajectory.get_values('technologies.count')) >= 0)
