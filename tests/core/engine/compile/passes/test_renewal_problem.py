"""Test the compilation of renewal laws."""

import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import JumpLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import draw, mean, normal
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System


def _system(label, laws, size=None, **values):
    """Return one element, or a group of a size, whose laws are a function of its namespace."""
    renew = Interaction('renew', roles={'target': label}, laws=lambda s: laws(s.target))
    return System(label, elements=[Element(label, values=values, size=size)], interactions=[renew])


def _counter(label='c', size=None):
    """Return a counter that adds one at every renewal."""
    return _system(label, lambda c: [JumpLaw(waiting=draw('wait', 'expon'), reset={c.n: c.n + 1})], size, n=0.0)


def _renewing(waiting):
    """Return an element with one renewal law of a waiting time, a function of the namespace of its symbols."""
    return _system('r', lambda r: [JumpLaw(waiting=waiting(r))], x=0.0)


def test_compiler_gives_renewal_laws_alone_the_renewal_form():
    """Test that a system of renewal laws alone compiles to the renewal form with a hidden count per member."""
    problem = _counter('counters', 3).prepare()._problem
    assert problem.form == FormKind.RENEWAL
    assert problem.states.names == ['counters.n', 'counters.renew~renewals']
    assert problem.renewals[0].label == 'renew'
    assert problem.renewals[0].members == 3


def test_compiler_partitions_renewals_beside_an_ode():
    """Test that a renewal subsystem and an ODE subsystem in sibling subsystems compile to a partitioned problem."""
    renewing = _counter().set_params(label='renewing')
    decaying = _system('d', lambda d: [RateLaw(d.y, -d.y)], y=1.0).set_params(label='decaying')
    problem = System(systems=[renewing, decaying]).prepare()._problem
    assert problem.form == FormKind.COUPLED
    assert [partition.problem.form for partition in problem.partitions] == [FormKind.RENEWAL, FormKind.ODE]


def test_compiler_keeps_the_form_of_rate_laws_with_renewals():
    """Test that renewal laws beside rate laws of the same component keep the ordinary differential form."""

    def laws(w):
        return [RateLaw(w.x, w.v), JumpLaw(waiting=draw('wait', 'expon'), reset={w.v: -w.v})]

    assert _system('w', laws, x=0.0, v=1.0).prepare()._problem.form == FormKind.ODE


def test_compiler_rejects_an_unknown_distribution():
    """Test that a draw from an unknown distribution names the law, the distribution, and the closest names."""
    typo = _renewing(lambda s: draw('wait', 'paretto', b=1.5))
    with pytest.raises(
        ValidationError,
        match=r"JumpLaw of renew draws 'wait' from the distribution 'paretto'.*pareto",
    ):
        typo.prepare()


def test_compiler_rejects_unknown_keywords():
    """Test that a draw with a keyword its distribution does not take lists the keywords it takes."""
    wrong = _renewing(lambda s: draw('wait', 'pareto', alpha=1.5))
    with pytest.raises(ValidationError, match=r'pareto takes b, loc, scale'):
        wrong.prepare()


@pytest.mark.parametrize(
    'waiting',
    [
        lambda s: normal('wait', 1, 1),
        lambda s: draw('wait', 'norm', loc=1),
        lambda s: draw('wait', 'levy_stable', alpha=1.5, beta=0),
    ],
)
def test_compiler_rejects_waiting_times_that_can_be_negative(waiting):
    """Test that a waiting time from a distribution with negative values asks for nonnegative support."""
    with pytest.raises(
        ValidationError,
        match=r'renewal law renew of renew draws its waiting time.*nonnegative support',
    ):
        _renewing(waiting).prepare()


def test_compiler_rejects_reductions_in_renewal_laws():
    """Test that a renewal law that reads a reduction over its group raises an error."""
    with pytest.raises(ValidationError, match=r'reads a reduction over a group'):
        _system('g', lambda g: [JumpLaw(waiting=draw('wait', 'expon'), reset={g.x: mean(g.x)})], 3, x=0.0).prepare()


def test_compiler_accepts_draws_in_update_laws_from_scipy():
    """Test that a SciPy draw in an update law takes one value per member after the uniform and normal draws."""

    def laws(g):
        return [UpdateLaw(g.x, g.x + draw('jump', 'lomax', c=2.0) + normal('noise'))]

    problem = _system('g', laws, 4, x=0.0).prepare()._problem
    assert problem.draws is not None
    assert problem.draws.size == 8
    assert problem.draws.sampled[0].rows == 4
