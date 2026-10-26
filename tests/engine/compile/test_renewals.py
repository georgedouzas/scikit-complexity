"""Test the compilation of renewal laws."""

import pytest

from skcomplex.core.base import Form, ValidationError
from skcomplex.core.components import Element
from skcomplex.core.laws import JumpLaw, RateLaw, UpdateLaw, draw, mean, normal
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system

CounterElement = Element.define(
    'CounterElement',
    states={'n': 0.0},
    laws=lambda s: [JumpLaw(waiting=draw('wait', 'expon'), reset={s.n: s.n + 1})],
)
DecayElement = Element.define('DecayElement', states={'y': 1.0}, laws=lambda s: [RateLaw(s.y, -s.y)])


def _renewal(waiting, reset=None):
    """Return an element type with one renewal law of a waiting time and a reset of its state `x`."""
    return Element.define(
        'RenewingElement',
        states={'x': 0.0},
        parameters={'a': 1.0},
        laws=lambda s: [JumpLaw(waiting=waiting(s), reset=reset(s) if reset else None)],
    )


def test_compile_system_gives_renewal_laws_alone_the_renewal_form():
    """Test that a system of renewal laws alone compiles to the renewal form with a hidden count per member."""
    problem = compile_system(System(components=[CounterElement.group('counters', 3)]))
    assert problem.form == Form.RENEWAL
    assert problem.states.names == ['counters.n', 'counters.counters~renewals']
    assert problem.renewals[0].label == 'counters'
    assert problem.renewals[0].members == 3


def test_compile_system_partitions_renewals_beside_an_ode():
    """Test that a renewal subsystem and an ODE subsystem in sibling subsystems compile to a partitioned problem."""
    system = System(
        components=[
            System('renewing', components=[CounterElement('c')]),
            System('decaying', components=[DecayElement('d')]),
        ],
    )
    problem = compile_system(system)
    assert problem.form == Form.COUPLED
    assert [partition.problem.form for partition in problem.partitions] == [Form.RENEWAL, Form.ODE]


def test_compile_system_keeps_the_form_of_rate_laws_with_renewals():
    """Test that renewal laws beside rate laws of the same component keep the ordinary differential form."""
    walker = Element.define(
        'WalkerElement',
        states={'x': 0.0, 'v': 1.0},
        laws=lambda s: [RateLaw(s.x, s.v), JumpLaw(waiting=draw('wait', 'expon'), reset={s.v: -s.v})],
    )
    assert compile_system(System(components=[walker('w')])).form == Form.ODE


def test_compile_system_rejects_an_unknown_distribution():
    """Test that a draw from an unknown distribution names the law, the distribution, and the closest names."""
    typo = _renewal(lambda s: draw('wait', 'paretto', b=1.5))
    with pytest.raises(ValidationError, match=r"RenewalLaw of r draws 'wait' from the distribution 'paretto'.*pareto"):
        compile_system(System(components=[typo('r')]))


def test_compile_system_rejects_unknown_keywords():
    """Test that a draw with a keyword its distribution does not take lists the keywords it takes."""
    wrong = _renewal(lambda s: draw('wait', 'pareto', alpha=1.5))
    with pytest.raises(ValidationError, match=r'pareto takes b, loc, scale'):
        compile_system(System(components=[wrong('r')]))


@pytest.mark.parametrize(
    'waiting',
    [
        lambda s: normal('wait', 1, 1),
        lambda s: draw('wait', 'norm', loc=1),
        lambda s: draw('wait', 'levy_stable', alpha=1.5, beta=0),
    ],
)
def test_compile_system_rejects_waiting_times_that_can_be_negative(waiting):
    """Test that a waiting time from a distribution with negative values asks for nonnegative support."""
    with pytest.raises(ValidationError, match=r'renewal law r of r draws its waiting time.*nonnegative support'):
        compile_system(System(components=[_renewal(waiting)('r')]))


def test_compile_system_rejects_reductions_in_renewal_laws():
    """Test that a renewal law that reads a reduction over its group raises an error."""
    reading = _renewal(lambda s: draw('wait', 'expon'), lambda s: {s.x: mean(s.x)})
    with pytest.raises(ValidationError, match=r'reads a reduction over a group'):
        compile_system(System(components=[reading.group('g', 3)]))


def test_compile_system_accepts_draws_in_update_laws_from_scipy():
    """Test that a SciPy draw in an update law takes one value per member after the uniform and normal draws."""
    jumping = Element.define(
        'JumpingElement',
        states={'x': 0.0},
        laws=lambda s: [UpdateLaw(s.x, s.x + draw('jump', 'lomax', c=2.0) + normal('noise'))],
    )
    problem = compile_system(System(components=[jumping.group('g', 4)]))
    assert problem.draws is not None
    assert problem.draws.size == 8
    assert problem.draws.sampled[0].rows == 4
