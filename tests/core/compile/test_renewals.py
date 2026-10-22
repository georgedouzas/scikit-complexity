"""Test the compilation of renewal laws and fractional noise laws."""

import numpy as np
import pytest

from skcomplex.core.base import Form, ValidationError
from skcomplex.core.compile import compile_system
from skcomplex.core.components import Element
from skcomplex.core.laws import FractionalNoise, Jump, Noise, Rate, Update, draw, mean, normal
from skcomplex.core.system import System

Counter = Element.define(
    'Counter',
    states={'n': 0.0},
    laws=lambda s: [Jump(waiting=draw('wait', 'expon'), reset={s.n: s.n + 1})],
)
Decay = Element.define('Decay', states={'y': 1.0}, laws=lambda s: [Rate(s.y, -s.y)])


def _renewal(waiting, reset=None):
    """Return an element type with one renewal law of a waiting time and a reset of its state `x`."""
    return Element.define(
        'Renewing',
        states={'x': 0.0},
        parameters={'a': 1.0},
        laws=lambda s: [Jump(waiting=waiting(s), reset=reset(s) if reset else None)],
    )


def _fractional(amplitude, hurst):
    """Return an element type driven by one fractional noise law."""
    return Element.define(
        'Fractional',
        states={'x': 0.0},
        parameters={'D': 1.0},
        laws=lambda s: [FractionalNoise(s.x, amplitude(s), hurst=hurst)],
    )


def test_compile_system_gives_renewal_laws_alone_the_renewal_form():
    """Test that a system of renewal laws alone compiles to the renewal form with a hidden count per member."""
    problem = compile_system(System(components=[Counter.group('counters', 3)]))
    assert problem.form is Form.RENEWAL
    assert problem.states.names == ['counters.n', 'counters.counters~renewals']
    assert problem.renewals[0].label == 'counters'
    assert problem.renewals[0].members == 3


def test_compile_system_partitions_renewals_beside_an_ode():
    """Test that a renewal subsystem and an ODE subsystem in sibling subsystems compile to a partitioned problem."""
    system = System(
        components=[System('renewing', components=[Counter('c')]), System('decaying', components=[Decay('d')])],
    )
    problem = compile_system(system)
    assert problem.form is Form.COUPLED
    assert [partition.problem.form for partition in problem.partitions] == [Form.RENEWAL, Form.ODE]


def test_compile_system_keeps_the_form_of_rate_laws_with_renewals():
    """Test that renewal laws beside rate laws of the same component keep the ordinary differential form."""
    walker = Element.define(
        'Walker',
        states={'x': 0.0, 'v': 1.0},
        laws=lambda s: [Rate(s.x, s.v), Jump(waiting=draw('wait', 'expon'), reset={s.v: -s.v})],
    )
    assert compile_system(System(components=[walker('w')])).form is Form.ODE


def test_compile_system_rejects_an_unknown_distribution():
    """Test that a draw from an unknown distribution names the law, the distribution, and the closest names."""
    typo = _renewal(lambda s: draw('wait', 'paretto', b=1.5))
    with pytest.raises(ValidationError, match=r"Renewal law of r draws 'wait' from the distribution 'paretto'.*pareto"):
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
        'Jumping',
        states={'x': 0.0},
        laws=lambda s: [Update(s.x, s.x + draw('jump', 'lomax', c=2.0) + normal('noise'))],
    )
    problem = compile_system(System(components=[jumping.group('g', 4)]))
    assert problem.draws is not None
    assert problem.draws.size == 8
    assert problem.draws.sampled[0].rows == 4


def test_compile_system_rejects_a_hurst_index_out_of_range():
    """Test that a Hurst index of 1 names the law and the value and states the open interval."""
    rough = _fractional(lambda s: 1, 1.0)
    with pytest.raises(ValidationError, match=r'fractional noise law of f.x of f has the Hurst index 1.*\(0, 1\)'):
        compile_system(System(components=[rough('f')]))


def test_compile_system_rejects_a_state_dependent_fractional_amplitude():
    """Test that a fractional amplitude that reads its own state names the state and the supported case."""
    multiplicative = _fractional(lambda s: s.x, 0.7)
    with pytest.raises(ValidationError, match=r'depends on the state f.x. Only state-independent amplitudes'):
        compile_system(System(components=[multiplicative('f')]))


def test_compile_system_lowers_a_hurst_index_of_one_half_to_noise():
    """Test that fractional noise of Hurst index 1/2 compiles to an ordinary noise row with no fractional rows."""
    problem = compile_system(System(components=[_fractional(lambda s: s.x, 0.5).group('f', 3)]))
    plain = compile_system(
        System(components=[Element.define('Plain', states={'x': 0.0}, laws=lambda s: [Noise(s.x, s.x)]).group('f', 3)]),
    )
    assert problem.form is Form.SDE
    assert problem.fractional == ()
    assert np.array_equal(problem.noise_rows, plain.noise_rows)


def test_compile_system_records_the_hurst_index_of_fractional_rows():
    """Test that fractional rows of a group record the Hurst index of every row from the parameters."""
    problem = compile_system(System(components=[_fractional(lambda s: (2 * s.D) ** 0.5, 0.3).group('f', 4)]))
    rows = problem.fractional[0]
    assert rows.positions.tolist() == [0, 1, 2, 3]
    assert np.allclose(rows.hurst(problem.initial_parameters), 0.3)
