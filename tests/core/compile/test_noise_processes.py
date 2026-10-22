"""Test named and correlated Wiener processes: their lowering, their drift corrections, and their validation."""

import numpy as np
import pytest

from skcomplex.core.base import Domain, Form, NoiseKind, ValidationError, role, state, symbol
from skcomplex.core.compile import compile_system, validate
from skcomplex.core.components import Element, Wiener
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Noise, Rate, Update
from skcomplex.core.solvers import SOLVERS, EulerMaruyama, Heun, Milstein
from skcomplex.core.space import DiscreteTime
from skcomplex.core.system import System

Stock = Element.define('Stock', states={'n': 0.0})


def _pair(process='common', conventions=('ito', 'ito')):
    """Return an element whose two states take unit noise on one local process."""
    first, second = conventions
    return Element.define(
        'Pair',
        states={'x': 1.0, 'y': 1.0},
        parameters={'sigma': 0.5},
        laws=lambda s: [
            Noise(s.x, s.sigma * s.y, convention=first, process=process),
            Noise(s.y, s.sigma * s.x, convention=second, process=process),
        ],
    )


class Drive(Interaction):
    """Unit noise on two elements from the two processes of a Wiener element."""

    first: str = role()
    second: str = role()
    noise: str = role()

    def laws(self, s):
        """Return the noise of both elements."""
        return [Noise(s.first.n, 1.0, process=s.noise.dW[0]), Noise(s.second.n, 1.0, process=s.noise.dW[1])]


class Misuse(Interaction):
    """Rate law that uses the handle of a Wiener process as a value."""

    target: str = role()
    noise: str = role()

    def laws(self, s):
        """Return the rate."""
        return [Rate(s.target.n, s.noise.dW)]


class Coupled(Interaction):
    """Stratonovich noise on a state, through a Wiener process another component also draws."""

    target: str = role()
    other: str = role()
    noise: str = role()

    def laws(self, s):
        """Return the noise of the target, whose amplitude reads the other state."""
        return [Noise(s.target.n, s.other.n, convention='stratonovich', process=s.noise.dW)]


class Single(Interaction):
    """Itô noise on one element from a Wiener process."""

    target: str = role()
    noise: str = role()

    def laws(self, s):
        """Return the noise."""
        return [Noise(s.target.n, 1.0, process=s.noise.dW)]


def _correlated(covariance):
    """Return two stocks driven by a Wiener element of a covariance."""
    return System(
        components=[Stock('a'), Stock('b'), Wiener('market', dimension=2, covariance=covariance)],
        interactions=[Drive('drive', first='a', second='b', noise='market')],
    )


def test_noise_rejects_unknown_process():
    """Test that a noise law rejects a process that is neither a label nor a handle."""
    with pytest.raises(ValidationError, match='process'):
        Noise(symbol('n'), 1.0, process=3)


def test_compile_system_without_processes_keeps_identity_columns():
    """Test that noise laws that name no process get one process per amplitude and no correlated block."""
    Walker = Element.define('Walker', states={'x': 0.0}, laws=lambda s: [Noise(s.x, 1.0), Noise(s.x, 2.0)])
    problem = compile_system(System(components=[Walker.group('walkers', 3)]))
    np.testing.assert_array_equal(problem.noise_columns, np.arange(6))
    assert problem.n_processes == 6
    assert problem.noise_blocks == ()


def test_compile_system_shares_local_process():
    """Test that two laws of one element on one local process share one column."""
    problem = compile_system(System(components=[_pair()('pair')]))
    np.testing.assert_array_equal(problem.noise_columns, [0, 0])
    assert problem.n_processes == 1


def test_compile_system_gives_members_their_own_local_process():
    """Test that a local process of a group law gives every member its own process."""
    problem = compile_system(System(components=[_pair().group('pairs', 3)]))
    np.testing.assert_array_equal(problem.noise_columns, [0, 1, 2, 0, 1, 2])


def test_compile_system_factors_covariance():
    """Test that a Wiener element with a covariance lowers to one block whose factor reproduces it."""
    problem = compile_system(_correlated([[1.0, 0.6], [0.6, 2.0]]))
    (block,) = problem.noise_blocks
    assert (block.start, block.size) == (0, 2)
    np.testing.assert_allclose(block.factor @ block.factor.T, [[1.0, 0.6], [0.6, 2.0]])


def test_compile_system_factors_singular_covariance():
    """Test that perfectly correlated processes compile with an eigenvalue factor."""
    (block,) = compile_system(_correlated([[1.0, 1.0], [1.0, 1.0]])).noise_blocks
    np.testing.assert_allclose(block.factor @ block.factor.T, np.ones((2, 2)), atol=1e-12)


def test_stratonovich_correction_covers_cross_terms():
    """Test that a shared Stratonovich process adds the cross terms `C G_kl dG_ij/dx_k` to the drift."""
    problem = compile_system(System(components=[_pair(conventions=('stratonovich', 'stratonovich'))('pair')]))
    drift = problem.drift(0.0, np.asarray([2.0, 3.0]), problem.initial_parameters)
    np.testing.assert_allclose(drift, [0.5 * 0.5 * 0.5 * 2.0, 0.5 * 0.5 * 0.5 * 3.0])


def test_stratonovich_correction_of_several_laws_adds_up():
    """Test that two Stratonovich laws with independent processes on one state compile and add their corrections."""
    Twice = Element.define(
        'Twice',
        states={'x': 1.0},
        laws=lambda s: [Noise(s.x, s.x, convention='stratonovich'), Noise(s.x, 2 * s.x, convention='stratonovich')],
    )
    problem = compile_system(System(components=[Twice('twice')]))
    drift = problem.drift(0.0, np.asarray([2.0]), problem.initial_parameters)
    assert drift[0] == pytest.approx(0.5 * 2.0 + 0.5 * 4 * 2.0)


def test_validate_reports_mixed_conventions():
    """Test that laws of different conventions on one process are reported with both laws and the process."""
    issues = validate(System(components=[_pair(conventions=('ito', 'stratonovich'))('pair')])).issues
    (issue,) = [issue for issue in issues if issue.code == 'noise-convention-mixed']
    assert 'ito Noise law of pair' in issue.message
    assert 'stratonovich Noise law of pair' in issue.message
    assert "'common'" in issue.message
    assert 'one convention' in issue.fix


def test_validate_reports_correction_out_of_scope():
    """Test that a Stratonovich law reading a state another component drives on its process is reported."""
    system = System(
        components=[Stock('a'), Stock('b', n=1.0), Wiener('market')],
        interactions=[
            Coupled('coupled', target='a', other='b', noise='market'),
            Single('single', target='b', noise='market'),
        ],
    )
    (issue,) = [issue for issue in validate(system).issues if issue.code == 'noise-correction-scope']
    assert 'Noise law of coupled' in issue.message
    assert 'Noise law of single' in issue.message
    assert 'Itô' in issue.fix


def test_validate_reports_invalid_covariance():
    """Test that a covariance with a negative eigenvalue is reported with the element and the eigenvalue."""
    (issue,) = validate(_correlated([[1.0, 2.0], [2.0, 1.0]])).issues
    assert issue.code == 'wiener-covariance'
    assert issue.path == 'market'
    assert '-1' in issue.message


@pytest.mark.parametrize(
    'covariance',
    [[[1.0, 0.0, 0.0]], [[1.0, 0.5], [0.0, 1.0]]],
)
def test_validate_reports_covariance_shape_and_symmetry(covariance):
    """Test that a covariance that is not square of the dimension or not symmetric is reported."""
    assert [issue.code for issue in validate(_correlated(covariance)).issues] == ['wiener-covariance']


def test_validate_reports_handle_as_value():
    """Test that a Wiener handle used inside a rate law is reported with the law."""
    system = System(
        components=[Stock('a'), Wiener('market')],
        interactions=[Misuse('misuse', target='a', noise='market')],
    )
    (issue,) = [issue for issue in validate(system).issues if issue.code == 'wiener-handle']
    assert 'Rate law of misuse' in issue.message
    assert 'only the process of a noise law' in issue.message


def test_validate_reports_wiener_under_discrete_time():
    """Test that a Wiener element in a system of discrete time is reported with the fix of continuous time."""
    Counter = Element.define('Counter', states={'k': state(0.0, domain=Domain.REAL)}, laws=lambda s: [Update(s.k, 1)])
    system = System(components=[Counter('counter'), Wiener('market')], time=DiscreteTime())
    (issue,) = [issue for issue in validate(system).issues if issue.code == 'wiener-time']
    assert issue.path == 'market'
    assert 'continuous time' in issue.fix


def test_validate_reports_milstein_with_shared_process():
    """Test that Milstein with a shared process gives one error that names the law and suggests other solvers."""
    (issue,) = validate(System(components=[_pair()('pair')]), solver='milstein').issues
    assert issue.code == 'noise-unsupported'
    assert 'Noise law of pair' in issue.message
    assert 'euler-maruyama' in issue.fix
    assert 'heun' in issue.fix


def test_validate_reports_lammps_with_shared_process():
    """Test that the LAMMPS backend with a shared process is reported with the backends that support it."""
    issues = validate(System(components=[_pair()('pair')]), solver='lammps').issues
    (issue,) = [issue for issue in issues if issue.code == 'noise-unsupported']
    assert 'Noise law of pair' in issue.message
    assert 'numba, numpy' in issue.fix


def test_validate_accepts_correlated_noise_with_automatic_solver():
    """Test that the automatic SDE solver declares correlated noise, so validation accepts it."""
    assert validate(_correlated([[1.0, 0.5], [0.5, 1.0]])).ok
    assert NoiseKind.CORRELATED in SOLVERS.resolve((Form.SDE, 'auto')).supports.noise


def test_declared_noise_support():
    """Test that Euler-Maruyama and Heun declare every noise structure and Milstein only diagonal noise."""
    assert set(EulerMaruyama.supports.noise) == set(NoiseKind)
    assert set(Heun.supports.noise) == set(NoiseKind)
    assert set(Milstein.supports.noise) == {NoiseKind.DIAGONAL}
