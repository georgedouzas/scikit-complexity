"""Test the state spec."""

from typing import Annotated, ClassVar

import numpy as np
import pytest
from sage.all__sagemath_symbolics import CC, SR, ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, FieldElement, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import (
    MatrixGroupSpec,
    ParameterSpec,
    RoleSpec,
    StateSpec,
    VisualSpec,
)

from ...conftest import RotorElement

COLOR = MatrixGroupSpec('SU', 3)
HEALTH = FiniteEnumeratedSet(['susceptible', 'infected', 'recovered'])


class _NematicElement(Element):
    """Element with a symmetric traceless tensor state, which keeps the matrix its laws read."""

    seen: ClassVar[list[object]] = []

    def __init__(self, label, Q: Annotated[object, StateSpec(tensor='traceless')] = 0.0):
        super().__init__(label)
        self.Q = Q

    def _get_laws(self, namespace):
        """Return no laws, after keeping the matrix of the state."""
        _NematicElement.seen.append(namespace.Q)
        return []


def _matrix_on(shape):
    """Return the matrix laws read of a symmetric traceless state and the size of the field."""
    _NematicElement.seen.clear()
    field = FieldElement('film', element=_NematicElement, space=LatticeSpace(shape))
    compiler = PassCompiler().prepare(CompositeSystem(elements=[field]))
    compiler.run(None)
    return _NematicElement.seen[-1], compiler.problem_.states


class PersonElement(Element):
    """Person whose health is one of three labels."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        health: Annotated[object, StateSpec(domain=HEALTH)] = 'susceptible',
    ) -> None:
        super().__init__(label, visual=visual)
        self.health = health


class ProgressionInteraction(Interaction):
    """Progression of the health of its target to the next label."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        step: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.step = step

    def _get_laws(self, s):
        """Return the progression of the health."""
        return [UpdateLaw(s.target.health, s.target.health + s.step)]


def test_check_rejects_vector_and_tensor_together():
    """A state that is both a vector and a tensor names both settings."""
    with pytest.raises(ValidationError, match=r'vector=True or tensor=.symmetric'):
        StateSpec(vector=True, tensor='symmetric')._check()


def test_check_rejects_unknown_symmetry():
    """An unknown tensor symmetry raises an error."""
    with pytest.raises(ValidationError, match='tensor symmetry'):
        StateSpec(tensor='skew')._check()


def test_tensor_traceless_stores_two_components_in_two_dimensions():
    """A symmetric traceless state on a 2D lattice stores 2 components per site, and its trace is exactly 0."""
    matrix, states = _matrix_on((32, 32))
    assert states.entries[0].columns == 2
    assert SR(matrix.trace()).is_trivial_zero()


def test_tensor_traceless_stores_five_components_in_three_dimensions():
    """A symmetric traceless state on a 3D lattice stores 5 components per site, with equal off-diagonal entries."""
    matrix, states = _matrix_on((4, 4, 4))
    assert states.entries[0].columns == 5
    assert matrix[1, 0].is_trivially_equal(matrix[0, 1])
    assert SR(matrix.trace()).is_trivial_zero()


def test_algebraic_state_has_order_one_and_no_lags():
    """Test that an algebraic state rejects an order other than 1 and lags, which its equations leave no room for."""
    assert StateSpec(algebraic=True).algebraic
    with pytest.raises(ValidationError, match='An algebraic state has order 1 and no lags'):
        StateSpec(algebraic=True, order=2)._check()
    with pytest.raises(ValidationError, match='An algebraic state has order 1 and no lags'):
        StateSpec(algebraic=True, lags=1)._check()


class _ColorLinkElement(Element):
    """Link with a matrix of SU(3)."""

    def __init__(self, label, visual=None, *, U: Annotated[object, StateSpec(group=COLOR)] = tuple(COLOR.identity)):
        super().__init__(label, visual=visual)
        self.U = U


def test_matrix_group_state_stores_the_real_components_of_every_matrix():
    """Test that an SU(3) state on the 32 links of a 4 by 4 lattice stores 18 components per link, the identity."""
    space = CellComplexSpace(LatticeSpace((4, 4)))
    links = GroupElement('links', element=_ColorLinkElement, space=space, on='edges')
    problem = PassCompiler().prepare(CompositeSystem(elements=[links])).run().problem_
    entry = problem.states.entries[0]
    assert (entry.rows, entry.columns) == (32, 18)
    matrices = COLOR.get_matrices_of(np.reshape(problem.initial_state, (32, 18)))
    assert np.array_equal(matrices, np.broadcast_to(np.eye(3), (32, 3, 3)))


@pytest.mark.parametrize('arguments', [{'vector': True}, {'tensor': 'full'}, {'order': 2}, {'lags': 1}])
def test_matrix_group_state_holds_its_matrices_alone(arguments):
    """Test that a matrix group state that is also a vector, a tensor, or of another order or with lags is refused."""
    with pytest.raises(ValidationError, match='holds its matrices alone'):
        StateSpec(group=COLOR, **arguments)._check()


@pytest.mark.parametrize(
    ('arguments', 'tag'),
    [
        ({}, 'real'),
        ({'domain': ZZ}, 'integer'),
        ({'domain': FiniteEnumeratedSet([0, 1])}, 'boolean'),
        ({'domain': FiniteEnumeratedSet(['S', 'I'])}, 'categorical'),
        ({'domain': CC}, 'complex'),
        ({'vector': True}, 'vector'),
        ({'group': COLOR}, 'matrix group'),
        ({'domain': None}, None),
    ],
)
def test_state_gives_its_domain_tag(arguments, tag):
    """Test that a state spec gives the domain tag of its values."""
    assert StateSpec(**arguments)._get_domain_tag() == tag


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (StateSpec(order=0), 'order'),
        (StateSpec(order=-0.5), 'order'),
        (StateSpec(order=float('nan')), 'order'),
        (StateSpec(order=1.5), 'order'),
        (StateSpec(lags=-1), 'lags'),
        (StateSpec(lags=1.5), 'lags'),
        (StateSpec(order=2, lags=1), 'not both'),
        (StateSpec(order=2, initial=(0.0, 1.0)), '2 initial values were given for 1'),
        (StateSpec(reference='banks', vector=True), 'scalar of order 1'),
        (StateSpec(reference='banks', lags=1), 'scalar of order 1'),
    ],
)
def test_state_spec_rejects_invalid_orders_and_lags(spec, message):
    """Test that an order that is not positive or integral, negative lags, or too many initial values are rejected."""
    with pytest.raises(ValidationError, match=message):
        spec._check()


@pytest.mark.parametrize('spec', [StateSpec(order=0.6), StateSpec(order=2.0), StateSpec(lags=2, initial=(1.0, 2.0))])
def test_state_spec_accepts_fractional_orders_and_lags(spec):
    """Test that a fractional order, an integral order given as a float, and lags with their initial values pass."""
    spec._check()


def test_categorical_values_round_trip_as_labels():
    """Test that categorical states take labels as values and return them as labels."""
    system = CompositeSystem(
        elements=[PersonElement('ann', health='infected')],
        interactions=[ProgressionInteraction('p', target='ann')],
    )
    trajectory = simulate(system, until=1)
    assert trajectory.get_values('ann.health').tolist() == ['infected', 'recovered']


def test_categorical_rejects_unknown_label():
    """Test that an initial value outside the alphabet is rejected."""
    with pytest.raises(ValidationError, match=r"ann\.health = 'dead' is not in its domain"):
        simulate(CompositeSystem(elements=[PersonElement('ann', health='dead')]), until=1)


def test_simulate_does_not_check_values_outside_alphabet():
    """Test that a run leaves a state that a law moves outside its alphabet unchecked."""
    system = CompositeSystem(
        elements=[PersonElement('ann', health='recovered')],
        interactions=[ProgressionInteraction('p', target='ann')],
    )
    unchecked = simulate(system, until=1)
    assert unchecked.states[-1, 0] == 3.0


def test_angle_wraps_into_one_turn():
    """Test that an angle growing past pi lies in (-pi, pi] after every step."""
    system = CompositeSystem(elements=[RotorElement('r', theta=6.0, omega=1.0)])
    values = simulate(system, until=3, dt=0.5).get_values('r.theta')
    assert np.all((values[1:] > -np.pi) & (values[1:] <= np.pi))
    assert values[-1] == pytest.approx(9.0 - 2 * np.pi, abs=1e-6)


def test_angle_swings_around_zero_without_jumping():
    """Test that an angle decreasing through zero stays near zero instead of jumping a full turn."""
    system = CompositeSystem(elements=[RotorElement('r', theta=0.1, omega=-1.0)])
    values = simulate(system, until=1, dt=0.1).get_values('r.theta')
    assert np.all((values > -np.pi) & (values <= np.pi))
    assert values[-1] == pytest.approx(-0.9, abs=1e-6)


def test_state_spec_rejects_an_unknown_role():
    """Test that a role is one of the names analyses read."""
    with pytest.raises(ValidationError, match='The role of a state'):
        StateSpec(role='spin')._check()
