"""Test element groups, pair interactions, and boundary laws in the model."""

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.base import RoleError, ValidationError, VisualSpec
from skcomplex.core.compile import compile_system
from skcomplex.core.model import (
    Boundary,
    Element,
    EuclideanSpace,
    Group,
    Interaction,
    LatticeSpace,
    PairInteraction,
    Particle,
    Rate,
    System,
)
from skcomplex.models.biology.ecology import ExponentialGrowth, Population, Predation


class Attraction(PairInteraction):
    """Pair interaction that returns a boundary law, which pair interactions do not take."""

    def laws(self, s):
        """Return a boundary law."""
        return [Boundary(s.first.v, s.first.v)]


def test_group_stores_arguments():
    """Test that a group stores its label, size, and values unmodified."""
    positions = np.zeros((1000, 2))
    group = Particle.group('gas', 1000, x=positions)
    assert group.label == 'gas'
    assert group.size == 1000
    assert group.values['x'] is positions
    assert clone(group).size == 1000


def test_group_position_state_has_member_shape():
    """Test that a group's vector state has one row per member."""
    system = System(components=[Particle.group('gas', 1000, x=np.zeros((1000, 2)))], space=EuclideanSpace(2))
    assert compile_system(system).states.entry('gas.x').shape == (1000, 2)


def test_group_wrong_number_of_values_raises_error():
    """Test that a group with the wrong number of values names the group, variable, and sizes."""
    system = System(components=[Particle.group('gas', 1000, x=np.zeros((999, 2)))], space=EuclideanSpace(2))
    with pytest.raises(ValidationError, match=r'gas\.x .*999 values .* 1000 members'):
        compile_system(system)


@pytest.mark.parametrize(('size', 'match'), [(0, 'positive integer'), (2.5, 'positive integer')])
def test_group_invalid_size_raises_error(size, match):
    """Test that a group size must be a positive integer."""
    with pytest.raises(ValidationError, match=match):
        compile_system(System(components=[Population.group('herds', size)]))


def test_group_unknown_value_raises_error():
    """Test that a group can only set variables of its element type."""
    with pytest.raises(ValidationError, match='does not declare'):
        compile_system(System(components=[Population.group('herds', 3, mass=1.0)]))


def test_group_non_element_type_raises_error():
    """Test that a group's element must be an Element subclass."""
    with pytest.raises(ValidationError, match='not an Element subclass'):
        compile_system(System(components=[Group('herds', element=int, size=3)]))


def test_role_binds_to_group_and_applies_to_every_member():
    """Test that an interaction bound to a group applies its law to every member."""
    system = System(
        components=[Population.group('herds', 3, n=[1.0, 2.0, 3.0])],
        interactions=[ExponentialGrowth('growth', target='herds', r=2.0)],
    )
    problem = compile_system(system)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert drift.tolist() == [2.0, 4.0, 6.0]


def test_single_element_bound_with_group_receives_sum():
    """Test that a single element bound next to a group receives the sum over the members."""
    system = System(
        components=[Population.group('rabbits', 3, n=[1.0, 2.0, 3.0]), Population('fox', n=2.0)],
        interactions=[Predation('hunt', predator='fox', prey='rabbits', a=1.0, b=1.0)],
    )
    problem = compile_system(system)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert drift.tolist() == [-2.0, -4.0, -6.0, 12.0]
    assert problem.has_group_sums
    assert 'fox.n' not in problem.equations


Box = Element.define('Box', states={'n': 0.0})
Flow = Interaction.define('Flow', roles=['source', 'sink'], laws=lambda s: [Rate(s.sink.n, s.source.n)])
Cell = Element.define('Cell', states={'size': 0.0})


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_interaction_reads_two_groups_member_by_member(backend):
    """Test that an interaction bound to two groups of equal size reads row `i` of both together."""
    system = System(
        components=[Box.group('source', 5, n=np.arange(5.0)), Box.group('sink', 5)],
        interactions=[Flow('flow', source='source', sink='sink')],
    )
    problem = compile_system(system, backend=backend)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.backend == backend
    assert problem.states.read(drift, 'sink.n').tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_interaction_reading_groups_of_different_sizes_raises_error():
    """Test that an interaction reading groups of different sizes member by member names both groups and sizes."""
    system = System(
        components=[Box.group('source', 5), Box.group('sink', 7)],
        interactions=[Flow('flow', source='source', sink='sink')],
    )
    with pytest.raises(RoleError, match=r'sink of 7 members, source of 5 members.*PairInteraction'):
        compile_system(system)


def test_group_takes_state_named_size():
    """Test that a state named size takes per-member values, since the group size is given by position."""
    group = Cell.group('cells', 3, size=[1.0, 2.0, 3.0])
    problem = compile_system(System(components=[group]))
    assert group.size == 3
    assert problem.states.read(problem.initial_state, 'cells.size').tolist() == [1.0, 2.0, 3.0]


def test_group_label_and_size_are_positional_only():
    """Test that the label and the size of a group cannot be given by keyword."""
    with pytest.raises(TypeError, match='positional argument'):
        Box.group(label='boxes', size=3)


def test_pair_interaction_rejects_other_law_kinds():
    """Test that pair interactions take rate laws only."""
    system = System(
        components=[Particle.group('gas', 3)],
        interactions=[Attraction('pull', first='gas', second='gas')],
        space=EuclideanSpace(2),
    )
    with pytest.raises(ValidationError, match='laws other than Rate'):
        compile_system(system)


def test_pair_interaction_needs_vector_position():
    """Test that pair interactions need a vector position state."""

    class Contact(PairInteraction):
        """Pair interaction between populations, which have no position."""

        def laws(self, s):
            """Return a rate law."""
            return [Rate(s.first.n, s.second.n)]

    system = System(
        components=[Population.group('herds', 3)],
        interactions=[Contact('contact', first='herds', second='herds')],
    )
    with pytest.raises(ValidationError, match="vector state 'x'"):
        compile_system(system)


def test_lattice_keeps_visual_of_element_type():
    """Test that a lattice group of a type that declares a visual takes that visual, not an image."""
    Column = Element.define('Column', states={'h': 0.0}, visual=VisualSpec('heights'))
    assert Column.lattice('columns', LatticeSpace((3,))).visual.kind == 'heights'


def test_lattice_without_type_visual_is_an_image():
    """Test that a lattice group of a type that declares no visual is drawn as an image."""
    Cell = Element.define('Cell', states={'h': 0.0})
    assert Cell.lattice('cells', LatticeSpace((3,))).visual.kind == 'image'
