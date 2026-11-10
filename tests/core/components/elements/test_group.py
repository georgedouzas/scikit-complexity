"""Test the `_group` module of `skcomplex.core.components.elements`."""

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest
from sage.all__sagemath_symbolics import var
from sklearn.base import clone

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import BaseSpace, ContinuousSpace, LatticeSpace, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.errors import SpaceError, ValidationError
from skcomplex.core.specs import LayerSpec, StateSpec, VisualSpec


class BallElement(Element):
    """Point of a continuous space, with a vector state `x` of order 2."""

    def __init__(
        self,
        label: str,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, order=2, role='position')] = 0.0,
        x_d1: npt.ArrayLike = 0.0,
        visual: VisualSpec | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `default_visual_`, points at the values of `x`."""
        super()._prepare(context)
        self.default_visual_ = VisualSpec('points', {'position': 'x'})


class PopulationElement(Element):
    """Population drawn as a line, which grows at a unit rate."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('line'),
        *,
        n: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n

    def _get_laws(self, s):
        """Return the growth of the population."""
        return [RateLaw(s.n, s.n)]


class CellElement(Element):
    """Cell with a state named like the size of a group."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        size: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.size = size


def test_group_stores_arguments():
    """Test that a group stores its label, size, and values unmodified, and clones from them."""
    positions = np.zeros((1000, 2))
    group = GroupElement('gas', element=BallElement, size=1000, values={'x': positions})
    assert (group.label, group.size) == ('gas', 1000)
    assert group.values['x'] is positions
    assert clone(group).size == 1000


def test_group_prepares_its_size_values_and_visual():
    """Test that a prepared group has its size, its values, and the visual of its element type."""
    group = GroupElement('herds', element=PopulationElement, size=3, values={'n': [1.0, 2.0, 3.0]}).prepare()
    assert group.size_ == 3
    assert group.values_ == {'n': [1.0, 2.0, 3.0]}
    assert group.default_visual_ == VisualSpec('line')
    plain = GroupElement('cells', element=CellElement, size=2).prepare()
    assert plain.default_visual_ == VisualSpec()


def test_group_runs_into_the_laws_of_one_member():
    """Test that a group runs into the laws of its element type, written for one member."""
    n = var('n')
    (law,) = GroupElement('herds', element=PopulationElement, size=3).prepare().run(SimpleNamespace(n=n)).laws_
    assert law.value == n


def test_group_shared_and_lattice_values_are_accepted():
    """Test that shared values, per-member arrays, and values on the sites of a space pass the checks of a group."""
    assert GroupElement('gas', element=BallElement, size=3, values={'x': [0.5, 0.5]}).prepare().size_ == 3
    sites = GroupElement('cells', element=PopulationElement, space=LatticeSpace((2, 3)), on='sites')
    assert sites.prepare().size_ == 6


def test_group_sizes_itself_from_the_parts_of_its_space():
    """Test that a group on the edges of a network or a layer without a size has one member per edge."""
    space = NetworkSpace(nx.cycle_graph(4), layers=[LayerSpec('work', nx.complete_graph(4))])
    assert GroupElement('wages', element=PopulationElement, space=space, on='edges').prepare().size_ == 4
    work = GroupElement('wages', element=PopulationElement, space=space, on='edges', layer='work')
    assert work.prepare().size_ == 6
    inherited = GroupElement('weights', element=PopulationElement, on='edges')
    assert inherited.prepare(NetworkSpace(nx.cycle_graph(30))).size_ == 30


def test_group_on_parts_without_a_space_raises_error():
    """Test that a group on parts with neither a size nor a space cannot be prepared."""
    with pytest.raises(SpaceError, match='neither a size nor a space'):
        GroupElement('weights', element=PopulationElement, on='edges').prepare()


def test_group_wrong_number_of_values_raises_error():
    """Test that preparing a group with the wrong number of values names the group, the variable, and both sizes."""
    group = GroupElement('gas', element=BallElement, size=1000, values={'x': np.zeros((999, 2))})
    with pytest.raises(ValidationError, match=r"'gas' has 999 values of x for 1000 members"):
        group.prepare()


@pytest.mark.parametrize('size', [0, 2.5, None])
def test_group_invalid_size_raises_error(size):
    """Test that checking a group rejects a size that is not a positive integer."""
    with pytest.raises(ValidationError, match='not a positive integer'):
        GroupElement('herds', element=PopulationElement, size=size).prepare()


def test_empty_variable_group_on_sites_is_accepted():
    """Test that a variable group on sites may start empty."""
    group = GroupElement('agents', element=PopulationElement, size=0, variable=True, on='sites')
    assert group.prepare() is group


def test_group_unknown_value_raises_error():
    """Test that checking a group with a value its element type does not take names the value."""
    with pytest.raises(ValueError, match="'foo'"):
        GroupElement('herds', element=PopulationElement, size=3, values={'foo': 1}).prepare()


def test_group_non_element_type_raises_error():
    """Test that checking a group rejects an element that is not an element class."""
    with pytest.raises(ValidationError, match='not an element class'):
        GroupElement('herds', element=int, size=3).prepare()


def test_unknown_part_is_rejected():
    """Test that checking a group on a part other than sites, edges, and hyperedges raises an error."""
    with pytest.raises(SpaceError, match='faces'):
        GroupElement('cells', element=PopulationElement, on='faces', space=LatticeSpace(4)).prepare()


def test_group_with_invalid_size_is_reported_with_its_path():
    """Test that a group built with an invalid size is reported by validating the system, with the path of the group."""
    system = CompositeSystem(
        systems=[CompositeSystem('meadow', elements=[GroupElement('herds', element=PopulationElement, size=0)])],
    )
    (issue,) = PassCompiler().prepare(system).report_.errors
    assert issue.message.startswith("meadow/herds: Group 'herds' has size 0")


def test_group_vector_state_has_member_shape():
    """Test that a group's vector state has one row per member."""
    system = CompositeSystem(
        elements=[GroupElement('gas', element=BallElement, size=1000, values={'x': np.zeros((1000, 2))})],
        space=ContinuousSpace(n_dim=2),
    )
    assert PassCompiler().prepare(system).run().problem_.states.get_entry('gas.x').shape == (1000, 2)


def test_group_takes_state_named_size():
    """Test that a state named size takes per-member values, since the group size is a constructor argument."""
    group = GroupElement('cells', element=CellElement, size=3, values={'size': [1.0, 2.0, 3.0]})
    problem = PassCompiler().prepare(CompositeSystem(elements=[group])).run().problem_
    assert problem.states.read(problem.initial_state, 'cells.size').tolist() == [1.0, 2.0, 3.0]


def test_clone_keeps_values_and_compiles_to_the_same_problem():
    """Test that a clone of a group of 100 members has the same values and compiles to the same problem."""
    group = GroupElement('herds', element=PopulationElement, size=100, values={'n': np.linspace(1.0, 2.0, 100)})
    cloned = clone(group)
    assert np.array_equal(cloned.values['n'], group.values['n'])
    first, second = (
        PassCompiler().prepare(CompositeSystem(elements=[group])).run().problem_,
        PassCompiler().prepare(CompositeSystem(elements=[cloned])).run().problem_,
    )
    assert np.array_equal(first.initial_state, second.initial_state)
    assert first.equations == second.equations


def test_get_states_returns_the_states_of_a_member():
    """Test that a group returns the states of its members, which it does not declare itself."""
    assert list(GroupElement('cells', element=PopulationElement, size=2).get_states()) == ['n']


@pytest.mark.parametrize(
    'arguments',
    [
        {'on': 'hyperedges', 'variable': True, 'size': 0, 'rank': 1},
        {'on': 'hyperedges', 'variable': False, 'size': 2, 'rank': 3},
        {'on': 'sites', 'variable': True, 'size': 2, 'rank': 3},
    ],
)
def test_group_rejects_a_rank_outside_variable_groups_on_hyperedges(arguments):
    """Test that a rank below 2, or of a group that is not a variable group on hyperedges, is rejected."""
    with pytest.raises(ValidationError, match='rank'):
        GroupElement('groups', element=Element, **arguments).prepare()


def test_group_on_hyperedges_takes_a_rank_and_an_empty_size():
    """Test that an empty variable group on hyperedges with a rank is valid."""
    group = GroupElement('groups', element=Element, on='hyperedges', variable=True, size=0, capacity=4, rank=3)
    assert group.prepare().size_ == 0


class HeightElement(Element):
    """Cell with a height and no visual of its own."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        h: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.h = h


class ColumnElement(HeightElement):
    """Column drawn by its heights."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('heights'),
        *,
        h: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, h=h)


def test_group_on_sites_flattens_values_of_the_lattice_shape():
    """Test that a group on the sites of a lattice has one member per site, in row-major order."""
    values = {'h': np.arange(12.0).reshape(3, 4)}
    cells = GroupElement('cells', element=HeightElement, space=LatticeSpace((3, 4)), on='sites', values=values)
    cells.prepare()
    assert (cells.size_, cells.on) == (12, 'sites')
    assert cells.values_['h'].tolist() == np.arange(12.0).tolist()


def test_group_on_sites_flattens_values_of_an_inherited_lattice():
    """Test that a group on sites without a space of its own reads the shape of the space it inherits."""
    cells = GroupElement('cells', element=HeightElement, on='sites', values={'h': np.eye(3, 4)})
    assert cells.prepare(LatticeSpace((3, 4))).values_['h'].tolist() == np.eye(3, 4).ravel().tolist()


def test_group_on_sites_of_a_network_reads_values_by_node():
    """Test that a group on the sites of a network takes a mapping from node to value in node order."""
    graph = nx.Graph([('b', 'a'), ('a', 'c')])
    values = {'h': {'a': 1, 'b': 2, 'c': 3}}
    cells = GroupElement('cells', element=HeightElement, space=NetworkSpace(graph), on='sites', values=values)
    assert cells.prepare().values_['h'].tolist() == [2, 1, 3]


def test_group_on_sites_rejects_values_of_another_shape():
    """Test that values that are neither of the lattice shape nor one per site are rejected."""
    values = {'h': np.zeros(5)}
    cells = GroupElement('cells', element=HeightElement, space=LatticeSpace((3, 4)), on='sites', values=values)
    with pytest.raises(ValidationError, match='5 values of h for 12 members'):
        cells.prepare()


def test_group_with_a_size_takes_values_as_given():
    """Test that a group with a size of its own does not reshape values of the lattice shape."""
    values = {'h': np.eye(3, 4)}
    cells = GroupElement('cells', element=HeightElement, size=3, space=LatticeSpace((3, 4)), on='sites', values=values)
    assert cells.prepare().values_['h'] is values['h']


def test_group_on_sites_keeps_the_visual_of_its_element_type():
    """Test that a group on sites of a type that declares a visual takes that visual, not that of its space."""
    columns = GroupElement('columns', element=ColumnElement, space=LatticeSpace((3,)), on='sites').prepare()
    assert columns.default_visual_.kind == 'heights'


def test_group_on_sites_takes_the_visual_of_its_space():
    """Test that a group on sites of a type that declares no visual is drawn as its space draws groups."""
    lattice = GroupElement('cells', element=HeightElement, space=LatticeSpace((3,)), on='sites').prepare()
    assert lattice.default_visual_ == VisualSpec('image')
    network = GroupElement('cells', element=HeightElement, space=NetworkSpace(nx.path_graph(3)), on='sites')
    assert network.prepare().default_visual_ == VisualSpec('sites')
    inherited = GroupElement('cells', element=HeightElement, on='sites')
    assert inherited.prepare(LatticeSpace((3,))).default_visual_ == VisualSpec('image')
