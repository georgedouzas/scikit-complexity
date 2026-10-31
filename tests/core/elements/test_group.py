"""Test the `_group` module of `skcomplex.core.elements`."""

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import var
from sklearn.base import clone

from skcomplex.core.elements import Element, GroupElement, PointElement
from skcomplex.core.laws import RateLaw
from skcomplex.core.spaces import ContinuousSpace, LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import PassCompiler, compile_system
from skcomplex.errors import SpaceError, ValidationError
from skcomplex.specs import LayerSpec, StateSpec, VisualSpec


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
    group = GroupElement('gas', element=PointElement, size=1000, values={'x': positions})
    assert (group.label, group.size) == ('gas', 1000)
    assert group.values['x'] is positions
    assert clone(group).size == 1000


def test_group_prepares_its_size_values_and_visual():
    """Test that a prepared group has its size, its values, and the visual of its element type."""
    group = GroupElement('herds', element=PopulationElement, size=3, values={'n': [1.0, 2.0, 3.0]}).check().prepare()
    assert group.size_ == 3
    assert group.values_ == {'n': [1.0, 2.0, 3.0]}
    assert group.default_visual_ == VisualSpec('line')
    plain = GroupElement('cells', element=CellElement, size=2).check().prepare()
    assert plain.default_visual_ == VisualSpec()


def test_group_runs_into_the_laws_of_one_member():
    """Test that a group runs into the laws of its element type, written for one member."""
    n = var('n')
    (law,) = GroupElement('herds', element=PopulationElement, size=3).check().prepare().run(SimpleNamespace(n=n)).laws_
    assert law.value == n


def test_group_shared_and_lattice_values_are_accepted():
    """Test that shared values, per-member arrays, and values on the sites of a space pass the checks of a group."""
    assert GroupElement('gas', element=PointElement, size=3, values={'x': [0.5, 0.5]}).check().prepare().size_ == 3
    sites = GroupElement('cells', element=PopulationElement, space=LatticeSpace((2, 3)), on='sites')
    assert sites.check().prepare().size_ == 6


def test_group_sizes_itself_from_the_parts_of_its_space():
    """Test that a group on the edges of a network or a layer without a size has one member per edge."""
    space = NetworkSpace(nx.cycle_graph(4), layers=[LayerSpec('work', nx.complete_graph(4))])
    assert GroupElement('wages', element=PopulationElement, space=space, on='edges').check().prepare().size_ == 4
    work = GroupElement('wages', element=PopulationElement, space=space, on='edges', layer='work')
    assert work.check().prepare().size_ == 6
    inherited = GroupElement('weights', element=PopulationElement, on='edges').check()
    assert inherited.prepare(NetworkSpace(nx.cycle_graph(30))).size_ == 30


def test_group_on_parts_without_a_space_raises_error():
    """Test that a group on parts with neither a size nor a space cannot be prepared."""
    with pytest.raises(SpaceError, match='neither a size nor a space'):
        GroupElement('weights', element=PopulationElement, on='edges').check().prepare()


def test_group_wrong_number_of_values_raises_error():
    """Test that preparing a group with the wrong number of values names the group, the variable, and both sizes."""
    group = GroupElement('gas', element=PointElement, size=1000, values={'x': np.zeros((999, 2))}).check()
    with pytest.raises(ValidationError, match=r"'gas' has 999 values of x for 1000 members"):
        group.prepare()


@pytest.mark.parametrize('size', [0, 2.5, None])
def test_group_invalid_size_raises_error(size):
    """Test that checking a group rejects a size that is not a positive integer."""
    with pytest.raises(ValidationError, match='not a positive integer'):
        GroupElement('herds', element=PopulationElement, size=size).check()


def test_empty_variable_group_on_sites_is_accepted():
    """Test that a variable group on sites may start empty."""
    group = GroupElement('agents', element=PopulationElement, size=0, variable=True, on='sites')
    assert group.check() is group


def test_group_unknown_value_raises_error():
    """Test that checking a group with a value its element type does not take names the value."""
    with pytest.raises(ValueError, match="'foo'"):
        GroupElement('herds', element=PopulationElement, size=3, values={'foo': 1}).check()


def test_group_non_element_type_raises_error():
    """Test that checking a group rejects an element that is not an element class."""
    with pytest.raises(ValidationError, match='not an element class'):
        GroupElement('herds', element=int, size=3).check()


def test_unknown_part_is_rejected():
    """Test that checking a group on a part other than sites, edges, and hyperedges raises an error."""
    with pytest.raises(SpaceError, match='faces'):
        GroupElement('cells', element=PopulationElement, on='faces', space=LatticeSpace(4)).check()


def test_group_with_invalid_size_is_reported_with_its_path():
    """Test that a group built with an invalid size is reported by validating the system, with the path of the group."""
    system = System(systems=[System('meadow', elements=[GroupElement('herds', element=PopulationElement, size=0)])])
    (issue,) = PassCompiler().check().prepare(system).report_.errors
    assert issue.message.startswith("meadow/herds: Group 'herds' has size 0")


def test_group_vector_state_has_member_shape():
    """Test that a group's vector state has one row per member."""
    system = System(
        elements=[GroupElement('gas', element=PointElement, size=1000, values={'x': np.zeros((1000, 2))})],
        space=ContinuousSpace(n_dim=2),
    )
    assert compile_system(system).states.entry('gas.x').shape == (1000, 2)


def test_group_takes_state_named_size():
    """Test that a state named size takes per-member values, since the group size is a constructor argument."""
    group = GroupElement('cells', element=CellElement, size=3, values={'size': [1.0, 2.0, 3.0]})
    problem = compile_system(System(elements=[group]))
    assert problem.states.read(problem.initial_state, 'cells.size').tolist() == [1.0, 2.0, 3.0]


def test_clone_keeps_values_and_compiles_to_the_same_problem():
    """Test that a clone of a group of 100 members has the same values and compiles to the same problem."""
    group = GroupElement('herds', element=PopulationElement, size=100, values={'n': np.linspace(1.0, 2.0, 100)})
    cloned = clone(group)
    assert np.array_equal(cloned.values['n'], group.values['n'])
    first, second = compile_system(System(elements=[group])), compile_system(System(elements=[cloned]))
    assert np.array_equal(first.initial_state, second.initial_state)
    assert first.equations == second.equations
