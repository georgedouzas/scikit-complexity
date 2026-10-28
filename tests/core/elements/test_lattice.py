"""Test the `_lattice` module of `skcomplex.core.elements`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.spaces import LatticeSpace, NetworkSpace
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec


class ColumnElement(Element):
    """Column drawn by its heights."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('heights'),
        *,
        h: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.h = h


class CellElement(Element):
    """Cell without a visual of its own."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        h: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.h = h


def test_lattice_group_has_one_member_per_site():
    """Test that a lattice group has one member per site and flattens values of the lattice shape."""
    cells = LatticeElement('cells', element=CellElement, space=LatticeSpace((3, 4)), values={'h': np.eye(3, 4)})
    cells.check().prepare()
    assert (cells.size_, cells.on) == (12, 'sites')
    assert cells.values_['h'].tolist() == np.eye(3, 4).ravel().tolist()


def test_lattice_group_on_a_network_reads_values_by_node():
    """Test that a lattice group on a network takes a mapping from node to value in node order."""
    graph = nx.Graph([('b', 'a'), ('a', 'c')])
    cells = LatticeElement(
        'cells',
        element=CellElement,
        space=NetworkSpace(graph),
        values={'h': {'a': 1, 'b': 2, 'c': 3}},
    )
    assert cells.check().prepare().values_['h'].tolist() == [2, 1, 3]


def test_lattice_group_rejects_values_of_another_shape():
    """Test that values that are neither of the lattice shape nor one per site are rejected."""
    cells = LatticeElement('cells', element=CellElement, space=LatticeSpace((3, 4)), values={'h': np.zeros(5)})
    with pytest.raises(ValidationError, match='5 values of h for 12 members'):
        cells.check().prepare()


def test_lattice_keeps_visual_of_element_type():
    """Test that a lattice group of a type that declares a visual takes that visual, not an image."""
    columns = LatticeElement('columns', element=ColumnElement, space=LatticeSpace((3,))).check().prepare()
    assert columns.default_visual_.kind == 'heights'


def test_lattice_without_type_visual_takes_the_visual_of_its_space():
    """Test that a lattice group of a type that declares no visual is drawn as its space draws groups."""
    assert LatticeElement('cells', element=CellElement, space=LatticeSpace((3,))).check().prepare().default_visual_ == (
        VisualSpec('image')
    )
    network = NetworkSpace(nx.path_graph(3))
    assert LatticeElement('cells', element=CellElement, space=network).check().prepare().default_visual_ == (
        VisualSpec('sites')
    )
