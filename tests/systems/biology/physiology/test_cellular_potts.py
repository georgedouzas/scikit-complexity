"""Test the cellular Potts modules of `skcomplex.systems.biology.physiology`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.systems import create_simulation
from skcomplex.systems.biology.physiology import CellAdhesionInteraction, cellular_potts

SHAPE = (30, 32)


def _series(lattice, until, **parameters):
    """Return the trajectory of a cellular Potts tissue on a lattice."""
    simulation = create_simulation('cellular_potts', shape=SHAPE, lattice=lattice, **parameters)
    return simulation.prepare().run(until=until).trajectory_


@pytest.mark.parametrize('lattice', ['square', 'triangular', 'disordered'])
def test_area_constraint_holds_the_cell_areas_near_the_target_on_average(lattice):
    """Test that the mean deviation of the cell areas from the target area stays below a tenth of it."""
    series = _series(lattice, 200)
    cells = np.rint(series.values('tissue.cell')[-1]).astype(int).ravel()
    assert np.unique(cells).size == (SHAPE[0] // 5) * (SHAPE[1] // 8)
    assert series.values('area_deviation')[50:].mean() < 0.1


def test_stronger_area_constraint_tightens_the_areas():
    """Test that a tenfold area strength lowers the deviation of the areas from the target."""
    loose = _series('square', 200, area_strength=0.1).values('area_deviation')[50:].mean()
    tight = _series('square', 200, area_strength=1.0).values('area_deviation')[50:].mean()
    assert tight < loose


@pytest.mark.parametrize('lattice', ['square', 'disordered'])
def test_differential_adhesion_sorts_the_kinds_of_cells(lattice):
    """Test that a heterotypic adhesion above the homotypic ones raises the share of homotypic contacts by 0.1."""
    shares = _series(lattice, 1500).values('homotypic_contacts')
    assert shares[-100:].mean() > shares[0] + 0.1


def test_equal_adhesion_does_not_sort_the_kinds_of_cells():
    """Test that equal adhesion between all kinds keeps the share of homotypic contacts near its initial value."""
    shares = _series('square', 1500, adhesion=((1.5, 1.5), (1.5, 1.5))).values('homotypic_contacts')
    assert abs(shares[-100:].mean() - shares[0]) < 0.1


def test_disordered_lattice_tiles_the_box_with_weighted_contacts():
    """Test that the Voronoi sites tile the periodic box and that the contact weights have mean 1."""
    graph = cellular_potts(SHAPE, lattice='disordered').elements[0].space.graph
    areas = [area for _, area in graph.nodes(data='volume')]
    weights = [weight for *_, weight in graph.edges(data='weight')]
    assert abs(sum(areas) - SHAPE[0] * SHAPE[1]) < 1e-6
    assert abs(np.mean(weights) - 1.0) < 1e-12
    assert 5.5 < 2 * graph.number_of_edges() / graph.number_of_nodes() < 6.5


@pytest.mark.parametrize(
    ('parameters', 'match'),
    [
        ({'lattice': 'hexagonal'}, 'lattice'),
        ({'shape': (31, 32)}, 'multiple of the cell shape'),
    ],
)
def test_cellular_potts_rejects_lattices_and_shapes_it_cannot_tile(parameters, match):
    """Test that an unknown lattice or a shape that the cells do not tile raise an error."""
    with pytest.raises(ValidationError, match=match):
        cellular_potts(**{'shape': SHAPE, **parameters})


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'kinds': (0, 1), 'adhesion': ((1.0, 2.0), (3.0, 1.0))}, 'symmetric matrix'),
        ({'kinds': (0, 2), 'adhesion': ((1.0, 2.0), (2.0, 1.0))}, 'rows of the adhesion'),
    ],
)
def test_cell_adhesion_rejects_adhesions_and_kinds_it_cannot_read(arguments, match):
    """Test that an asymmetric adhesion or a kind outside it raise an error."""
    with pytest.raises(ValidationError, match=match):
        CellAdhesionInteraction('adhesion', target='tissue', **arguments).prepare()
