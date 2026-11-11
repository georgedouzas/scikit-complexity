"""Test the cellular Potts modules of `skcomplex.systems.biology.physiology`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import BallNeighborhood, DiscreteTime, LatticeSpace, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec
from skcomplex.systems.biology.physiology import (
    CellAdhesionInteraction,
    CellularPottsSystem,
    RegionSamplingInteraction,
    RegionScoreSpec,
)

SHAPE = (30, 32)
INFINITE = float('inf')
SIDE = 12
TARGET = 30.0


def _series(lattice, until, **parameters):
    """Return the trajectory of a cellular Potts tissue on a lattice."""
    return simulate(CellularPottsSystem(shape=SHAPE, lattice=lattice, **parameters), until=until, random_state=0)


def _area(labels):
    """Return a volume constraint of strength 1 and target 40."""
    del labels
    return 1.0, 40.0


class SiteElement(Element):
    """Site that holds the label of its region."""

    def __init__(self, label, region: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.region = region


def _unit_contact(first, second):
    """Return a contact energy of 1 between any two labels."""
    return np.ones(np.broadcast(first, second).shape)


def _region_volume(labels):
    """Return a volume constraint of strength 1 and target `TARGET` for every label but the medium 0."""
    return np.where(labels == 0, 0.0, 1.0), np.full(labels.shape, TARGET)


def _run(space, regions, score, *, until=200, intensity=1.0, neighborhood=None, seed=0):
    """Return the recorded labels of a sampling of regions on a space."""
    copies = RegionSamplingInteraction(
        'copies',
        target='sites',
        state='region',
        objective=score,
        move='copy',
        intensity=intensity,
        neighborhood=neighborhood,
    )
    sites = GroupElement('sites', element=SiteElement, space=space, values={'region': regions}, on='sites')
    system = CompositeSystem(elements=[sites], interactions=[copies], time=DiscreteTime())
    values = simulate(system, until=until, random_state=seed).get_values('sites.region')
    return np.rint(values.reshape(values.shape[0], -1)).astype(int)


def _energy(labels, table, contact, volume):
    """Return the objective of a region score recomputed from the labels, over a table of neighbours."""
    first, second = np.repeat(np.arange(labels.size), table.shape[1]), table.ravel()
    boundary = labels[first] != labels[second]
    pairs = 0.5 * (contact(labels[first], labels[second]) * boundary).sum()
    ids = np.arange(labels.max() + 1)
    strengths, targets = volume(ids)
    return pairs + (strengths * (np.bincount(labels, minlength=ids.size) - targets) ** 2).sum()


def _single_region():
    """Return a lattice of medium with one square region of 9 sites."""
    regions = np.zeros((SIDE, SIDE))
    regions[4:7, 4:7] = 1
    return regions


@pytest.mark.parametrize('lattice', ['square', 'triangular', 'disordered'])
def test_area_constraint_holds_the_cell_areas_near_the_target_on_average(lattice):
    """Test that the mean deviation of the cell areas from the target area stays below a tenth of it."""
    series = _series(lattice, 200)
    cells = np.rint(series.get_values('tissue.cell')[-1]).astype(int).ravel()
    assert np.unique(cells).size == (SHAPE[0] // 5) * (SHAPE[1] // 8)
    assert series.get_values('area_deviation')[50:].mean() < 0.1


def test_stronger_area_constraint_tightens_the_areas():
    """Test that a tenfold area strength lowers the deviation of the areas from the target."""
    loose = _series('square', 200, area_strength=0.1).get_values('area_deviation')[50:].mean()
    tight = _series('square', 200, area_strength=1.0).get_values('area_deviation')[50:].mean()
    assert tight < loose


@pytest.mark.parametrize('lattice', ['square', 'disordered'])
def test_differential_adhesion_sorts_the_kinds_of_cells(lattice):
    """Test that a heterotypic adhesion above the homotypic ones raises the share of homotypic contacts by 0.1."""
    shares = _series(lattice, 1500).get_values('homotypic_contacts')
    assert shares[-100:].mean() > shares[0] + 0.1


def test_equal_adhesion_does_not_sort_the_kinds_of_cells():
    """Test that equal adhesion between all kinds keeps the share of homotypic contacts near its initial value."""
    shares = _series('square', 1500, adhesion=((1.5, 1.5), (1.5, 1.5))).get_values('homotypic_contacts')
    assert abs(shares[-100:].mean() - shares[0]) < 0.1


def test_disordered_lattice_tiles_the_box_with_weighted_contacts():
    """Test that the Voronoi sites tile the periodic box and that the contact weights have mean 1."""
    graph = CellularPottsSystem(SHAPE, lattice='disordered').prepare().system_.elements[0].space.graph
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
        CellularPottsSystem(**{'shape': SHAPE, **parameters}).prepare()


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


def test_region_score_keeps_its_constraints_and_hits():
    """Test that a valid score keeps its constraints and its number of hits."""
    area = _area
    score = RegionScoreSpec(np.maximum, volume=area, hits=3)
    assert (score.volume, score.surface, score.hits) == (area, None, 3)


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'contact': 1.0}, 'contact of a region score'),
        ({'contact': np.maximum, 'volume': 40.0}, 'volume of a region score'),
        ({'contact': np.maximum, 'surface': 'perimeter'}, 'surface of a region score'),
        ({'contact': np.maximum, 'hits': 0}, 'at least one copy'),
    ],
)
def test_region_score_rejects_what_it_cannot_run(arguments, match):
    """Test that a contact or a constraint that is not callable, or no hits, raise an error."""
    with pytest.raises(ValidationError, match=match):
        RegionScoreSpec(**arguments)._check()


def test_volume_constraint_holds_the_volume_of_a_region_near_its_target_on_average():
    """Test that a region of 9 sites in a medium grows to a volume of 30 on average and stays a region."""
    labels = _run(LatticeSpace((SIDE, SIDE)), _single_region(), RegionScoreSpec(_unit_contact, _region_volume))
    volumes = (labels[50:] == 1).sum(axis=1)
    assert abs(volumes.mean() - TARGET) < 2.0
    assert volumes.std() < 5.0


def test_zero_temperature_copies_never_raise_the_objective():
    """Test that copies at infinite intensity keep the objective recomputed from the labels from growing."""
    space = LatticeSpace((SIDE, SIDE)).prepare()
    regions = np.kron(np.arange(4).reshape(2, 2), np.ones((SIDE // 2, SIDE // 2)))
    score = RegionScoreSpec(_unit_contact, _region_volume)
    labels = _run(space, regions, score, until=30, intensity=INFINITE)
    table = space._get_index_table(BallNeighborhood())
    energies = [_energy(frame, table, _unit_contact, _region_volume) for frame in labels]
    assert np.all(np.diff(energies) <= 1e-9)


def test_contact_energy_between_labels_sorts_them():
    """Test that a heterotypic contact above the mean of the homotypic ones shortens the boundaries between kinds."""
    rng = np.random.default_rng(0)
    regions = rng.integers(2, size=(SIDE, SIDE)).astype(float)

    def contact(first, second):
        return np.where(np.asarray(first) == np.asarray(second), 0.0, 1.0)

    labels = _run(LatticeSpace((SIDE, SIDE)), regions, RegionScoreSpec(contact), until=100, intensity=2.0)
    table = LatticeSpace((SIDE, SIDE)).prepare()._get_index_table(BallNeighborhood())
    boundary = [(frame[:, None] != frame[table]).sum() for frame in labels]
    assert boundary[-1] < boundary[0] / 3


def test_surface_constraint_holds_the_surface_of_a_region():
    """Test that a strong surface constraint holds the weighted number of outer contacts of a region near its target."""

    def surface(labels):
        return np.where(labels == 0, 0.0, 1.0), np.full(labels.shape, 40.0)

    space = LatticeSpace((SIDE, SIDE)).prepare()
    score = RegionScoreSpec(lambda first, second: np.zeros(np.broadcast(first, second).shape), surface=surface)
    labels = _run(space, _single_region(), score)
    table = space._get_index_table(BallNeighborhood())
    surfaces = [((frame[:, None] != frame[table]) & (frame[:, None] == 1)).sum() for frame in labels[50:]]
    assert abs(np.mean(surfaces) - 40.0) < 4.0


def test_network_contacts_read_edge_weights_and_node_volumes():
    """Test that a network weighs the volume of a region by the node attribute `'volume'`."""
    graph = nx.grid_2d_graph(SIDE, SIDE, periodic=True)
    nx.set_node_attributes(graph, 0.5, 'volume')
    nx.set_edge_attributes(graph, 2.0, 'weight')
    labels = _run(NetworkSpace(graph), _single_region().ravel(), RegionScoreSpec(_unit_contact, _region_volume))
    volumes = 0.5 * (labels[50:] == 1).sum(axis=1)
    assert abs(volumes.mean() - TARGET) < 2.0


def test_label_copies_are_reproducible_with_a_seed():
    """Test that two runs with one seed copy the same labels."""
    score = RegionScoreSpec(_unit_contact, _region_volume)
    first = _run(LatticeSpace((SIDE, SIDE)), _single_region(), score, until=10)
    second = _run(LatticeSpace((SIDE, SIDE)), _single_region(), score, until=10)
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'objective': RegionScoreSpec(_unit_contact), 'move': 'flip'}, "move 'copy'"),
        ({'objective': RegionScoreSpec(_unit_contact), 'move': 'copy', 'rule': 'heat-bath'}, 'metropolis'),
        ({'objective': RegionScoreSpec(_unit_contact, hits=0), 'move': 'copy'}, 'at least one copy'),
    ],
)
def test_region_score_rejects_moves_and_rules_it_cannot_take(arguments, match):
    """Test that a region score needs label copies and that a label copy needs a region score."""
    with pytest.raises(ValidationError, match=match):
        RegionSamplingInteraction('copies', target='sites', state='region', **arguments).prepare()


def test_region_score_rejects_negative_labels():
    """Test that a state with a negative label raises an error."""
    regions = -_single_region()
    with pytest.raises(ValidationError, match='labels from 0'):
        _run(LatticeSpace((SIDE, SIDE)), regions, RegionScoreSpec(_unit_contact), until=1)
