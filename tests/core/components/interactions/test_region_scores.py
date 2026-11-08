"""Test the region scores and the label copies of `_sampling` in `skcomplex.core.components.interactions`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace, MooreNeighborhood, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import RegionScoreSpec, RowScoreSpec, StateSpec

INFINITE = float('inf')
SIDE = 12
TARGET = 30.0


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
    copies = SamplingInteraction(
        'copies',
        target='sites',
        state='region',
        objective=score,
        move='copy',
        intensity=intensity,
        neighborhood=neighborhood,
    )
    sites = LatticeElement('sites', element=SiteElement, space=space, values={'region': regions})
    system = System(elements=[sites], interactions=[copies], time=DiscreteTime())
    values = simulate(system, until=until, random_state=seed).values('sites.region')
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
    table = space._index_table(MooreNeighborhood())
    energies = [_energy(frame, table, _unit_contact, _region_volume) for frame in labels]
    assert np.all(np.diff(energies) <= 1e-9)


def test_contact_energy_between_labels_sorts_them():
    """Test that a heterotypic contact above the mean of the homotypic ones shortens the boundaries between kinds."""
    rng = np.random.default_rng(0)
    regions = rng.integers(2, size=(SIDE, SIDE)).astype(float)

    def contact(first, second):
        return np.where(np.asarray(first) == np.asarray(second), 0.0, 1.0)

    labels = _run(LatticeSpace((SIDE, SIDE)), regions, RegionScoreSpec(contact), until=100, intensity=2.0)
    table = LatticeSpace((SIDE, SIDE)).prepare()._index_table(MooreNeighborhood())
    boundary = [(frame[:, None] != frame[table]).sum() for frame in labels]
    assert boundary[-1] < boundary[0] / 3


def test_surface_constraint_holds_the_surface_of_a_region():
    """Test that a strong surface constraint holds the weighted number of outer contacts of a region near its target."""

    def surface(labels):
        return np.where(labels == 0, 0.0, 1.0), np.full(labels.shape, 40.0)

    space = LatticeSpace((SIDE, SIDE)).prepare()
    score = RegionScoreSpec(lambda first, second: np.zeros(np.broadcast(first, second).shape), surface=surface)
    labels = _run(space, _single_region(), score)
    table = space._index_table(MooreNeighborhood())
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
        ({'objective': RowScoreSpec(np.sum), 'move': 'copy'}, 'copies labels'),
        ({'objective': RegionScoreSpec(_unit_contact, hits=0), 'move': 'copy'}, 'at least one copy'),
    ],
)
def test_region_score_rejects_moves_and_rules_it_cannot_take(arguments, match):
    """Test that a region score needs label copies and that a label copy needs a region score."""
    with pytest.raises(ValidationError, match=match):
        SamplingInteraction('copies', target='sites', state='region', **arguments).prepare()


def test_region_score_rejects_negative_labels():
    """Test that a state with a negative label raises an error."""
    regions = -_single_region()
    with pytest.raises(ValidationError, match='labels from 0'):
        _run(LatticeSpace((SIDE, SIDE)), regions, RegionScoreSpec(_unit_contact), until=1)
