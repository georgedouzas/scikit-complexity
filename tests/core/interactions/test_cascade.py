"""Test the `_cascade` module of `skcomplex.core.interactions`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.interactions import CascadeInteraction
from skcomplex.core.spaces import (
    AdjacencyNeighborhood,
    DiscreteTime,
    LatticeSpace,
    NetworkSpace,
    VonNeumannNeighborhood,
)
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec

SIDE = 50
STEPS = 12000
TRANSIENT = 4000
RUNS = 100


class SiteElement(Element):
    """Site with a height."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        height: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.height = height


def _drop(values, *, random):
    """Return the heights with one grain dropped on a random site."""
    following = values.copy()
    following[tuple(random.integers(size) for size in values.shape)] += 1
    return following


def _run(space, heights, threshold, until, seed=0, **arguments):
    """Return a simulation of a cascade over the sites of a space, run for some steps."""
    system = System(
        elements=[LatticeElement('sites', element=SiteElement, space=space, values={'height': heights})],
        interactions=[CascadeInteraction('cascade', target='sites', state='height', threshold=threshold, **arguments)],
        time=DiscreteTime(),
    )
    return Simulation(system, dt=until, random_state=seed).check().prepare().run(until=until)


def test_sandpile_avalanches_follow_a_power_law_over_two_decades():
    """Test that the avalanche sizes of a 50 by 50 sandpile lie on a power law from 1 to 300 firings."""
    heights = np.random.default_rng(0).integers(0, 4, size=(SIDE, SIDE)).astype(float)
    space = LatticeSpace((SIDE, SIDE), boundary='fixed')
    simulation = _run(space, heights, 4, STEPS, share=1, drive=_drop)
    sizes = np.array([event.size for event in simulation.trajectory_.events if event.time > TRANSIENT])
    edges = np.logspace(0, 2.5, 11)
    counts, _ = np.histogram(sizes, bins=edges)
    centers, density = np.sqrt(edges[1:] * edges[:-1]), counts / np.diff(edges) / sizes.size
    logarithms = np.log10(density)
    slope, intercept = np.polyfit(np.log10(centers), logarithms, 1)
    residuals = logarithms - (slope * np.log10(centers) + intercept)
    assert np.all(counts > 0)
    assert -1.5 < slope < -0.9
    assert 1 - residuals.var() / logarithms.var() > 0.95
    assert all(event.label == 'cascade' for event in simulation.trajectory_.events)


def test_redistribution_keeps_the_grains_inside_a_periodic_lattice():
    """Test that a relaxing site on a periodic ring gives its threshold to its neighbours and the total is kept."""
    simulation = _run(
        LatticeSpace(5),
        [0.0, 1.0, 3.0, 1.0, 0.0],
        2,
        1,
        share=1,
        neighborhood=VonNeumannNeighborhood(n_dim=1, center=True),
    )
    assert simulation.state_.tolist() == [1.0] * 5
    (event,) = simulation.trajectory_.events
    assert (event.size, event.duration) == (4, 3)


def test_reset_passes_a_fraction_of_the_relaxed_state_to_the_neighbours():
    """Test that in the mode reset a relaxing node falls to 0 and its neighbours gain a share of its state."""
    space = NetworkSpace(nx.path_graph(3))
    simulation = _run(
        space,
        [0.0, 1.2, 0.1],
        1.0,
        1,
        share=0.5,
        neighborhood=AdjacencyNeighborhood(center=True),
        mode='reset',
    )
    assert np.allclose(simulation.state_, [0.6, 0.0, 0.7])
    (event,) = simulation.trajectory_.events
    assert event.size == 1


def test_random_mode_passes_parcels_to_random_neighbours():
    """Test that in the mode random a relaxing site sends each parcel to a random neighbour, keeping them inside."""
    heights = np.zeros((9, 9))
    heights[4, 4] = 2.0
    single = 0
    for seed in range(RUNS):
        simulation = _run(LatticeSpace((9, 9), boundary='fixed'), heights.ravel(), 2, 1, seed, share=1, mode='random')
        following = simulation.state_.reshape(9, 9)
        (event,) = simulation.trajectory_.events
        assert following.sum() == 2.0
        if event.size == 1:
            single += 1
            assert following[4, 4] == 0.0
            assert sorted(following[[3, 5, 4, 4], [4, 4, 3, 5]].tolist()) == [0.0, 0.0, 1.0, 1.0]
    assert single / RUNS == pytest.approx(3 / 4, abs=3 * np.sqrt(3 / 16 / RUNS))


def test_random_mode_drives_a_stochastic_sandpile_to_a_stationary_density():
    """Test that random parcels of two grains on a bounded lattice settle at a mean height below the threshold."""
    space = LatticeSpace((20, 20), boundary='fixed')
    simulation = _run(space, np.zeros(400), 2, 3000, share=1, mode='random', drive=_drop)
    heights = simulation.trajectory_.values('sites.height')
    assert heights.max() < 2
    assert 0.5 < heights[-1].mean() < 1.0
    assert max(event.size for event in simulation.trajectory_.events) > 50


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'mode': 'flip'}, 'mode of the cascade'),
        ({'drive': 1.0}, 'callable'),
        ({'neighborhood': VonNeumannNeighborhood()}, 'center=True'),
        ({'mode': 'random', 'neighborhood': AdjacencyNeighborhood(center=True)}, 'lattice neighbourhood'),
        ({'mode': 'random', 'threshold': 3.0, 'share': 2.0}, 'divides the threshold'),
    ],
)
def test_check_rejects_unknown_modes_drives_and_neighbourhoods(arguments, match):
    """Test that unknown modes, drives, neighbourhoods, and parcels that do not divide the threshold raise errors."""
    with pytest.raises(ValidationError, match=match):
        CascadeInteraction('cascade', target='sites', **arguments).check()
