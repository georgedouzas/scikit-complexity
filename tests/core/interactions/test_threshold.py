"""Test the `_threshold` module of `skcomplex.core.interactions`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.interactions import ThresholdInteraction
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


def _run(space, heights, threshold, until, **arguments):
    """Return a simulation of a threshold over the sites of a space, run for some steps."""
    system = System(
        elements=[LatticeElement('sites', element=SiteElement, space=space, values={'height': heights})],
        interactions=[ThresholdInteraction('cascade', nodes='sites', state='height', threshold=threshold, **arguments)],
        time=DiscreteTime(),
    )
    return Simulation(system, dt=until, random_state=0).check().prepare().run(until=until)


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
    """Test that a firing site on a periodic ring gives its threshold to its neighbours and the total is kept."""
    simulation = _run(
        LatticeSpace(5), [0.0, 1.0, 3.0, 1.0, 0.0], 2, 1, share=1, over=VonNeumannNeighborhood(n_dim=1, center=True),
    )
    assert simulation.state_.tolist() == [1.0] * 5
    (event,) = simulation.trajectory_.events
    assert (event.size, event.duration) == (4, 3)


def test_reset_passes_a_fraction_of_the_fired_state_to_the_neighbours():
    """Test that in the mode reset a firing node falls to 0 and its neighbours gain a share of its state."""
    space = NetworkSpace(nx.path_graph(3))
    simulation = _run(
        space,
        [0.0, 1.2, 0.1],
        1.0,
        1,
        share=0.5,
        over=AdjacencyNeighborhood(center=True),
        mode='reset',
    )
    assert np.allclose(simulation.state_, [0.6, 0.0, 0.7])
    (event,) = simulation.trajectory_.events
    assert event.size == 1


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'mode': 'flip'}, 'mode of the threshold'),
        ({'drive': 1.0}, 'callable'),
        ({'over': VonNeumannNeighborhood()}, 'center=True'),
    ],
)
def test_check_rejects_unknown_modes_drives_and_neighbourhoods(arguments, match):
    """Test that an unknown mode, a drive that is not callable, or a neighbourhood without its centre raise errors."""
    with pytest.raises(ValidationError, match=match):
        ThresholdInteraction('cascade', nodes='sites', **arguments).check()
