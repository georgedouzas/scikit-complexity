"""Test the `_excitable` module of `skcomplex.models.biology.neuroscience`."""

import networkx as nx
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.models.biology.neuroscience import (
    kinouchi_copelli,
)


def _dynamic_range(rates, response):
    """Return the dynamic range in dB of a response curve, `10 log10(r_0.9 / r_0.1)`."""
    response = np.maximum.accumulate(response)
    low, high = response[0], response[-1]
    stimulus = [np.exp(np.interp(low + share * (high - low), response, np.log(rates))) for share in (0.1, 0.9)]
    return 10 * np.log10(stimulus[1] / stimulus[0])


def test_kinouchi_copelli_activity_dies_out_below_and_persists_above_the_critical_point():
    """Test that without stimulus the activity dies out below a branching ratio of 1 and sustains itself above."""
    graph = nx.gnp_random_graph(2000, 10 / 1999, seed=0)
    for sigma, persists in ((0.8, False), (1.5, True)):
        trajectory = Simulation(kinouchi_copelli(graph, sigma=sigma), dt=1, random_state=0).run(300).trajectory_
        assert (trajectory['neurons.phase'][-50:] == 1).any() == persists


def test_kinouchi_copelli_dynamic_range_peaks_at_the_critical_point():
    """Test that the dynamic range of the response to the stimulus peaks at a branching ratio of 1."""
    graph = nx.gnp_random_graph(2000, 10 / 1999, seed=0)
    rates = np.logspace(-5, 1, 16)
    sigmas = (0.5, 0.75, 1.0, 1.25, 1.5)
    ranges = []
    for sigma in sigmas:
        response = []
        for rate in rates:
            system = kinouchi_copelli(graph, sigma=sigma, stimulus=rate)
            trajectory = Simulation(system, dt=1, random_state=0).run(1000).trajectory_
            response.append((trajectory['neurons.phase'][200:] == 1).mean())
        ranges.append(_dynamic_range(rates, np.array(response)))
    assert sigmas[int(np.argmax(ranges))] == 1.0
    assert min(ranges[2] - ranges[1], ranges[2] - ranges[3]) > 1.0
