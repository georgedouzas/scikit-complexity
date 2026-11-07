"""Test the `_gauge_agents` module of `skcomplex.systems.social.opinion_dynamics`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import cell_complex
from skcomplex.core.errors import ValidationError
from skcomplex.systems import create_simulation, create_system
from skcomplex.systems.social.opinion_dynamics import gauge_network_agents


def test_frustration_blocks_agreement():
    """Test that a flat ring of 12 agents agrees with zero energy, and a flux of pi keeps an energy and a twist."""
    flat = create_simulation('gauge_network_agents').prepare().run().trajectory_
    assert flat.values('energy')[-1] < 1e-10
    assert flat.values('twist')[-1] < 1e-5
    assert np.ptp(np.angle(np.exp(1j * flat.values('agents.theta')[-1]))) < 1e-5
    frustrated = create_simulation('gauge_network_agents', flux=np.pi).prepare().run().trajectory_
    assert frustrated.values('energy')[-1] == pytest.approx(12 * (1 - np.cos(np.pi / 12)), abs=1e-6)
    assert frustrated.values('twist')[-1] == pytest.approx(np.pi, abs=1e-6)


def test_curvature_and_betti_numbers_read_the_connection():
    """Test the curvature of the connection and the Betti numbers of the ring with its one face."""
    for flux, curved in ((0.0, 0.0), (np.pi, 1.0)):
        system = create_system('gauge_network_agents', flux=flux)
        links = system.elements[1]
        space = links.space
        curvature = cell_complex(links.values['A'], kind='curvature', space=space)
        assert curvature.share[0] == curved
    assert cell_complex(None, kind='betti', space=space).betti == (1, 0, 0)


def test_flux_needs_a_loop():
    """Test that a flux on a tree raises an error, and that connections on a given graph are taken as they are."""
    with pytest.raises(ValidationError, match='needs a loop'):
        gauge_network_agents(flux=1.0, graph=nx.path_graph(4))
    system = gauge_network_agents(graph=nx.path_graph(3), connections=[0.5, -0.5])
    assert system.elements[1].values['A'].tolist() == [0.5, -0.5]
