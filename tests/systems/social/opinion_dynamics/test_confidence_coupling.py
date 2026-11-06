"""Test the `_confidence_coupling` module of `skcomplex.systems.social.opinion_dynamics`."""

import networkx as nx
import numpy as np

from skcomplex.core.components.elements import LatticeElement
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.social.opinion_dynamics import ConfidenceCouplingInteraction, ConfidentElement


def test_confidence_coupling_takes_the_mean_of_the_trusted_neighbours():
    """Test that one step gives every agent the mean opinion of the trusted neighbours, itself included."""
    opinions = np.array([0.1, 0.2, 0.35, 0.9, 0.95])
    people = LatticeElement(
        'people',
        element=ConfidentElement,
        space=NetworkSpace(nx.complete_graph(5)),
        values={'opinion': opinions, 'confidence': 0.2},
    )
    system = System(
        elements=[people],
        interactions=[ConfidenceCouplingInteraction('compromise', target='people')],
        time=DiscreteTime(),
    )
    values = Simulation(system).prepare().run(until=1).trajectory_.values('people.opinion')[-1]
    close = np.abs(opinions[None, :] - opinions[:, None]) < 0.2
    np.testing.assert_allclose(values, (close * opinions).sum(1) / close.sum(1), atol=1e-12)
