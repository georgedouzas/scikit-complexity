"""Test the `_adaptive` module of `skcomplex.systems.physics.synchronization`."""

import math

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.synchronization import AdaptiveKuramotoSystem


def test_adaptive_kuramoto_splits_identical_oscillators_into_frequency_clusters():
    """Test that identical adaptive oscillators form several clusters, each of one mean frequency (Fig.

    12).
    """
    trajectory = simulate(AdaptiveKuramotoSystem())
    theta = np.asarray(trajectory.values('nodes.theta'))
    frequencies = np.sort((theta[-1] - theta[-201]) / 200)
    clusters = np.split(frequencies, np.flatnonzero(np.diff(frequencies) > 0.05) + 1)
    assert len(clusters) >= 2
    assert max(np.ptp(cluster) for cluster in clusters) < 0.01


def test_adaptive_kuramoto_weights_follow_the_phases_inside_a_cluster():
    """Test that the weights between members of the largest cluster approach `-sin(theta_i - theta_j + beta)`."""
    system = AdaptiveKuramotoSystem(20).prepare().system_
    trajectory = simulate(system, until=2000.0, dt=1.0)
    theta = np.asarray(trajectory.values('nodes.theta'))
    kappa = np.asarray(trajectory.values('links.kappa'))[-1]
    frequencies = (theta[-1] - theta[-201]) / 200
    median = np.median(frequencies)
    cluster = set(np.flatnonzero(np.abs(frequencies - median) < 0.01))
    sources, targets = system.elements[0].space.prepare().edges_[None]
    inside = [index for index, (j, i) in enumerate(zip(sources, targets, strict=True)) if i in cluster and j in cluster]
    expected = -np.sin(theta[-1, targets[inside]] - theta[-1, sources[inside]] + 0.23 * math.pi)
    assert len(inside) > 0
    assert np.abs(kappa[inside] - expected).max() < 0.1
