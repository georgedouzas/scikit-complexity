"""Test the `_fisheries` module of `skcomplex.models.economics.resources`."""

import numpy as np

from skcomplex.models.economics.resources import (
    hilsa_fishery,
)

from ...conftest import _run_monitored


def test_hilsa_fishery_settles_and_collapses_with_the_catch():
    """Test that the stock settles at the spawning-ground limit, gains from a lower juvenile catch, and collapses."""
    base = _run_monitored(hilsa_fishery(), 0.5, 240)['hilsa.spawners']
    assert np.ptp(base[-24:]) < 1e-3 * base[-1]
    protected = _run_monitored(hilsa_fishery(juvenile_harvest=0.2), 0.5, 240)['hilsa.spawners']
    assert protected[-1] > 2 * base[-1]
    overfished = _run_monitored(hilsa_fishery(adult_harvest=0.9), 0.5, 240)['hilsa.spawners']
    assert overfished[-1] < 0.1 * overfished[0]
