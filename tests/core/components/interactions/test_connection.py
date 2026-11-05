"""Test the `_connection` module of `skcomplex.core.components.interactions`."""

from typing import Annotated, ClassVar

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import ConnectionInteraction
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ConnectorSpec, ParameterSpec, StateSpec

PIN = ConnectorSpec('pin', potential='v', flow='i')
CHEMICAL = ConnectorSpec('chemical', potential='mu', flow='n')
ALGEBRAIC = StateSpec(algebraic=True)


class SourceResistorElement(Element):
    """Resistor from one pin to a fixed voltage."""

    ports: ClassVar = {'a': PIN}

    def __init__(
        self,
        label: str,
        *,
        a_v: Annotated[float, ALGEBRAIC] = 0.0,
        a_i: Annotated[float, ALGEBRAIC] = 0.0,
        R: Annotated[float, ParameterSpec()] = 1.0,
        e: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.a_v = a_v
        self.a_i = a_i
        self.R = R
        self.e = e

    def _get_laws(self, s):
        """Return Ohm's law from the pin to the fixed voltage."""
        return [EquationLaw(s.a_v - s.e, s.R * s.a_i)]


class ReservoirElement(Element):
    """Reservoir with a port of a connector the test defines, whose flow follows its potential difference."""

    ports: ClassVar = {'a': CHEMICAL}

    def __init__(
        self,
        label: str,
        *,
        a_mu: Annotated[float, ALGEBRAIC] = 0.0,
        a_n: Annotated[float, ALGEBRAIC] = 0.0,
        level: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.a_mu = a_mu
        self.a_n = a_n
        self.level = level

    def _get_laws(self, s):
        """Return the flow of the reservoir."""
        return [EquationLaw(s.a_n, s.a_mu - s.level)]


def _run(system):
    """Return the trajectory of a system over one time unit."""
    return Simulation(system, dt=0.5).prepare().run(until=1.0).trajectory_


def test_connection_balances_the_currents_of_three_resistors():
    """Test Kirchhoff's current law at a node where three resistors meet."""
    system = System(
        elements=[
            SourceResistorElement('r1', R=1.0, e=3.0),
            SourceResistorElement('r2', R=2.0, e=0.0),
            SourceResistorElement('r3', R=4.0, e=-1.0),
        ],
        interactions=[ConnectionInteraction('node', members=['r1', 'r2', 'r3'], ports=['a', 'a', 'a'])],
    )
    trajectory = _run(system)
    currents = [trajectory.values(f'{name}.a_i').ravel() for name in ('r1', 'r2', 'r3')]
    voltage = (3.0 / 1.0 - 1.0 / 4.0) / (1 / 1.0 + 1 / 2.0 + 1 / 4.0)
    assert np.allclose(sum(currents), 0.0, atol=1e-12)
    assert np.allclose(trajectory.values('r1.a_v').ravel(), voltage)
    assert np.allclose(trajectory.values('r3.a_v').ravel(), voltage)
    assert np.allclose(currents[0], voltage - 3.0)


def test_connection_reads_a_connector_the_user_defines():
    """Test that a connection reads the potential and the flow of a connector the user defines."""
    system = System(
        elements=[ReservoirElement('first', level=1.0), ReservoirElement('second', level=3.0)],
        interactions=[ConnectionInteraction('node', members=['first', 'second'], ports=['a', 'a'])],
    )
    trajectory = _run(system)
    assert np.allclose(trajectory.values('first.a_mu').ravel(), 2.0)
    assert np.allclose(trajectory.values('second.a_mu').ravel(), 2.0)
    assert np.allclose(trajectory.values('first.a_n').ravel(), 1.0)
    assert np.allclose(trajectory.values('second.a_n').ravel(), -1.0)


def test_connection_rejects_members_and_ports_of_different_lengths():
    """Test that a connection names one port per member."""
    with pytest.raises(ValidationError, match='Name one port per member'):
        ConnectionInteraction('node', members=['first', 'second'], ports=['a']).prepare()
