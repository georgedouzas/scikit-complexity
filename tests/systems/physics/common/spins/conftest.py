"""Share the spin holder and the runs of the spin sampling tests."""

from typing import Annotated

import numpy as np
import numpy.typing as npt

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import BaseInteraction
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


class SiteElement(Element):
    """Site of a lattice with a spin, a color, an angle, a coupling, a field, and a temperature."""

    def __init__(
        self,
        label: str,
        *,
        spin: Annotated[float, StateSpec()] = 1.0,
        color: Annotated[float, StateSpec()] = 0.0,
        theta: Annotated[float, StateSpec()] = 0.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        h: Annotated[float, ParameterSpec()] = 0.0,
        temperature: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.spin = spin
        self.color = color
        self.theta = theta
        self.J = J
        self.h = h
        self.temperature = temperature


def _run(interaction: BaseInteraction, state: str, values: dict[str, object]) -> npt.NDArray[np.float64]:
    """Return the values of a state of a 12 by 12 lattice of sites after 100 sweeps of a sampling interaction."""
    sites = LatticeElement('sites', element=SiteElement, space=LatticeSpace((12, 12)), values=values)
    system = System(elements=[sites], interactions=[interaction], time=DiscreteTime())
    trajectory = Simulation(system, random_state=0).prepare().run(until=100).trajectory_
    return np.asarray(trajectory.values(f'sites.{state}'))
