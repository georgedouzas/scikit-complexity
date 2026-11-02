"""Chemical kinetics: reactions, oscillating reactors, and excitable media."""

from ._catalyst_pellet import CatalystPelletElement
from ._cubic_autocatalysis import CubicAutocatalysisElement
from ._cyclic import CyclicCellElement, cyclic_cellular_automaton
from ._hodgepodge import HodgepodgeCellElement, hodgepodge
from ._nanoparticles import nanoparticle_corona
from ._oregonator import OregonatorElement
from ._oscillators import brusselator_ode, cubic_autocatalysis, oregonator
from ._pulsed_electrode import PulsedElectrodeElement
from ._reactions import (
    SchloglElement,
    chemical_langevin_equation,
    chemical_langevin_network,
    michaelis_menten,
)
from ._reactors import catalyst_pellet, pulsed_electrochemical_reactor

__all__: list[str] = [
    'CatalystPelletElement',
    'CubicAutocatalysisElement',
    'CyclicCellElement',
    'HodgepodgeCellElement',
    'OregonatorElement',
    'PulsedElectrodeElement',
    'SchloglElement',
    'brusselator_ode',
    'catalyst_pellet',
    'chemical_langevin_equation',
    'chemical_langevin_network',
    'cubic_autocatalysis',
    'cyclic_cellular_automaton',
    'hodgepodge',
    'michaelis_menten',
    'nanoparticle_corona',
    'oregonator',
    'pulsed_electrochemical_reactor',
]
