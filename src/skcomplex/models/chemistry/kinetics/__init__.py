"""Chemical kinetics: reactions, oscillating reactors, and excitable media."""

from ._cyclic import CyclicCell, cyclic_cellular_automaton
from ._hodgepodge import HodgepodgeCell, hodgepodge
from ._nanoparticles import nanoparticle_corona
from ._oscillators import (
    BrusselatorReaction,
    CubicAutocatalysis,
    Oregonator,
    brusselator_ode,
    cubic_autocatalysis,
    oregonator,
)
from ._reactions import Binding, Dissociation, Schlogl, Species, chemical_langevin_equation, michaelis_menten
from ._reactors import CatalystPellet, PulsedElectrode, catalyst_pellet, pulsed_electrochemical_reactor

__all__: list[str] = [
    'Binding',
    'BrusselatorReaction',
    'CatalystPellet',
    'CubicAutocatalysis',
    'CyclicCell',
    'Dissociation',
    'HodgepodgeCell',
    'Oregonator',
    'PulsedElectrode',
    'Schlogl',
    'Species',
    'brusselator_ode',
    'catalyst_pellet',
    'chemical_langevin_equation',
    'cubic_autocatalysis',
    'cyclic_cellular_automaton',
    'hodgepodge',
    'michaelis_menten',
    'nanoparticle_corona',
    'oregonator',
    'pulsed_electrochemical_reactor',
]
