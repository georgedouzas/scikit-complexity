"""Chemical kinetics: reactions, oscillating reactors, and excitable media."""

from ._cyclic import CyclicCellComponent, cyclic_cellular_automaton
from ._hodgepodge import HodgepodgeCellComponent, hodgepodge
from ._nanoparticles import nanoparticle_corona
from ._oscillators import (
    BrusselatorReactionComponent,
    CubicAutocatalysisComponent,
    OregonatorComponent,
    brusselator_ode,
    cubic_autocatalysis,
    oregonator,
)
from ._reactions import (
    BindingInteraction,
    DissociationInteraction,
    SchloglComponent,
    SpeciesComponent,
    chemical_langevin_equation,
    chemical_langevin_network,
    michaelis_menten,
)
from ._reactors import (
    CatalystPelletComponent,
    PulsedElectrodeComponent,
    catalyst_pellet,
    pulsed_electrochemical_reactor,
)

__all__: list[str] = [
    'BindingInteraction',
    'BrusselatorReactionComponent',
    'CatalystPelletComponent',
    'CubicAutocatalysisComponent',
    'CyclicCellComponent',
    'DissociationInteraction',
    'HodgepodgeCellComponent',
    'OregonatorComponent',
    'PulsedElectrodeComponent',
    'SchloglComponent',
    'SpeciesComponent',
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
