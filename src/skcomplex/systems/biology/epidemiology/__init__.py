"""Epidemiology: compartmental, network, metapopulation, and agent-based systems of infection and recovery."""

from ._active_host import ActiveHostElement
from ._adaptive import activity_driven_epidemics, adaptive_sis, quenched_mean_field_sis
from ._agents import agent_based_sir, network_sir, random_walker_epidemics
from ._carrier import CarrierElement
from ._degree_class import DegreeClassElement
from ._factories import sir, sirs, sis
from ._host import HostElement
from ._immunity_loss import ImmunityLossInteraction
from ._infection import InfectionInteraction
from ._networks import heterogeneous_mean_field_sis, metapopulation
from ._outbreaks import ImmunizingDegreeClassElement, heterogeneous_mean_field_sir
from ._person import PersonElement
from ._recovery import RecoveryInteraction
from ._recurrent import EpidemicCenterElement, delayed_sirs
from ._spreading import SpreadingInteraction
from ._subpopulation import SubpopulationElement
from ._traveller import TravellerElement
from ._vaccination import VaccinatedPopulationElement, sir_vaccination
from ._walker import WalkerElement

__all__: list[str] = [
    'ActiveHostElement',
    'CarrierElement',
    'DegreeClassElement',
    'EpidemicCenterElement',
    'HostElement',
    'ImmunityLossInteraction',
    'ImmunizingDegreeClassElement',
    'InfectionInteraction',
    'PersonElement',
    'RecoveryInteraction',
    'SpreadingInteraction',
    'SubpopulationElement',
    'TravellerElement',
    'VaccinatedPopulationElement',
    'WalkerElement',
    'activity_driven_epidemics',
    'adaptive_sis',
    'agent_based_sir',
    'delayed_sirs',
    'heterogeneous_mean_field_sir',
    'heterogeneous_mean_field_sis',
    'metapopulation',
    'network_sir',
    'quenched_mean_field_sis',
    'random_walker_epidemics',
    'sir',
    'sir_vaccination',
    'sirs',
    'sis',
]
