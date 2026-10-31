"""Epidemiology: compartmental, network, metapopulation, and agent-based models of infection and recovery."""

from ._adaptive import (
    ActiveHostElement,
    CarrierElement,
    HostElement,
    activity_driven_epidemics,
    adaptive_sis,
    quenched_mean_field_sis,
)
from ._agents import (
    PersonElement,
    TravellerElement,
    WalkerElement,
    agent_based_sir,
    network_sir,
    random_walker_epidemics,
)
from ._compartments import ImmunityLossInteraction, InfectionInteraction, RecoveryInteraction
from ._factories import sir, sirs, sis
from ._networks import DegreeClassElement, SubpopulationElement, heterogeneous_mean_field_sis, metapopulation
from ._outbreaks import ImmunizingDegreeClassElement, heterogeneous_mean_field_sir
from ._recurrent import EpidemicCenterElement, delayed_sirs
from ._vaccination import VaccinatedPopulationElement, sir_vaccination

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
