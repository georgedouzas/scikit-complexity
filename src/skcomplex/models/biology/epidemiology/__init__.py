"""Epidemiology: compartmental, network, metapopulation, and agent-based models of infection and recovery."""

from ._agents import Person, Traveller, Walker, agent_based_sir, network_sir, random_walker_epidemics
from ._compartments import Compartment, ImmunityLoss, Infection, Recovery
from ._factories import sir, sirs, sis
from ._networks import DegreeClass, Subpopulation, heterogeneous_mean_field_sis, metapopulation
from ._outbreaks import ImmunizingDegreeClass, heterogeneous_mean_field_sir
from ._recurrent import EpidemicCenter, Visits, delayed_sirs

__all__: list[str] = [
    'Compartment',
    'DegreeClass',
    'EpidemicCenter',
    'ImmunityLoss',
    'ImmunizingDegreeClass',
    'Infection',
    'Person',
    'Recovery',
    'Subpopulation',
    'Traveller',
    'Visits',
    'Walker',
    'agent_based_sir',
    'delayed_sirs',
    'heterogeneous_mean_field_sir',
    'heterogeneous_mean_field_sis',
    'metapopulation',
    'network_sir',
    'random_walker_epidemics',
    'sir',
    'sirs',
    'sis',
]
