"""Epidemiology: compartmental, network, metapopulation, and agent-based models of infection and recovery."""

from ._adaptive import (
    ActiveHost,
    Carrier,
    Contact,
    Expiry,
    Host,
    Rewiring,
    activity_driven_epidemics,
    adaptive_sis,
    quenched_mean_field_sis,
)
from ._agents import Person, Traveller, Walker, agent_based_sir, network_sir, random_walker_epidemics
from ._compartments import Compartment, ImmunityLoss, Infection, Recovery
from ._factories import sir, sirs, sis
from ._networks import DegreeClass, Subpopulation, heterogeneous_mean_field_sis, metapopulation
from ._outbreaks import ImmunizingDegreeClass, heterogeneous_mean_field_sir
from ._recurrent import EpidemicCenter, Visits, delayed_sirs
from ._vaccination import VaccinatedPopulation, sir_vaccination

__all__: list[str] = [
    'ActiveHost',
    'Carrier',
    'Compartment',
    'Contact',
    'DegreeClass',
    'EpidemicCenter',
    'Expiry',
    'Host',
    'ImmunityLoss',
    'ImmunizingDegreeClass',
    'Infection',
    'Person',
    'Recovery',
    'Rewiring',
    'Subpopulation',
    'Traveller',
    'VaccinatedPopulation',
    'Visits',
    'Walker',
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
