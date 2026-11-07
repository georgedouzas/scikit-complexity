"""Epidemiology: compartmental, network, metapopulation, and agent-based systems of infection and recovery."""

from ..common.networks import HostElement, SpreadingInteraction
from ._adaptive import (
    ActiveHostElement,
    CarrierElement,
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
from ._cities import ColocationInteraction, CommuterElement, episims_city
from ._concurrency import PartnershipEpidemicElement, pair_formation_sis, watts_may_concurrency
from ._factories import (
    ImmunityLossInteraction,
    InfectionInteraction,
    RecoveryInteraction,
    sir,
    sirs,
    sis,
    zombie_outbreak,
)
from ._networks import DegreeClassElement, SubpopulationElement, heterogeneous_mean_field_sis, metapopulation
from ._outbreaks import ImmunizingDegreeClassElement, heterogeneous_mean_field_sir
from ._records import PathogenElement, ResidentElement, ResistanceInfectionInteraction, record_dynamics_epidemic
from ._recurrent import EpidemicCenterElement, delayed_sirs
from ._vaccination import VaccinatedPopulationElement, sir_vaccination

__all__: list[str] = [
    'ActiveHostElement',
    'CarrierElement',
    'ColocationInteraction',
    'CommuterElement',
    'DegreeClassElement',
    'EpidemicCenterElement',
    'HostElement',
    'ImmunityLossInteraction',
    'ImmunizingDegreeClassElement',
    'InfectionInteraction',
    'PartnershipEpidemicElement',
    'PathogenElement',
    'PersonElement',
    'RecoveryInteraction',
    'ResidentElement',
    'ResistanceInfectionInteraction',
    'SpreadingInteraction',
    'SubpopulationElement',
    'TravellerElement',
    'VaccinatedPopulationElement',
    'WalkerElement',
    'activity_driven_epidemics',
    'adaptive_sis',
    'agent_based_sir',
    'delayed_sirs',
    'episims_city',
    'heterogeneous_mean_field_sir',
    'heterogeneous_mean_field_sis',
    'metapopulation',
    'network_sir',
    'pair_formation_sis',
    'quenched_mean_field_sis',
    'random_walker_epidemics',
    'record_dynamics_epidemic',
    'sir',
    'sir_vaccination',
    'sirs',
    'sis',
    'watts_may_concurrency',
    'zombie_outbreak',
]
