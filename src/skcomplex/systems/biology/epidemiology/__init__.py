"""Epidemiology systems of infection and recovery.

The systems are compartmental, network, metapopulation, and agent-based.
"""

from ..common.networks import HostElement, SpreadingInteraction
from ._adaptive import (
    ActiveHostElement,
    ActivityDrivenEpidemicsSystem,
    AdaptiveSISSystem,
    QuenchedMeanFieldSISSystem,
)
from ._agents import (
    AgentBasedSIRSystem,
    NetworkSIRSystem,
    PersonElement,
    RandomWalkerEpidemicsSystem,
    TravellerElement,
    WalkerElement,
)
from ._cities import ColocationInteraction, CommuterElement, EpisimsCitySystem
from ._concurrency import PairFormationSISSystem, PartnershipEpidemicElement, WattsMayConcurrencySystem
from ._differentiable_covid import (
    CaseCountElement,
    CaseCountInteraction,
    CovidAgentElement,
    CovidTransmissionInteraction,
    DifferentiableCovidAbmSystem,
)
from ._factories import (
    ImmunityLossInteraction,
    InfectionInteraction,
    RecoveryInteraction,
    SIRSSystem,
    SIRSystem,
    SISSystem,
    ZombieOutbreakSystem,
)
from ._networks import DegreeClassElement, HeterogeneousMeanFieldSISSystem, MetapopulationSystem
from ._outbreaks import HeterogeneousMeanFieldSIRSystem, ImmunizingDegreeClassElement
from ._records import PathogenElement, RecordDynamicsEpidemicSystem, ResidentElement, ResistanceInfectionInteraction
from ._recurrent import DelayedSIRSSystem, EpidemicCenterElement
from ._vaccination import SIRVaccinationSystem

__all__: list[str] = [
    'ActiveHostElement',
    'ActivityDrivenEpidemicsSystem',
    'AdaptiveSISSystem',
    'AgentBasedSIRSystem',
    'CaseCountElement',
    'CaseCountInteraction',
    'ColocationInteraction',
    'CommuterElement',
    'CovidAgentElement',
    'CovidTransmissionInteraction',
    'DegreeClassElement',
    'DelayedSIRSSystem',
    'DifferentiableCovidAbmSystem',
    'EpidemicCenterElement',
    'EpisimsCitySystem',
    'HeterogeneousMeanFieldSIRSystem',
    'HeterogeneousMeanFieldSISSystem',
    'HostElement',
    'ImmunityLossInteraction',
    'ImmunizingDegreeClassElement',
    'InfectionInteraction',
    'MetapopulationSystem',
    'NetworkSIRSystem',
    'PairFormationSISSystem',
    'PartnershipEpidemicElement',
    'PathogenElement',
    'PersonElement',
    'QuenchedMeanFieldSISSystem',
    'RandomWalkerEpidemicsSystem',
    'RecordDynamicsEpidemicSystem',
    'RecoveryInteraction',
    'ResidentElement',
    'ResistanceInfectionInteraction',
    'SIRSSystem',
    'SIRSystem',
    'SIRVaccinationSystem',
    'SISSystem',
    'SpreadingInteraction',
    'TravellerElement',
    'WalkerElement',
    'WattsMayConcurrencySystem',
    'ZombieOutbreakSystem',
]
