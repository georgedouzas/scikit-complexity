"""Opinion dynamics: voting, coevolving networks, persuasion, bounded confidence, conventions, and diffusion."""

from ._adopter import AdopterElement
from ._adopter_view import AdopterView
from ._adoption import AdoptionInteraction
from ._axelrod import axelrod, culture_observables, largest_region
from ._believer import BelieverElement
from ._believer_view import BelieverView
from ._bounded_confidence import deffuant, hegselmann_krause
from ._citizen import CitizenElement
from ._citizen_view import CitizenView
from ._club_member import ClubMemberElement
from ._coevolution import coevolving_voter, join_a_club
from ._coevolving_voter import CoevolvingVoterElement
from ._cognition import cognitive_consistency
from ._complex_agent import ComplexAgentElement
from ._complex_agent_view import ComplexAgentView
from ._confidence_coupling import ConfidenceCouplingInteraction
from ._confident import ConfidentElement
from ._conventions import naming_game
from ._conversations import ConversationsInteraction, settle
from ._cultural_influence import CulturalInfluenceInteraction, cultural_traits
from ._diffusion import influentials, threshold_adoption
from ._discussant import DiscussantElement
from ._discussant_view import DiscussantView
from ._discussions import DiscussionsInteraction
from ._encounters import EncountersInteraction
from ._gauge_agents import ConnectionElement, PhaseAgentElement, gauge_network_agents
from ._hypergraph_consensus import hypergraph_consensus
from ._influencee import InfluenceeElement
from ._influencee_view import InfluenceeView
from ._kinetic_opinion import (
    OpinionEncountersInteraction,
    kinetic_opinion,
    mean_opinion,
    opinion_grid,
    opinion_observables,
    opinion_variance,
    persuasion_table,
)
from ._majority_voter import MajorityVoterElement
from ._namer import NamerElement
from ._namer_view import NamerView
from ._naming_games import NamingGamesInteraction
from ._opinion import OpinionElement
from ._persuader import PersuaderElement
from ._persuader_view import PersuaderView
from ._persuasion import PersuasionInteraction, galam_majority, social_impact, sznajd
from ._random_pairs import RandomPairsInteraction
from ._schedule import ScheduleElement
from ._separation import SeparationInteraction
from ._social_impact import SocialImpactInteraction
from ._standardization import StandardizationInteraction, standardized_agents
from ._thinker import ThinkerElement
from ._thinker_view import ThinkerView
from ._three_body_consensus import ThreeBodyConsensusInteraction
from ._tie import TieElement
from ._transported_coupling import TransportedCouplingInteraction
from ._variant_voter import VariantVoterElement
from ._variant_voter_view import VariantVoterView
from ._variant_voting import VariantVotingInteraction
from ._village import VillageElement
from ._village_view import VillageView
from ._voter import VoterElement
from ._voter_imitation import VoterImitationInteraction
from ._voter_view import VoterView
from ._voters import majority_rule, voter_model, voter_variants

__all__: list[str] = [
    'AdopterElement',
    'AdopterView',
    'AdoptionInteraction',
    'BelieverElement',
    'BelieverView',
    'CitizenElement',
    'CitizenView',
    'ClubMemberElement',
    'CoevolvingVoterElement',
    'ComplexAgentElement',
    'ComplexAgentView',
    'ConfidenceCouplingInteraction',
    'ConfidentElement',
    'ConnectionElement',
    'ConversationsInteraction',
    'CulturalInfluenceInteraction',
    'DiscussantElement',
    'DiscussantView',
    'DiscussionsInteraction',
    'EncountersInteraction',
    'InfluenceeElement',
    'InfluenceeView',
    'MajorityVoterElement',
    'NamerElement',
    'NamerView',
    'NamingGamesInteraction',
    'OpinionElement',
    'OpinionEncountersInteraction',
    'PersuaderElement',
    'PersuaderView',
    'PersuasionInteraction',
    'PhaseAgentElement',
    'RandomPairsInteraction',
    'ScheduleElement',
    'SeparationInteraction',
    'SocialImpactInteraction',
    'StandardizationInteraction',
    'ThinkerElement',
    'ThinkerView',
    'ThreeBodyConsensusInteraction',
    'TieElement',
    'TransportedCouplingInteraction',
    'VariantVoterElement',
    'VariantVoterView',
    'VariantVotingInteraction',
    'VillageElement',
    'VillageView',
    'VoterElement',
    'VoterImitationInteraction',
    'VoterView',
    'axelrod',
    'coevolving_voter',
    'cognitive_consistency',
    'cultural_traits',
    'culture_observables',
    'deffuant',
    'galam_majority',
    'gauge_network_agents',
    'hegselmann_krause',
    'hypergraph_consensus',
    'influentials',
    'join_a_club',
    'kinetic_opinion',
    'largest_region',
    'majority_rule',
    'mean_opinion',
    'naming_game',
    'opinion_grid',
    'opinion_observables',
    'opinion_variance',
    'persuasion_table',
    'settle',
    'social_impact',
    'standardized_agents',
    'sznajd',
    'threshold_adoption',
    'voter_model',
    'voter_variants',
]
