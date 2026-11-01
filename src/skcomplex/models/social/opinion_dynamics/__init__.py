"""Opinion dynamics: voting, coevolving networks, persuasion, bounded confidence, conventions, and diffusion."""

from ._adopter_view import AdopterView
from ._axelrod import (
    CulturalInfluenceInteraction,
    VillageElement,
    axelrod,
    cultural_traits,
    culture_observables,
    largest_region,
)
from ._believer_view import BelieverView
from ._bounded_confidence import (
    ConfidentElement,
    DiscussantElement,
    EncountersInteraction,
    deffuant,
    hegselmann_krause,
)
from ._citizen_view import CitizenView
from ._coevolution import (
    ClubMemberElement,
    CoevolvingVoterElement,
    SeparationInteraction,
    TieElement,
    coevolving_voter,
    join_a_club,
)
from ._cognition import ConversationsInteraction, ThinkerElement, cognitive_consistency, settle
from ._complex_agent_view import ComplexAgentView
from ._confidence_coupling import ConfidenceCouplingInteraction
from ._conventions import NamerElement, NamingGamesInteraction, naming_game
from ._diffusion import AdopterElement, AdoptionInteraction, BelieverElement, influentials, threshold_adoption
from ._discussant_view import DiscussantView
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
from ._namer_view import NamerView
from ._persuader_view import PersuaderView
from ._persuasion import (
    CitizenElement,
    DiscussionsInteraction,
    InfluenceeElement,
    PersuaderElement,
    PersuasionInteraction,
    SocialImpactInteraction,
    galam_majority,
    social_impact,
    sznajd,
)
from ._random_pairs import RandomPairsInteraction
from ._standardization import ComplexAgentElement, StandardizationInteraction, standardized_agents
from ._thinker_view import ThinkerView
from ._variant_voter_view import VariantVoterView
from ._variant_voting import VariantVotingInteraction
from ._village_view import VillageView
from ._voter_imitation import VoterImitationInteraction
from ._voter_view import VoterView
from ._voters import (
    MajorityVoterElement,
    VariantVoterElement,
    VoterElement,
    majority_rule,
    voter_model,
    voter_variants,
)

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
    'OpinionEncountersInteraction',
    'PersuaderElement',
    'PersuaderView',
    'PersuasionInteraction',
    'RandomPairsInteraction',
    'SeparationInteraction',
    'SocialImpactInteraction',
    'StandardizationInteraction',
    'ThinkerElement',
    'ThinkerView',
    'TieElement',
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
    'hegselmann_krause',
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
