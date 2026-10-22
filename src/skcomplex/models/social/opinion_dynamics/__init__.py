"""Opinion dynamics: voting, persuasion, bounded confidence, conventions, cognition, and diffusion."""

from ._axelrod import CultureCensus, Village, axelrod, cultural_traits, largest_region
from ._bounded_confidence import Discussant, Encounters, deffuant
from ._cognition import Conversations, Thinker, cognitive_consistency, settle
from ._conventions import Namer, NamingGames, naming_game
from ._diffusion import Adopter, Adoption, Believer, influentials, threshold_adoption
from ._kinetic_opinion import (
    OpinionEncounters,
    OpinionLevel,
    kinetic_opinion,
    mean_opinion,
    opinion_grid,
    opinion_observables,
    opinion_variance,
    persuasion_table,
)
from ._persuasion import (
    Citizen,
    Discussions,
    Influencee,
    Persuader,
    SocialImpact,
    galam_majority,
    social_impact,
    sznajd,
)
from ._standardization import ComplexAgent, Standardization, standardized_agents
from ._voters import MajorityVoter, VariantVoter, Voter, majority_rule, voter_model, voter_variants

__all__: list[str] = [
    'Adopter',
    'Adoption',
    'Believer',
    'Citizen',
    'ComplexAgent',
    'Conversations',
    'CultureCensus',
    'Discussant',
    'Discussions',
    'Encounters',
    'Influencee',
    'MajorityVoter',
    'Namer',
    'NamingGames',
    'OpinionEncounters',
    'OpinionLevel',
    'Persuader',
    'SocialImpact',
    'Standardization',
    'Thinker',
    'VariantVoter',
    'Village',
    'Voter',
    'axelrod',
    'cognitive_consistency',
    'cultural_traits',
    'deffuant',
    'galam_majority',
    'influentials',
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
