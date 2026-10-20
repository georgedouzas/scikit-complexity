"""Opinion dynamics: voting, persuasion, bounded confidence, conventions, cognition, and diffusion."""

from ._axelrod import CultureCensus, Village, axelrod, cultural_traits, largest_region
from ._bounded_confidence import Discussant, Encounters, deffuant
from ._cognition import Conversations, Thinker, cognitive_consistency, settle
from ._conventions import Namer, NamingGames, naming_game
from ._diffusion import Adopter, Adoption, Believer, influentials, threshold_adoption
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
    'largest_region',
    'majority_rule',
    'naming_game',
    'settle',
    'social_impact',
    'standardized_agents',
    'sznajd',
    'threshold_adoption',
    'voter_model',
    'voter_variants',
]
