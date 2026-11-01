"""Define laws, the rules by which states change."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseLaw
from ._boundary import BoundaryLaw
from ._cascade import CascadeLaw
from ._crossing import CrossingLaw
from ._destruction import DestructionLaw
from ._dissipator import DissipatorLaw
from ._drive import DriveLaw
from ._emission import EmissionLaw
from ._encounter import EncounterLaw
from ._equation import EquationLaw
from ._extremal import ExtremalLaw
from ._flux import FluxLaw
from ._fractional_noise import FractionalNoiseLaw
from ._generator import GeneratorLaw
from ._gradient_step import GradientStepLaw
from ._jump import JumpLaw
from ._local_rule import LocalRuleLaw
from ._match import MatchLaw
from ._noise import NoiseLaw
from ._proliferation import ProliferationLaw
from ._rate import RateLaw
from ._rule import RuleLaw
from ._sample import SampleLaw
from ._transfer import Sink, Source, TransferLaw
from ._transition import TransitionLaw
from ._update import UpdateLaw

__all__: list[str] = [
    'BaseLaw',
    'BoundaryLaw',
    'CascadeLaw',
    'CrossingLaw',
    'DestructionLaw',
    'DissipatorLaw',
    'DriveLaw',
    'EmissionLaw',
    'EncounterLaw',
    'EquationLaw',
    'ExtremalLaw',
    'FluxLaw',
    'FractionalNoiseLaw',
    'GeneratorLaw',
    'GradientStepLaw',
    'JumpLaw',
    'LocalRuleLaw',
    'MatchLaw',
    'NoiseLaw',
    'ProliferationLaw',
    'RateLaw',
    'RuleLaw',
    'SampleLaw',
    'Sink',
    'Source',
    'TransferLaw',
    'TransitionLaw',
    'UpdateLaw',
]
