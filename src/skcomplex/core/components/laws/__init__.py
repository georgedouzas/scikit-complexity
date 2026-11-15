"""Define laws, the rules by which states change.

A law is one equation or rule for the states it targets, such as a rate, the next value at a step, a random jump, or an
algebraic equation. Elements, interactions, and systems return laws, and the compiler lowers them to a problem.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._array_update import ArrayUpdateLaw
from ._base import BaseLaw
from ._bilinear import BilinearLaw
from ._boundary import BoundaryLaw
from ._cascade import CascadeLaw
from ._constraint import ConstraintLaw
from ._dissipator import DissipatorLaw
from ._event import EventLaw
from ._extremal import ExtremalLaw
from ._flux import FluxLaw
from ._generator import GeneratorLaw
from ._gradient_step import GradientStepLaw
from ._jump import JumpLaw
from ._linear_program import LinearProgramLaw
from ._local_rule import LocalRuleLaw
from ._match import MatchLaw
from ._noise import NoiseLaw
from ._production import ProductionLaw
from ._rate import RateLaw
from ._rule import RuleLaw
from ._sample import SampleLaw
from ._substitution import SubstitutionLaw
from ._transfer import Sink, Source, TransferLaw
from ._transition import TransitionLaw
from ._update import UpdateLaw

__all__: list[str] = [
    'ArrayUpdateLaw',
    'BaseLaw',
    'BilinearLaw',
    'BoundaryLaw',
    'CascadeLaw',
    'ConstraintLaw',
    'DissipatorLaw',
    'EventLaw',
    'ExtremalLaw',
    'FluxLaw',
    'GeneratorLaw',
    'GradientStepLaw',
    'JumpLaw',
    'LinearProgramLaw',
    'LocalRuleLaw',
    'MatchLaw',
    'NoiseLaw',
    'ProductionLaw',
    'RateLaw',
    'RuleLaw',
    'SampleLaw',
    'Sink',
    'Source',
    'SubstitutionLaw',
    'TransferLaw',
    'TransitionLaw',
    'UpdateLaw',
]
