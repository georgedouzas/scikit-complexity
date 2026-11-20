"""Define laws, the rules by which states change.

A law is one equation or rule for the states it targets, such as a rate, the next value at a step, a random jump, or an
algebraic equation. Elements, interactions, and systems return laws, and the compiler lowers them to a problem.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseLaw
from ._boundary import BoundaryLaw
from ._call_spec import CallSpec
from ._constraint import ConstraintLaw
from ._dataset_spec import DatasetSpec
from ._draw_spec import DrawSpec
from ._event import EventLaw
from ._jump import JumpLaw
from ._linear_objective_spec import LinearObjectiveSpec
from ._linear_program import LinearProgramLaw
from ._linear_rule_spec import LinearRuleSpec
from ._local_rule import LocalRuleLaw
from ._move_spec import MoveSpec
from ._noise import NoiseLaw
from ._objective import ObjectiveLaw
from ._objective_spec import ObjectiveSpec
from ._outer_totalistic_spec import OuterTotalisticSpec
from ._partner_spec import PartnerSpec
from ._rate import RateLaw
from ._requirement_spec import RequirementSpec
from ._rule import RuleLaw
from ._sample import SampleLaw
from ._sampler_spec import SamplerSpec
from ._table_spec import TableSpec
from ._topology_request_spec import EDGE_KINDS, TopologyRequestSpec
from ._totalistic_spec import TotalisticSpec
from ._transfer import Sink, Source, TransferLaw
from ._transition import TransitionLaw
from ._update import UpdateLaw

__all__: list[str] = [
    'EDGE_KINDS',
    'BaseLaw',
    'BoundaryLaw',
    'CallSpec',
    'ConstraintLaw',
    'DatasetSpec',
    'DrawSpec',
    'EventLaw',
    'JumpLaw',
    'LinearObjectiveSpec',
    'LinearProgramLaw',
    'LinearRuleSpec',
    'LocalRuleLaw',
    'MoveSpec',
    'NoiseLaw',
    'ObjectiveLaw',
    'ObjectiveSpec',
    'OuterTotalisticSpec',
    'PartnerSpec',
    'RateLaw',
    'RequirementSpec',
    'RuleLaw',
    'SampleLaw',
    'SamplerSpec',
    'Sink',
    'Source',
    'TableSpec',
    'TopologyRequestSpec',
    'TotalisticSpec',
    'TransferLaw',
    'TransitionLaw',
    'UpdateLaw',
]
