"""Define laws, the equations and rules that change the values of elements.

A law says how one value changes, such as `RateLaw(s.ball.x.diff(s.t, 2), -s.g)` for a falling ball, the next value
after a step, a random jump, or an equation the compiler solves. Interactions return laws in `laws_`, and the compiler
turns them into a problem that a solver runs. The specs here hold the settings that laws and their rules read.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseLaw
from ._call_spec import CallSpec
from ._constraint import ConstraintLaw
from ._dataset_spec import DatasetSpec
from ._draw_spec import DrawSpec
from ._event import EventLaw
from ._jump import JumpLaw
from ._linear_objective_spec import LinearObjectiveSpec
from ._linear_rule_spec import LinearRuleSpec
from ._move_spec import MoveSpec
from ._noise import NoiseLaw
from ._objective import ObjectiveLaw
from ._objective_spec import ObjectiveSpec
from ._outer_totalistic_spec import OuterTotalisticSpec
from ._partner_spec import PartnerSpec
from ._rate import RateLaw
from ._requirement_spec import RequirementSpec
from ._sampler_spec import SamplerSpec
from ._table_spec import TableSpec
from ._topology_request_spec import EDGE_KINDS, TopologyRequestSpec
from ._totalistic_spec import TotalisticSpec
from ._update import UpdateLaw

__all__: list[str] = [
    'EDGE_KINDS',
    'BaseLaw',
    'CallSpec',
    'ConstraintLaw',
    'DatasetSpec',
    'DrawSpec',
    'EventLaw',
    'JumpLaw',
    'LinearObjectiveSpec',
    'LinearRuleSpec',
    'MoveSpec',
    'NoiseLaw',
    'ObjectiveLaw',
    'ObjectiveSpec',
    'OuterTotalisticSpec',
    'PartnerSpec',
    'RateLaw',
    'RequirementSpec',
    'SamplerSpec',
    'TableSpec',
    'TopologyRequestSpec',
    'TotalisticSpec',
    'UpdateLaw',
]
