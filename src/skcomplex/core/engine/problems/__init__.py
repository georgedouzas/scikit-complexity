"""Define what a system compiles to: a problem, its kernels, and run frames.

A problem holds the values of a system in flat vectors and the compiled functions of its laws, such as the one state
`'population.x'` of a logistic growth. The bases, the problems of the general compiler and of operator systems, and the
frames stay public for observers, analyses, and system packages that bring a form of their own.
"""

from ._agent_snapshot_spec import AgentSnapshotSpec
from ._base import BaseKernel, BasePrinter, BaseSearch
from ._base_problem_spec import BaseProblemSpec
from ._base_record_spec import BaseRecordSpec
from ._compiled_element_spec import CompiledElementSpec
from ._expression_printer import ExpressionPrinter
from ._frame_spec import FrameSpec
from ._layout_spec import LayoutSpec
from ._local_operator_spec import LocalOperatorSpec
from ._network_snapshot_spec import NetworkSnapshotSpec
from ._operator_kernel import OperatorKernel
from ._operator_problem_spec import OperatorProblemSpec
from ._operator_spec import OperatorSpec
from ._problem_spec import ProblemSpec
from ._renewal_record_spec import RenewalRecordSpec
from ._run_state_spec import RunStateSpec

__all__: list[str] = [
    'AgentSnapshotSpec',
    'BaseKernel',
    'BasePrinter',
    'BaseProblemSpec',
    'BaseRecordSpec',
    'BaseSearch',
    'CompiledElementSpec',
    'ExpressionPrinter',
    'FrameSpec',
    'LayoutSpec',
    'LocalOperatorSpec',
    'NetworkSnapshotSpec',
    'OperatorKernel',
    'OperatorProblemSpec',
    'OperatorSpec',
    'ProblemSpec',
    'RenewalRecordSpec',
    'RunStateSpec',
]
