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
from ._diffusive_limit_spec import DiffusiveLimitSpec
from ._expression_printer import ExpressionPrinter
from ._frame_spec import FrameSpec
from ._kernel_spec import KernelSpec
from ._kinetic_problem_spec import KineticProblemSpec
from ._layout_spec import LayoutSpec
from ._learning_problem_spec import LearningProblemSpec
from ._linear_term_spec import LinearTermSpec
from ._loop_term_spec import LoopTermSpec
from ._lowered_kernels_spec import LoweredKernelsSpec
from ._network_snapshot_spec import NetworkSnapshotSpec
from ._operator_kernel import OperatorKernel
from ._operator_problem_spec import OperatorProblemSpec
from ._partition_spec import PartitionSpec
from ._problem_spec import ProblemSpec
from ._reference_spec import ReferenceSpec
from ._run_state_spec import RunStateSpec
from ._transfer_network_spec import TransferNetworkSpec

__all__: list[str] = [
    'AgentSnapshotSpec',
    'BaseKernel',
    'BasePrinter',
    'BaseProblemSpec',
    'BaseRecordSpec',
    'BaseSearch',
    'CompiledElementSpec',
    'DiffusiveLimitSpec',
    'ExpressionPrinter',
    'FrameSpec',
    'KernelSpec',
    'KineticProblemSpec',
    'LayoutSpec',
    'LearningProblemSpec',
    'LinearTermSpec',
    'LoopTermSpec',
    'LoweredKernelsSpec',
    'NetworkSnapshotSpec',
    'OperatorKernel',
    'OperatorProblemSpec',
    'PartitionSpec',
    'ProblemSpec',
    'ReferenceSpec',
    'RunStateSpec',
    'TransferNetworkSpec',
]
