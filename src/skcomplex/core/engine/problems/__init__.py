"""Define what a system compiles to: a problem, its kernels, and run frames.

The bases, the problems of the shared compiler and of operator systems, and the frames stay public for observers,
analyses, and system packages that bring a form of their own. The other problems of the forms of the library are
private.
"""

from ._agent_snapshot_spec import AgentSnapshotSpec
from ._base import BaseKernel, BasePrinter, BaseSearch
from ._base_problem_spec import BaseProblemSpec
from ._base_record_spec import BaseRecordSpec
from ._base_structure_spec import BaseStructureSpec
from ._compiled_element_spec import CompiledElementSpec
from ._extremal_record_spec import ExtremalRecordSpec
from ._frame_spec import FrameSpec
from ._layout_spec import LayoutSpec
from ._local_operator_spec import LocalOperatorSpec
from ._network_snapshot_spec import NetworkSnapshotSpec
from ._operator_kernel import OperatorKernel
from ._operator_problem_spec import OperatorProblemSpec
from ._operator_spec import OperatorSpec
from ._patch_structure_spec import PatchStructureSpec
from ._problem_spec import ProblemSpec
from ._renewal_record_spec import RenewalRecordSpec
from ._run_state_spec import RunStateSpec
from ._word_structure_spec import WordStructureSpec

__all__: list[str] = [
    'AgentSnapshotSpec',
    'BaseKernel',
    'BasePrinter',
    'BaseProblemSpec',
    'BaseRecordSpec',
    'BaseSearch',
    'BaseStructureSpec',
    'CompiledElementSpec',
    'ExtremalRecordSpec',
    'FrameSpec',
    'LayoutSpec',
    'LocalOperatorSpec',
    'NetworkSnapshotSpec',
    'OperatorKernel',
    'OperatorProblemSpec',
    'OperatorSpec',
    'PatchStructureSpec',
    'ProblemSpec',
    'RenewalRecordSpec',
    'RunStateSpec',
    'WordStructureSpec',
]
