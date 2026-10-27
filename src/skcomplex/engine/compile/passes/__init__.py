"""Hold the passes of a compilation, the steps of the shared core that turn a system into a problem."""

from ._fields import ConservedNoisePass, FieldFunctionsPass, FieldJacobianPass, FieldProblemPass, LowerFieldsPass
from ._kinetics import KineticProblem
from ._lattices import LatticeProblemPass
from ._quantum import (
    DEFAULT_MAX_DIMENSION,
    HilbertSpaceError,
    QuantumCompiler,
    QuantumProblem,
    QuantumTerm,
    contains_quantum,
)
from ._rewriting import (
    CompiledGraph,
    CompiledPatch,
    RewritingCompiler,
    RewritingProblem,
    compile_rewriting,
    structure_of,
)
from ._topology import (
    MaskRequestsPass,
    RequestJumpsPass,
    SplitRequestsPass,
    TopologyFunctionsPass,
    TopologyProblemPass,
    TowardsPass,
)

__all__: list[str] = [
    'DEFAULT_MAX_DIMENSION',
    'CompiledGraph',
    'CompiledPatch',
    'ConservedNoisePass',
    'FieldFunctionsPass',
    'FieldJacobianPass',
    'FieldProblemPass',
    'HilbertSpaceError',
    'KineticProblem',
    'LatticeProblemPass',
    'LowerFieldsPass',
    'MaskRequestsPass',
    'QuantumCompiler',
    'QuantumProblem',
    'QuantumTerm',
    'RequestJumpsPass',
    'RewritingCompiler',
    'RewritingProblem',
    'SplitRequestsPass',
    'TopologyFunctionsPass',
    'TopologyProblemPass',
    'TowardsPass',
    'compile_rewriting',
    'contains_quantum',
    'structure_of',
]
