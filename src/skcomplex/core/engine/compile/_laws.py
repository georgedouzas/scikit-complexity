"""Name the law classes the compiler treats alike and read law arguments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...components.laws import EventLaw, ObjectiveLaw
from ...components.laws._boundary import _BoundaryLaw
from ...components.laws._linear_program import _LinearProgramLaw
from ...components.laws._local_rule import _LocalRuleLaw
from ...components.laws._rule import _RuleLaw
from ...components.laws._sample import _SampleLaw
from ..problems._bilinear_law import BilinearLaw
from ._array_update_law import ArrayUpdateLaw
from ._flux_law import _FluxLaw
from ._relaxation_law import RelaxationLaw

FORMLESS_LAWS = (_BoundaryLaw, EventLaw, BilinearLaw, _FluxLaw, _LinearProgramLaw, ObjectiveLaw)
KERNEL_FREE_LAWS = (
    BilinearLaw,
    _LocalRuleLaw,
    ArrayUpdateLaw,
    RelaxationLaw,
    _SampleLaw,
    _FluxLaw,
    _LinearProgramLaw,
    ObjectiveLaw,
)
LATTICE_LAWS = (_LocalRuleLaw, ArrayUpdateLaw, RelaxationLaw, _SampleLaw)
AGENT_LAWS = (_RuleLaw,)
