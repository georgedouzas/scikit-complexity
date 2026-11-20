"""Name the law classes the compiler treats alike and read law arguments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...components.laws import (
    BoundaryLaw,
    EventLaw,
    LinearProgramLaw,
    LocalRuleLaw,
    ObjectiveLaw,
    RuleLaw,
    SampleLaw,
    TransferLaw,
)
from ..problems._bilinear_law import BilinearLaw
from ._array_update_law import ArrayUpdateLaw
from ._flux_law import _FluxLaw
from ._relaxation_law import RelaxationLaw

FORMLESS_LAWS = (BoundaryLaw, EventLaw, BilinearLaw, _FluxLaw, LinearProgramLaw, ObjectiveLaw)
KERNEL_FREE_LAWS = (
    TransferLaw,
    BilinearLaw,
    LocalRuleLaw,
    ArrayUpdateLaw,
    RelaxationLaw,
    SampleLaw,
    _FluxLaw,
    LinearProgramLaw,
    ObjectiveLaw,
)
LATTICE_LAWS = (LocalRuleLaw, ArrayUpdateLaw, RelaxationLaw, SampleLaw)
AGENT_LAWS = (RuleLaw,)
