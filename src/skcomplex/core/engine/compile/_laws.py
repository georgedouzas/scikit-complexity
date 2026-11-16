"""Name the law classes the compiler treats alike and read law arguments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...components.laws import (
    ArrayUpdateLaw,
    BilinearLaw,
    BoundaryLaw,
    EventLaw,
    FluxLaw,
    LinearProgramLaw,
    LocalRuleLaw,
    ObjectiveLaw,
    RuleLaw,
    SampleLaw,
    TransferLaw,
)
from ._relaxation_law import RelaxationLaw

FORMLESS_LAWS = (BoundaryLaw, EventLaw, BilinearLaw, FluxLaw, LinearProgramLaw, ObjectiveLaw)
KERNEL_FREE_LAWS = (
    TransferLaw,
    BilinearLaw,
    LocalRuleLaw,
    ArrayUpdateLaw,
    RelaxationLaw,
    SampleLaw,
    FluxLaw,
    LinearProgramLaw,
    ObjectiveLaw,
)
LATTICE_LAWS = (LocalRuleLaw, ArrayUpdateLaw, RelaxationLaw, SampleLaw)
AGENT_LAWS = (RuleLaw,)
