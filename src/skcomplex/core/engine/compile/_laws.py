"""Name the law classes the compiler treats alike and read law arguments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...components.laws import (
    ArrayUpdateLaw,
    BoundaryLaw,
    CascadeLaw,
    EventLaw,
    ExtremalLaw,
    FluxLaw,
    LinearProgramLaw,
    LocalRuleLaw,
    ObjectiveLaw,
    RuleLaw,
    SampleLaw,
    TransferLaw,
)
from ..problems._bilinear_law import BilinearLaw

FORMLESS_LAWS = (BoundaryLaw, EventLaw, BilinearLaw, FluxLaw, LinearProgramLaw, ObjectiveLaw)
KERNEL_FREE_LAWS = (
    TransferLaw,
    BilinearLaw,
    LocalRuleLaw,
    ArrayUpdateLaw,
    CascadeLaw,
    ExtremalLaw,
    SampleLaw,
    FluxLaw,
    LinearProgramLaw,
    ObjectiveLaw,
)
LATTICE_LAWS = (LocalRuleLaw, ArrayUpdateLaw, CascadeLaw, ExtremalLaw, SampleLaw)
AGENT_LAWS = (RuleLaw,)
