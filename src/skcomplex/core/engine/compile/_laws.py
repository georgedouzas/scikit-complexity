"""Name the law classes the compiler treats alike and read law arguments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...components.laws import (
    ArrayUpdateLaw,
    BilinearLaw,
    BoundaryLaw,
    CascadeLaw,
    EventLaw,
    ExtremalLaw,
    LinearProgramLaw,
    LocalRuleLaw,
    ObjectiveLaw,
    RuleLaw,
    SampleLaw,
    TransferLaw,
)
from ._flux_law import _FluxLaw

FORMLESS_LAWS = (BoundaryLaw, EventLaw, BilinearLaw, _FluxLaw, LinearProgramLaw, ObjectiveLaw)
KERNEL_FREE_LAWS = (
    TransferLaw,
    BilinearLaw,
    LocalRuleLaw,
    ArrayUpdateLaw,
    CascadeLaw,
    ExtremalLaw,
    SampleLaw,
    _FluxLaw,
    LinearProgramLaw,
    ObjectiveLaw,
)
LATTICE_LAWS = (LocalRuleLaw, ArrayUpdateLaw, CascadeLaw, ExtremalLaw, SampleLaw)
AGENT_LAWS = (RuleLaw,)
