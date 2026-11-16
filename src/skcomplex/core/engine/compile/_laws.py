"""Name the law classes the compiler treats alike and read law arguments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...components.laws import (
    ArrayUpdateLaw,
    BilinearLaw,
    BoundaryLaw,
    CascadeLaw,
    EventLaw,
    FluxLaw,
    LinearProgramLaw,
    LocalRuleLaw,
    ObjectiveLaw,
    RuleLaw,
    SampleLaw,
    TransferLaw,
)

FORMLESS_LAWS = (BoundaryLaw, EventLaw, BilinearLaw, FluxLaw, LinearProgramLaw, ObjectiveLaw)
KERNEL_FREE_LAWS = (
    TransferLaw,
    BilinearLaw,
    LocalRuleLaw,
    ArrayUpdateLaw,
    CascadeLaw,
    SampleLaw,
    FluxLaw,
    LinearProgramLaw,
    ObjectiveLaw,
)
LATTICE_LAWS = (LocalRuleLaw, ArrayUpdateLaw, CascadeLaw, SampleLaw)
AGENT_LAWS = (RuleLaw,)
