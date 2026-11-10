"""Name the law classes the compiler treats alike and read law arguments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...components.laws import (
    ArrayUpdateLaw,
    BilinearLaw,
    BoundaryLaw,
    CascadeLaw,
    CrossingLaw,
    ExtremalLaw,
    FluxLaw,
    LinearProgramLaw,
    LocalRuleLaw,
    RuleLaw,
    SampleLaw,
    TransferLaw,
)

FORMLESS_LAWS = (BoundaryLaw, CrossingLaw, BilinearLaw, FluxLaw, LinearProgramLaw)
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
)
LATTICE_LAWS = (LocalRuleLaw, ArrayUpdateLaw, CascadeLaw, ExtremalLaw, SampleLaw)
AGENT_LAWS = (RuleLaw,)
