"""Name the law classes the compiler treats alike and read law arguments.

The values the arguments of a law hold are read.

`FORMLESS_LAWS` leave the form of a system as the other laws decide it, `KERNEL_FREE_LAWS` are lowered by their own
passes with no numeric kernel, `LATTICE_LAWS` update the sites of lattices and networks, and `AGENT_LAWS` make an
element an agent that runs on the agent stepper.
"""

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
