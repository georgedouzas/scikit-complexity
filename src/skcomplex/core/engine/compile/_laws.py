"""Name the law classes the compiler treats alike and read law arguments.

The values the arguments of a law hold are read.
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

_FORMLESS_LAWS = (BoundaryLaw, CrossingLaw, BilinearLaw, FluxLaw, LinearProgramLaw)
"""Laws that leave the form of a system as the other laws decide it."""

_KERNEL_FREE_LAWS = (
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
"""Laws that their own passes lower, which no numeric kernel is written for."""

LATTICE_LAWS = (LocalRuleLaw, ArrayUpdateLaw, CascadeLaw, ExtremalLaw, SampleLaw)
"""Laws that update the sites of lattices and networks."""

_AGENT_LAWS = (RuleLaw,)
"""Laws that make an element an agent, run on the agent stepper."""
