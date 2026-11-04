"""Name the law classes the compiler treats alike, and read the values the arguments of a law hold."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...core.laws import (
    BilinearLaw,
    BoundaryLaw,
    CascadeLaw,
    CrossingLaw,
    DriveLaw,
    EncounterLaw,
    ExtremalLaw,
    FluxLaw,
    LinearProgramLaw,
    LocalRuleLaw,
    RuleLaw,
    SampleLaw,
    TransferLaw,
)

_FORMLESS_LAWS = (BoundaryLaw, CrossingLaw, EncounterLaw, BilinearLaw, FluxLaw, LinearProgramLaw)
"""Laws that leave the form of a system as the other laws decide it."""

_KERNEL_FREE_LAWS = (
    TransferLaw,
    EncounterLaw,
    BilinearLaw,
    LocalRuleLaw,
    DriveLaw,
    CascadeLaw,
    ExtremalLaw,
    SampleLaw,
    FluxLaw,
    LinearProgramLaw,
)
"""Laws that their own passes lower, which no numeric kernel is written for."""

LATTICE_LAWS = (LocalRuleLaw, DriveLaw, CascadeLaw, ExtremalLaw, SampleLaw)
"""Laws that update the sites of lattices and networks."""

_AGENT_LAWS = (RuleLaw,)
"""Laws that make an element an agent, run on the agent stepper."""
