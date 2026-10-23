"""Compile the objectives of optimal control problems, which belong to the component that declares them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...laws import Law, Objective
from .._specs import KernelSpec


def _objective_owner(spec: KernelSpec, law: Law, agents: set[str]) -> str | None:
    """Return the component that declares an objective, or `None` for another law."""
    del agents
    return spec.owner if isinstance(law, Objective) else None


OWNER_RULES = (_objective_owner,)
