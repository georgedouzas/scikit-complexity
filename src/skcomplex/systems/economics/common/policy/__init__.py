"""Policy rules shared by the economics packages.

A policy rule acts on the system once per step, such as a central bank or a regulator.
"""

from ._policy_rule import PolicyRuleInteraction

__all__: list[str] = [
    'PolicyRuleInteraction',
]
