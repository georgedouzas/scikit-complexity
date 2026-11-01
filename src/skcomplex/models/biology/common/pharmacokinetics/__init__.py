"""Pharmacokinetics shared by the biology packages: the input of drugs in schedules of infusions."""

from ._dosing import DosingInteraction

__all__: list[str] = [
    'DosingInteraction',
]
