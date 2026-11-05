"""Translate system files of other tools into the source of system modules."""

from ._system_dynamics import SystemDynamicsTranslator

__all__: list[str] = ['SystemDynamicsTranslator']
