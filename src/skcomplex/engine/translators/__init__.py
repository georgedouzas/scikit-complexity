"""Translate model files of other tools into the source of model modules."""

from ._system_dynamics import SystemDynamicsTranslator

__all__: list[str] = ['SystemDynamicsTranslator']
