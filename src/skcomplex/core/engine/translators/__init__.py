"""Translate model files of other tools into the source of system modules.

A translator reads a model file, such as a stock and flow model of Vensim, and writes the Python source of a system
module that builds the same model on elements and interactions.
"""

from ._system_dynamics import SystemDynamicsTranslator

__all__: list[str] = [
    'SystemDynamicsTranslator',
]
