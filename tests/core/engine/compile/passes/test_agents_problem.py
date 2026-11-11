"""Test the `_agents_problem` module of `skcomplex.core.engine.compile.passes`."""

from typing import Annotated

import pytest
from sage.all__sagemath_symbolics import CC

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.errors import BackendError
from skcomplex.core.specs import StateSpec, VisualSpec


def test_mesa_backend_rejects_complex_agent_state():
    """Test that an agent type with a complex state names the state and the Mesa backend."""

    class WalkerElement(Element):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            z: Annotated[complex, StateSpec(domain=CC)] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.z = z

        def _get_laws(self, s):
            return [RuleLaw(lambda agent, system: None)]

    with pytest.raises(BackendError, match=r'walkers.*complex state z.*Mesa'):
        PassCompiler().prepare(CompositeSystem(elements=[GroupElement('walkers', element=WalkerElement, size=3)])).run()
