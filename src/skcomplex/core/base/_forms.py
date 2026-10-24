"""Define the mathematical forms a system compiles to."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ._terms import Term


class Form(Term):
    """Mathematical form of a compiled system, a term of an open vocabulary that extensions add forms to.

    Args:
        name:
            The name of the form in upper case.

        value:
            The description of the form.

    Examples:
        >>> from skcomplex.core.base import Form
        >>> Form.SDE.value
        'stochastic differential equations'
    """

    ODE: ClassVar['Form']
    FRACTIONAL: ClassVar['Form']
    SDE: ClassVar['Form']
    JUMP: ClassVar['Form']
    DISCRETE: ClassVar['Form']
    AGENT: ClassVar['Form']
    LATTICE: ClassVar['Form']
    MONTE_CARLO: ClassVar['Form']
    REWRITING: ClassVar['Form']
    RENEWAL: ClassVar['Form']
    COUPLED: ClassVar['Form']
    QUANTUM: ClassVar['Form']


Form.ODE = Form('ODE', 'ordinary differential equations')
Form.FRACTIONAL = Form('FRACTIONAL', 'fractional differential equations')
Form.SDE = Form('SDE', 'stochastic differential equations')
Form.JUMP = Form('JUMP', 'jump process')
Form.DISCRETE = Form('DISCRETE', 'discrete map')
Form.AGENT = Form('AGENT', 'agent-based model')
Form.LATTICE = Form('LATTICE', 'lattice dynamics')
Form.MONTE_CARLO = Form('MONTE_CARLO', 'Monte Carlo sampling')
Form.REWRITING = Form('REWRITING', 'rewriting system')
Form.RENEWAL = Form('RENEWAL', 'renewal process')
Form.COUPLED = Form('COUPLED', 'co-simulation')
Form.QUANTUM = Form('QUANTUM', 'quantum dynamics')
