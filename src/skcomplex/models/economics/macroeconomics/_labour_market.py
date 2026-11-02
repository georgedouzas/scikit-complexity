"""Labour market in which unemployed adults, in random order, take the best-paying vacancy within their sight."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from ....core.interactions import MatchingInteraction
from ....engine.solvers import ModelView
from ....specs import (
    MatchSpec,
)
from ._business_view import BusinessView
from ._settler_view import SettlerView


class LabourMarketInteraction(MatchingInteraction[ModelView]):
    """Labour market in which unemployed adults, in random order, take the best-paying vacancy within their sight.

    A matching of the people, who request a job, with the firms, which provide their vacancies: a person whose
    `hireable` limit admits a job takes the firm of the highest wage within its sight that still has a vacancy, and
    moves to it. Build it with
    `PartnerSpec(sight, 'score', score=LabourMarketInteraction.score, clearing='provider')`.

    Examples:
        >>> from skcomplex.specs import PartnerSpec
        >>> from skcomplex.models.economics.macroeconomics import LabourMarketInteraction
        >>> LabourMarketInteraction(
        ...     'jobs',
        ...     requesters='people',
        ...     providers='businesses',
        ...     rule=PartnerSpec(5.0, 'score', score=LabourMarketInteraction.score, clearing='provider'),
        ...     limit=('hireable', 'limit'),
        ...     quantity=('quantity', 'vacancies'),
        ...     on_match=LabourMarketInteraction.hire,
        ... ).check().providers
        'businesses'
    """

    @staticmethod
    def score(person: object, firm: object) -> float:
        """Return the wage of a firm, the score by which a person prefers it.

        Args:
            person:
                The person who seeks a job.

            firm:
                The firm with a vacancy.

        Returns:
            score:
                The wage of the firm.
        """
        del person
        return float(firm.wage) if isinstance(firm, BusinessView) else -math.inf

    @staticmethod
    def hire(match: MatchSpec, model: ModelView) -> None:
        """Employ a matched person at its firm and move it there.

        Args:
            match:
                The match of a person with a firm.

            model:
                The model view of the market.
        """
        del model
        person, firm = match.requester, match.provider
        if isinstance(person, SettlerView) and isinstance(firm, BusinessView):
            person.employer = float(firm.id)
            person.move_to(firm.location)
