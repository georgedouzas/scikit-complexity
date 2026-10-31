"""Define alignment, which turns the members of a group towards the mean direction of the members around them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, atan2, cos, oo, pi, sin, sqrt, vector
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, UpdateLaw
from ..laws.utils import neighbors, uniform, where
from ..spaces import RadiusNeighborhood
from ._interaction import Interaction

SYMMETRIES = ('polar', 'nematic')


class AlignmentInteraction(Interaction):
    """Alignment of the members of a group with the members within a radius, every step of discrete time, in the plane.

    Every member takes the direction of the sum of the unit directions of the members within the `radius`, itself
    included, plus a uniform angle in `[-noise pi, noise pi]`. Its direction is its heading, an angle, or the direction
    of its velocity, whose length it keeps. The alignment has one of two symmetries:

    - `'polar'` sums the directions of the members as they are.
    - `'nematic'` flips every direction that points away from the member's own, so members align along an axis.

    Two optional zones add unit vectors to the sum. The repulsion zone points a member away from every other member
    within the distance `repulsion`, and the attraction zone points it towards every member farther than the `radius`
    and within the distance `attraction`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        members:
            The path of the group whose members align.

        symmetry:
            The symmetry of the alignment, `'polar'` or `'nematic'`.

        heading:
            The name of the state that holds the heading angle of a member, or `None` when the velocity is given.

        velocity:
            The name of the vector state that holds the velocity of a member, or `None` when the heading is given.

        position:
            The name of the vector state that holds the position of a member, which the zones read.

        radius:
            The distance within which members align.

        repulsion:
            The distance within which members repel each other, or `None` for no repulsion zone.

        attraction:
            The distance within which members farther than the radius attract each other, or `None` for no
            attraction zone.

        noise:
            The amplitude of the uniform angle added to the direction, as a fraction of `pi`.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.interactions import AlignmentInteraction
        >>> members = SimpleNamespace(theta=var('theta'), other=SimpleNamespace(theta=var('theta_j')))
        >>> alignment = AlignmentInteraction('alignment', members='flock', symmetry='nematic').check().prepare()
        >>> [type(law).__name__ for law in alignment.run(SimpleNamespace(members=members, noise=0)).laws_]
        ['UpdateLaw']
    """

    def __init__(
        self,
        label: str,
        *,
        members: Annotated[str, RoleSpec()],
        symmetry: str = 'polar',
        heading: str | None = 'theta',
        velocity: str | None = None,
        position: str = 'x',
        radius: float = 1.0,
        repulsion: float | None = None,
        attraction: float | None = None,
        noise: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label)
        self.members = members
        self.symmetry = symmetry
        self.heading = heading
        self.velocity = velocity
        self.position = position
        self.radius = radius
        self.repulsion = repulsion
        self.attraction = attraction
        self.noise = noise

    def _check_zones(self) -> None:
        """Raise an error for a radius that is not positive, or zones that do not surround the radius as they should."""
        if not isinstance(self.radius, int | float) or isinstance(self.radius, bool) or not self.radius > 0:
            error_msg = f'The radius of {self.label!r} is {self.radius!r}. Give a positive number.'
            raise ValidationError(error_msg)
        if self.repulsion is not None and not self.repulsion > 0:
            error_msg = f'The repulsion of {self.label!r} is {self.repulsion!r}. Give a positive distance, or None.'
            raise ValidationError(error_msg)
        if self.attraction is not None and not self.attraction > self.radius:
            error_msg = (
                f'The attraction of {self.label!r} is {self.attraction!r}. Give a distance larger than the radius '
                f'{self.radius!r}, or None.'
            )
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for an unknown symmetry, not exactly one of a heading and a velocity, or invalid zones.

        Raises:
            ValidationError:
                The symmetry is unknown, the heading and the velocity are both given or both missing, or a distance is
                invalid.
        """
        super()._check()
        if self.symmetry not in SYMMETRIES:
            error_msg = f'Unknown symmetry {self.symmetry!r} of {self.label!r}. Use one of {", ".join(SYMMETRIES)}.'
            raise ValidationError(error_msg)
        if (self.heading is None) == (self.velocity is None):
            error_msg = (
                f'{self.label!r} has the heading {self.heading!r} and the velocity {self.velocity!r}. Give the name '
                'of exactly one of them.'
            )
            raise ValidationError(error_msg)
        self._check_zones()

    def _get_direction(self, member: SimpleNamespace) -> tuple[Expression, Expression]:
        """Return the components of the unit direction of a member, from its heading or its velocity."""
        if self.heading is not None:
            angle = getattr(member, self.heading)
            return cos(angle), sin(angle)
        velocity = getattr(member, str(self.velocity))
        speed = sqrt(velocity.dot_product(velocity))
        return velocity[0] / speed, velocity[1] / speed

    def _get_alignment(self, members: SimpleNamespace) -> tuple[Expression, Expression]:
        """Return the sum of the directions of the members within the radius, flipped for a nematic symmetry."""
        own, other = self._get_direction(members), self._get_direction(members.other)
        side = SR(1) if self.symmetry == 'polar' else where(own[0] * other[0] + own[1] * other[1] >= 0, 1, -1)
        within = RadiusNeighborhood(self.radius, center=True)
        return neighbors(side * other[0], over=within), neighbors(side * other[1], over=within)

    def _get_zones(self, members: SimpleNamespace) -> tuple[Expression, Expression]:
        """Return the sum of the unit vectors of the repulsion and the attraction zones."""
        separation = getattr(members.other, self.position) - getattr(members, self.position)
        distance = sqrt(separation[0] ** 2 + separation[1] ** 2)
        along = (separation[0] / distance, separation[1] / distance)
        zones = [SR(0), SR(0)]
        for axis in range(2):
            if self.repulsion is not None:
                zones[axis] -= neighbors(along[axis], over=RadiusNeighborhood(self.repulsion))
            if self.attraction is not None:
                beyond = where(distance > self.radius, along[axis], 0)
                zones[axis] += neighbors(beyond, over=RadiusNeighborhood(self.attraction))
        return zones[0], zones[1]

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the update of the heading or the velocity of every member to its new direction.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `members` and its neighbour `members.other`.

        Returns:
            laws:
                The update law of the heading or the velocity.
        """
        members = namespace.members
        along_x, along_y = self._get_alignment(members)
        if self.repulsion is not None or self.attraction is not None:
            zone_x, zone_y = self._get_zones(members)
            along_x, along_y = along_x + zone_x, along_y + zone_y
        angle = atan2(along_y, along_x) + uniform('turn', -namespace.noise * pi, namespace.noise * pi)
        if self.heading is not None:
            return [UpdateLaw(getattr(members, self.heading), angle)]
        velocity = getattr(members, str(self.velocity))
        speed = sqrt(velocity.dot_product(velocity))
        return [UpdateLaw(velocity, speed * vector(SR, [cos(angle), sin(angle)]))]
