# TODO: Move & rename this file for common usage (not specific for PDM)
# TODO: Remove @classmethod + @property decorators (deprecated in Python 3.13)
"""
Pdm enums implementation.
"""

from enum import IntEnum


class StateIndex:
    """Index mapping for array representation of ego states."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _VELOCITY_X = 3
    _VELOCITY_Y = 4
    _ACCELERATION_X = 5
    _ACCELERATION_Y = 6
    _STEERING_ANGLE = 7
    _STEERING_RATE = 8
    _ANGULAR_VELOCITY = 9
    _ANGULAR_ACCELERATION = 10

    @classmethod
    def size(cls):
        """
        Size."""
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        """
        X."""
        return cls._X

    @classmethod
    @property
    def Y(cls):
        """
        Y."""
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        """
        Heading."""
        return cls._HEADING

    @classmethod
    @property
    def VELOCITY_X(cls):
        """
        Velocity y."""
        return cls._VELOCITY_X

    @classmethod
    @property
    def VELOCITY_Y(cls):
        """
        Acceleration x."""
        return cls._VELOCITY_Y

    @classmethod
    @property
    def ACCELERATION_X(cls):
        """
        Steering angle."""
        return cls._ACCELERATION_X

    @classmethod
    @property
    def ACCELERATION_Y(cls):
        """
        Steering rate."""
        return cls._ACCELERATION_Y

    @classmethod
    @property
    def STEERING_ANGLE(cls):
        """
        Angular velocity."""
        return cls._STEERING_ANGLE

    @classmethod
    @property
    def STEERING_RATE(cls):
        """
        Point."""
        return cls._STEERING_RATE

    @classmethod
    @property
    def ANGULAR_VELOCITY(cls):
        """
        State se2."""
        return cls._ANGULAR_VELOCITY

    @classmethod
    @property
    def ANGULAR_ACCELERATION(cls):
        """
        Velocity 2d."""
        return cls._ANGULAR_ACCELERATION

    @classmethod
    @property
    def POINT(cls):
        """
        Acceleration 2d."""
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)

    @classmethod
    @property
    def VELOCITY_2D(cls):
        """State se2."""
        # assumes velocity X, Y have subsequent indices
        return slice(cls._VELOCITY_X, cls._VELOCITY_Y + 1)

    @classmethod
    @property
    def ACCELERATION_2D(cls):
        """Velocity 2d."""
        # assumes acceleration X, Y have subsequent indices
        return slice(cls._ACCELERATION_X, cls._ACCELERATION_Y + 1)


class PointIndex(IntEnum):
    """Index mapping for (x,y) arrays."""

    X = 0
    Y = 1


class SE2Index(IntEnum):
    """Index mapping for state se2 (x,y,θ) arrays."""

    X = 0
    Y = 1
    HEADING = 2


class DynamicStateIndex(IntEnum):
    """Index mapping for dynamic car state (output of controller)."""

    ACCELERATION_X = 0
    STEERING_RATE = 1


class StateIDMIndex(IntEnum):
    """Index mapping for IDM states."""

    PROGRESS = 0
    VELOCITY = 1


class LeadingAgentIndex(IntEnum):
    """Index mapping for leading agent state (for IDM policies)."""

    PROGRESS = 0
    VELOCITY = 1
    LENGTH_REAR = 2


class BBCoordsIndex(IntEnum):
    """Index mapping for corners and center of bounding boxes."""

    FRONT_LEFT = 0
    REAR_LEFT = 1
    REAR_RIGHT = 2
    FRONT_RIGHT = 3
    CENTER = 4


class EgoAreaIndex(IntEnum):
    """Index mapping for area of ego agent (used in PDMScorer)."""

    MULTIPLE_LANES = 0
    NON_DRIVABLE_AREA = 1
    ONCOMING_TRAFFIC = 2


class MultiMetricIndex(IntEnum):
    """Index mapping multiplicative metrics (used in PDMScorer)."""

    NO_COLLISION = 0
    DRIVABLE_AREA = 1
    TRAFFIC_LIGHT_COMPLIANCE = 2
    DRIVING_DIRECTION = 3


class WeightedMetricIndex(IntEnum):
    """Index mapping weighted metrics (used in PDMScorer)."""

    PROGRESS = 0
    TTC = 1
    LANE_KEEPING = 2
    HISTORY_COMFORT = 3
    TWO_FRAME_EXTENDED_COMFORT = 4
