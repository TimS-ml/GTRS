"""
Enumeration definitions for NAVSIM data indexing and types.

This module defines integer enumerations for accessing structured data arrays
including SE(2) states, bounding boxes, and lidar point clouds. These enums
provide type-safe, named access to array indices.
"""

from enum import IntEnum


class SceneFrameType(IntEnum):
    """
    Enumeration for scene frame types.

    Distinguishes between original logged data frames and synthetically
    generated frames in simulation scenarios.
    """

    ORIGINAL = 0  # Frame from original logged data
    SYNTHETIC = 1  # Synthetically generated frame


class StateSE2Index(IntEnum):
    """
    Enumeration for indexing SE(2) state arrays.

    SE(2) represents 2D poses with position (x, y) and heading (orientation).
    This enum provides named access to components of SE(2) state arrays.

    Components:
        X: X-coordinate position
        Y: Y-coordinate position
        HEADING: Orientation angle in radians
    """

    _X = 0
    _Y = 1
    _HEADING = 2

    @classmethod
    def size(cls):
        """
        Get the total number of components in an SE(2) state.

        Returns:
            int: Number of state components (3 for SE(2): x, y, heading).
        """
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        """Index for X coordinate in SE(2) arrays."""
        return cls._X

    @classmethod
    @property
    def Y(cls):
        """Index for Y coordinate in SE(2) arrays."""
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        """Index for heading/orientation in SE(2) arrays."""
        return cls._HEADING

    @classmethod
    @property
    def POINT(cls):
        """Slice for accessing 2D point (x, y) from SE(2) arrays."""
        # Assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        """Slice for accessing full SE(2) state (x, y, heading) from arrays."""
        # Assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)


class BoundingBoxIndex(IntEnum):
    """
    Enumeration for indexing 3D bounding box arrays.

    Provides named access to components of 3D bounding boxes including position
    (x, y, z), dimensions (length, width, height), and orientation (heading).
    Used for object detection and tracking in autonomous driving scenarios.
    """

    _X = 0
    _Y = 1
    _Z = 2
    _LENGTH = 3
    _WIDTH = 4
    _HEIGHT = 5
    _HEADING = 6

    @classmethod
    def size(cls):
        """
        Get the total number of components in a bounding box.

        Returns:
            int: Number of bounding box components (7: x, y, z, length, width, height, heading).
        """
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        """Index for X coordinate (east direction) in bounding box arrays."""
        return cls._X

    @classmethod
    @property
    def Y(cls):
        """Index for Y coordinate (north direction) in bounding box arrays."""
        return cls._Y

    @classmethod
    @property
    def Z(cls):
        """Index for Z coordinate (up direction) in bounding box arrays."""
        return cls._Z

    @classmethod
    @property
    def LENGTH(cls):
        """Index for length dimension (along vehicle's longitudinal axis) in bounding box arrays."""
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        """Index for width dimension (along vehicle's lateral axis) in bounding box arrays."""
        return cls._WIDTH

    @classmethod
    @property
    def HEIGHT(cls):
        """Index for height dimension (vertical extent) in bounding box arrays."""
        return cls._HEIGHT

    @classmethod
    @property
    def HEADING(cls):
        """Index for heading/yaw angle in bounding box arrays."""
        return cls._HEADING

    @classmethod
    @property
    def POINT2D(cls):
        """Slice for accessing 2D position (x, y) from bounding box arrays."""
        # Assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def POSITION(cls):
        """Slice for accessing 3D position (x, y, z) from bounding box arrays."""
        # Assumes X, Y, Z have subsequent indices
        return slice(cls._X, cls._Z + 1)

    @classmethod
    @property
    def DIMENSION(cls):
        """Slice for accessing dimensions (length, width, height) from bounding box arrays."""
        # Assumes LENGTH, WIDTH, HEIGHT have subsequent indices
        return slice(cls._LENGTH, cls._HEIGHT + 1)


class LidarIndex(IntEnum):
    """
    Enumeration for indexing lidar point cloud arrays.

    Provides named access to components of lidar point clouds including 3D position
    (x, y, z), intensity, ring index, and point ID. Used for processing lidar sensor
    data in autonomous driving applications.
    """

    _X = 0
    _Y = 1
    _Z = 2
    _INTENSITY = 3
    _RING = 4
    _ID = 5

    @classmethod
    def size(cls):
        """
        Get the total number of components in a lidar point.

        Returns:
            int: Number of lidar point components (6: x, y, z, intensity, ring, id).
        """
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        """Index for X coordinate in lidar point cloud arrays."""
        return cls._X

    @classmethod
    @property
    def Y(cls):
        """Index for Y coordinate in lidar point cloud arrays."""
        return cls._Y

    @classmethod
    @property
    def Z(cls):
        """Index for Z coordinate in lidar point cloud arrays."""
        return cls._Z

    @classmethod
    @property
    def INTENSITY(cls):
        """Index for intensity/reflectance value in lidar point cloud arrays."""
        return cls._INTENSITY

    @classmethod
    @property
    def RING(cls):
        """Index for ring/laser index in lidar point cloud arrays."""
        return cls._RING

    @classmethod
    @property
    def ID(cls):
        """Index for point ID in lidar point cloud arrays."""
        return cls._ID

    @classmethod
    @property
    def POINT2D(cls):
        """Slice for accessing 2D position (x, y) from lidar point cloud arrays."""
        # Assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def POSITION(cls):
        """Slice for accessing 3D position (x, y, z) from lidar point cloud arrays."""
        # Assumes X, Y, Z have subsequent indices
        return slice(cls._X, cls._Z + 1)
