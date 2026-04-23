from __future__ import annotations

import math

WHEEL_RADIUS = 0.0205  # e-puck wheel radius [m]
AXLE_LENGTH = 0.052    # distance between left and right wheels [m]

# If the map rotates the wrong way when you turn the robot, set this to -1.
ROTATION_SIGN = 0.75


def calculate_diff_drive_velocities(vx: float, w: float) -> list[float]:
    """Return linear wheel velocities [left, right] for desired body motion.

    Compatible with set_velocity which converts to angular by dividing by WHEEL_RADIUS.
    """
    half_axle = AXLE_LENGTH / 2.0
    return [
        (vx - w * half_axle) / WHEEL_RADIUS,
        (vx + w * half_axle) / WHEEL_RADIUS,
    ]


class DiffDriveOdometry:
    """Dead-reckoning pose estimator for a differential-drive robot."""

    def __init__(self, left_encoder, right_encoder, compass = None) -> None:
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._prev_left: float | None = None
        self._prev_right: float | None = None
        self.left_encoder = left_encoder
        self.right_encoder = right_encoder
        self.compass = compass

    def update(self) -> None:
        """Update the pose estimate from current wheel encoder positions (radians)."""
        left_pos = self.left_encoder.getValue()
        right_pos = self.right_encoder.getValue()
        
        if self.compass is not None:
            compass_values = self.compass.getValues()
            self.theta = math.atan2(compass_values[1], compass_values[0]) * -1

        if self._prev_left is None or self._prev_right is None:
            self._prev_left = left_pos
            self._prev_right = right_pos
            return

        dl = (left_pos - self._prev_left) * WHEEL_RADIUS
        dr = (right_pos - self._prev_right) * WHEEL_RADIUS
        self._prev_left = left_pos
        self._prev_right = right_pos

        dc = (dl + dr) / 2.0
        
        if self.compass is None:
            dtheta = ROTATION_SIGN * (dr - dl) / AXLE_LENGTH
            self.theta += dtheta

        self.x += dc * math.cos(self.theta)
        self.y += dc * math.sin(self.theta)

    def get_pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.theta
