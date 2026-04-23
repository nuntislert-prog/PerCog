from __future__ import annotations

import math

import numpy as np


class OccupancyGrid:
    """Log-odds occupancy grid built from 2-D lidar scans."""

    FREE_DELTA = -0.4
    OCC_DELTA = 0.9
    CLAMP_MIN = -5.0
    CLAMP_MAX = 5.0

    def __init__(
        self,
        world_min: tuple[float, float] = (-2.0, -2.0),
        world_max: tuple[float, float] = (2.0, 2.0),
        resolution: float = 0.01,
    ) -> None:
        self.world_min = world_min
        self.world_max = world_max
        self.resolution = resolution
        self.width = int(math.ceil((world_max[0] - world_min[0]) / resolution))
        self.height = int(math.ceil((world_max[1] - world_min[1]) / resolution))
        self.grid = np.zeros((self.height, self.width), dtype=np.float64)

    def _world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        gx = int(round((wx - self.world_min[0]) / self.resolution))
        gy = int(round((wy - self.world_min[1]) / self.resolution))
        # Webot coordinates are inverted
        gy = self.height - gy
        return gx, gy

    def _in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height

    def update(
        self,
        pose: tuple[float, float, float],
        ranges: list[float],
        fov: float,
        max_range: float,
    ) -> None:
        """Integrate one lidar scan into the grid.

        Args:
            pose: (x, y, theta) in metres / radians.
            ranges: flat list of range readings.
            fov: horizontal field of view in radians.
            max_range: sensor maximum range; readings at or above this are skipped.
        """
        rx, ry, rtheta = pose
        num_points = len(ranges)
        if num_points == 0:
            return

        ox, oy = self._world_to_grid(rx, ry)

        for i, r in enumerate(ranges):
            if r <= 0 or math.isnan(r) or math.isinf(r):
                continue

            hit = r < max_range

            angle = rtheta + fov / 2.0 - i * (fov / num_points)

            end_x = rx + r * math.cos(angle)
            end_y = ry + r * math.sin(angle)
            ex, ey = self._world_to_grid(end_x, end_y)

            for bx, by in self._bresenham(ox, oy, ex, ey):
                if not self._in_bounds(bx, by):
                    continue
                if bx == ex and by == ey:
                    break
                self.grid[by, bx] += self.FREE_DELTA
                self.grid[by, bx] = max(self.grid[by, bx], self.CLAMP_MIN)

            if hit and self._in_bounds(ex, ey):
                self.grid[ey, ex] += self.OCC_DELTA
                self.grid[ey, ex] = min(self.grid[ey, ex], self.CLAMP_MAX)

    def render(self) -> np.ndarray:
        """Return a uint8 image: 255 = occupied (black), 128 = unknown, 0 = free (white)."""
        prob = 1.0 / (1.0 + np.exp(-self.grid))
        img = (prob * 255).astype(np.uint8)
        return img

    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points
