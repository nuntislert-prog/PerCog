from __future__ import annotations
 
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
 
import numpy as np
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Tuneable constants  (change these to experiment)
# ──────────────────────────────────────────────────────────────────────────────
 
# Keyframe creation: new node only when the robot has moved at least…
KF_DIST_M   = 0.08   # …0.08 m  (translational threshold)
KF_ANGLE_RAD = 0.15  # …0.15 rad (~8.6°) (rotational threshold)
 
# Loop-closure: search among nodes farther than this many hops away
LC_MIN_HOPS    = 10    # ignore recent neighbours (already connected by odom edges)
LC_SCORE_THRESH = 0.55 # normalised cross-correlation threshold [0..1]
LC_MAX_CANDIDATES = 6  # check at most this many spatial neighbours per keyframe
 
# Graph optimisation
OPT_ITERATIONS = 15    # Gauss-Newton iterations per optimisation call
 
# Occupancy grid rasterisation (same log-odds constants as original)
FREE_DELTA = -0.4
OCC_DELTA  =  0.9
CLAMP_MIN  = -5.0
CLAMP_MAX  =  5.0
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────
 
@dataclass
class Node:
    """One keyframe in the pose graph."""
    idx:    int
    x:      float
    y:      float
    theta:  float
    ranges: List[float]       # raw lidar scan stored for loop-closure & rasterisation
    fov:    float
    max_range: float
 
 
@dataclass
class Edge:
    """A relative-pose constraint between two nodes."""
    i: int          # index of source node
    j: int          # index of target node
    dx: float       # relative translation x  (in frame of node i)
    dy: float       # relative translation y
    dtheta: float   # relative rotation
    # Information matrix weight (scalar; full 3×3 would be overkill here)
    omega: float = 1.0
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Helper: pose composition and inversion
# ──────────────────────────────────────────────────────────────────────────────
 
def _compose(p1: Tuple[float,float,float],
             p2: Tuple[float,float,float]) -> Tuple[float,float,float]:
    """Return p1 ⊕ p2  (apply relative pose p2 in frame of p1)."""
    x1, y1, t1 = p1
    dx, dy, dt = p2
    c, s = math.cos(t1), math.sin(t1)
    return (x1 + c*dx - s*dy,
            y1 + s*dx + c*dy,
            t1 + dt)
 
 
def _relative_pose(p_from: Tuple[float,float,float],
                   p_to:   Tuple[float,float,float]) -> Tuple[float,float,float]:
    """Return the relative pose of p_to as seen from p_from."""
    x1, y1, t1 = p_from
    x2, y2, t2 = p_to
    c, s = math.cos(t1), math.sin(t1)
    dx =  c*(x2-x1) + s*(y2-y1)
    dy = -s*(x2-x1) + c*(y2-y1)
    dt = _wrap_angle(t2 - t1)
    return dx, dy, dt
 
 
def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Scan-matching helper: 1-D circular cross-correlation for loop closure
# ──────────────────────────────────────────────────────────────────────────────
 
def _scan_correlation(scan_a: List[float], scan_b: List[float],
                      max_range: float) -> float:
    """
    Normalised cross-correlation between two lidar range scans.
    Returns a score in [0, 1].  Higher = more similar.
 
    Both scans are converted to binary (hit / no-hit) vectors first so
    that absolute distance differences don't dominate.
    """
    n = min(len(scan_a), len(scan_b))
    if n == 0:
        return 0.0
 
    def _binarise(s):
        return np.array([1.0 if (r < max_range and not math.isnan(r)
                                 and not math.isinf(r)) else 0.0
                         for r in s[:n]])
 
    va = _binarise(scan_a)
    vb = _binarise(scan_b)
 
    # Circular cross-correlation via FFT – finds best rotational alignment
    fa = np.fft.rfft(va)
    fb = np.fft.rfft(vb)
    cc = np.fft.irfft(fa * np.conj(fb))
    best = float(np.max(cc))
 
    # Normalise: perfect match → 1.0
    norm = math.sqrt(float(np.sum(va)) * float(np.sum(vb)))
    if norm < 1e-6:
        return 0.0
    return min(best / norm, 1.0)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Bresenham line (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────
 
def _bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int,int]]:
    points: List[Tuple[int,int]] = []
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 <  dx:
            err += dx; y0 += sy
    return points
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Graph optimiser  (Gauss-Newton, scalar information weights)
# ──────────────────────────────────────────────────────────────────────────────
 
class _GraphOptimiser:
    """
    Minimises  Σ  eᵢⱼᵀ Ωᵢⱼ eᵢⱼ   over all node poses.
 
    The error vector for edge (i→j) is:
        e = measurement – prediction
          = [dx_meas, dy_meas, dθ_meas] – relative_pose(xi, xj)
 
    We fix node 0 (the origin) so the system is not under-determined.
    Solved with sparse Gauss-Newton:  H Δx = –b,  H built from Jacobians.
    """
 
    def optimise(self, nodes: List[Node], edges: List[Edge],
                 iterations: int = OPT_ITERATIONS) -> None:
        n = len(nodes)
        if n < 2:
            return
 
        # Pack node poses into a flat array  [x0,y0,t0, x1,y1,t1, …]
        poses = np.array([[nd.x, nd.y, nd.theta] for nd in nodes],
                         dtype=np.float64)  # shape (n, 3)
 
        for _ in range(iterations):
            # 3n × 3n sparse system built as dense (fine for < 500 nodes)
            H = np.zeros((3*n, 3*n), dtype=np.float64)
            b = np.zeros( 3*n,       dtype=np.float64)
 
            for edge in edges:
                i, j = edge.i, edge.j
                xi = tuple(poses[i])
                xj = tuple(poses[j])
 
                # Predicted relative pose
                pred = _relative_pose(xi, xj)   # type: ignore[arg-type]
                meas = (edge.dx, edge.dy, edge.dtheta)
 
                # Error
                ex = meas[0] - pred[0]
                ey = meas[1] - pred[1]
                et = _wrap_angle(meas[2] - pred[2])
                e  = np.array([ex, ey, et])
 
                # Jacobians ∂e/∂xi and ∂e/∂xj  (analytical, 3×3 each)
                ti = xi[2]
                c  = math.cos(ti)
                s  = math.sin(ti)
                dx = xj[0] - xi[0]
                dy = xj[1] - xi[1]
 
                # ∂(relative_pose)/∂xi
                Ji = np.array([
                    [-c, -s,  -s*dx + c*dy],   # ∂pred_x / ∂xi
                    [ s, -c,  -c*dx - s*dy],   # ∂pred_y / ∂xi
                    [ 0,  0,  -1           ],   # ∂pred_t / ∂xi
                ], dtype=np.float64)
 
                # ∂(relative_pose)/∂xj
                Jj = np.array([
                    [c,  s,  0],
                    [-s, c,  0],
                    [0,  0,  1],
                ], dtype=np.float64)
 
                # ∂error/∂xi = -Ji ,  ∂error/∂xj = -Jj  (error = meas - pred)
                Ai = -Ji
                Aj = -Jj
                w  = edge.omega
 
                si, sj = slice(3*i, 3*i+3), slice(3*j, 3*j+3)
 
                H[si, si] += w * Ai.T @ Ai
                H[si, sj] += w * Ai.T @ Aj
                H[sj, si] += w * Aj.T @ Ai
                H[sj, sj] += w * Aj.T @ Aj
 
                b[si] += w * Ai.T @ e
                b[sj] += w * Aj.T @ e
 
            # Fix node 0 (add large diagonal to pin it)
            H[0:3, 0:3] += np.eye(3) * 1e6
 
            # Solve H Δx = –b
            try:
                delta = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                break
 
            poses += delta.reshape(n, 3)
            poses[:, 2] = np.arctan2(np.sin(poses[:, 2]),
                                     np.cos(poses[:, 2]))  # wrap angles
 
            if np.max(np.abs(delta)) < 1e-5:
                break   # converged
 
        # Write back
        for k, nd in enumerate(nodes):
            nd.x     = float(poses[k, 0])
            nd.y     = float(poses[k, 1])
            nd.theta = float(poses[k, 2])
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Main class  –  same public interface as the original OccupancyGrid
# ──────────────────────────────────────────────────────────────────────────────
 
class OccupancyGrid:
    """
    Graph-SLAM occupancy map.
 
    Internally keeps a pose graph of keyframes and uses Gauss-Newton
    optimisation with loop-closure constraints.  The rendered map is
    rebuilt from the optimised poses after each optimisation pass.
 
    Public API is identical to the original OccupancyGrid so main.py
    does not need any modification.
    """
 
    def __init__(
        self,
        world_min: Tuple[float, float] = (-2.0, -2.0),
        world_max: Tuple[float, float] = (2.0,  2.0),
        resolution: float = 0.01,
    ) -> None:
        self.world_min  = world_min
        self.world_max  = world_max
        self.resolution = resolution
 
        self.width  = int(math.ceil((world_max[0] - world_min[0]) / resolution))
        self.height = int(math.ceil((world_max[1] - world_min[1]) / resolution))
 
        # The rasterised occupancy grid (rebuilt after optimisation)
        self._grid: np.ndarray = np.zeros((self.height, self.width),
                                          dtype=np.float64)
        # Dirty flag: True means we need to re-rasterise
        self._map_dirty: bool = False
 
        # ── Pose graph ──────────────────────────────────────────────
        self._nodes:  List[Node] = []
        self._edges:  List[Edge] = []
        self._optimiser = _GraphOptimiser()
 
        # Last pose at which a keyframe was created (used for threshold check)
        self._last_kf_pose: Optional[Tuple[float,float,float]] = None
 
        # Current raw odometry pose (updated every timestep, not just keyframes)
        self._current_pose: Tuple[float,float,float] = (0.0, 0.0, 0.0)
 
        # How many loop-closure edges have been added so far (for stats)
        self._lc_count: int = 0
 
    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────
 
    def update(
        self,
        pose:      Tuple[float, float, float],
        ranges:    List[float],
        fov:       float,
        max_range: float,
    ) -> None:
        """
        Process one lidar scan at the given odometry pose.
 
        Steps
        -----
        1. Record current odometry pose.
        2. Decide whether to create a new keyframe.
        3. If yes → add odometry edge to previous keyframe.
        4. Search for loop-closure candidates.
        5. If a loop closure is found → add LC edge → run graph optimisation
           → rebuild the full occupancy grid from corrected poses.
        6. If no loop closure → fast incremental rasterisation of this scan only.
        """
        self._current_pose = pose
 
        if not self._should_create_keyframe(pose):
            # Between keyframes: do a quick incremental raster update so the
            # display stays responsive even before the next keyframe.
            self._rasterise_scan(pose, ranges, fov, max_range, incremental=True)
            return
 
        # ── Create new keyframe node ────────────────────────────────
        new_node = Node(
            idx       = len(self._nodes),
            x         = pose[0],
            y         = pose[1],
            theta     = pose[2],
            ranges    = list(ranges),
            fov       = fov,
            max_range = max_range,
        )
        self._nodes.append(new_node)
        self._last_kf_pose = pose
 
        # ── Add odometry edge from previous keyframe ────────────────
        if len(self._nodes) >= 2:
            prev = self._nodes[-2]
            curr = self._nodes[-1]
            rel  = _relative_pose((prev.x, prev.y, prev.theta),
                                  (curr.x,  curr.y,  curr.theta))
            self._edges.append(Edge(
                i=prev.idx, j=curr.idx,
                dx=rel[0], dy=rel[1], dtheta=rel[2],
                omega=5.0,   # odometry edges have moderate trust
            ))
 
        # ── Loop-closure search ─────────────────────────────────────
        lc_found = self._search_loop_closure(new_node)
 
        if lc_found:
            # Full graph optimisation + complete map rebuild
            self._optimise_and_rebuild()
        else:
            # Just rasterise this one scan incrementally (fast path)
            self._rasterise_scan(pose, ranges, fov, max_range, incremental=True)
 
    def render(self) -> np.ndarray:
        """Return a uint8 image: ~255 = occupied, ~128 = unknown, ~0 = free."""
        prob = 1.0 / (1.0 + np.exp(-self._grid))
        return (prob * 255).astype(np.uint8)
 
    # ──────────────────────────────────────────────────────────────────────────
    # Keyframe decision
    # ──────────────────────────────────────────────────────────────────────────
 
    def _should_create_keyframe(self, pose: Tuple[float,float,float]) -> bool:
        if self._last_kf_pose is None:
            return True                        # always create the very first node
        dx    = pose[0] - self._last_kf_pose[0]
        dy    = pose[1] - self._last_kf_pose[1]
        dtheta = abs(_wrap_angle(pose[2] - self._last_kf_pose[2]))
        dist  = math.hypot(dx, dy)
        return dist > KF_DIST_M or dtheta > KF_ANGLE_RAD
 
    # ──────────────────────────────────────────────────────────────────────────
    # Loop-closure detection
    # ──────────────────────────────────────────────────────────────────────────
 
    def _search_loop_closure(self, new_node: Node) -> bool:
        """
        Compare the new keyframe's scan with spatially nearby but temporally
        distant keyframes using scan correlation.
 
        Returns True if at least one loop closure edge was added.
        """
        n = len(self._nodes)
        if n < LC_MIN_HOPS + 2:
            return False         # not enough history yet
 
        # Candidate pool: all nodes except recent ones
        candidates = self._nodes[:n - LC_MIN_HOPS]
 
        # Sort by Euclidean distance (cheap pre-filter)
        def _dist(nd: Node) -> float:
            return math.hypot(nd.x - new_node.x, nd.y - new_node.y)
 
        candidates_sorted = sorted(candidates, key=_dist)
        candidates_near   = candidates_sorted[:LC_MAX_CANDIDATES]
 
        found = False
        for cand in candidates_near:
            score = _scan_correlation(new_node.ranges, cand.ranges,
                                      new_node.max_range)
            if score >= LC_SCORE_THRESH:
                # Compute relative pose between the two keyframes
                rel = _relative_pose(
                    (cand.x,     cand.y,     cand.theta),
                    (new_node.x, new_node.y, new_node.theta),
                )
                self._edges.append(Edge(
                    i=cand.idx, j=new_node.idx,
                    dx=rel[0], dy=rel[1], dtheta=rel[2],
                    omega=2.0 * score,  # higher score → more trust
                ))
                self._lc_count += 1
                found = True
                # One closure per keyframe is enough to trigger optimisation
                break
 
        return found
 
    # ──────────────────────────────────────────────────────────────────────────
    # Graph optimisation and map rebuild
    # ──────────────────────────────────────────────────────────────────────────
 
    def _optimise_and_rebuild(self) -> None:
        """Run Gauss-Newton on the full graph, then re-rasterise the map."""
        self._optimiser.optimise(self._nodes, self._edges)
        self._rebuild_map_from_graph()
 
    def _rebuild_map_from_graph(self) -> None:
        """
        Clear the occupancy grid and re-rasterise every stored keyframe scan
        using the (now optimised) node poses.
        """
        self._grid[:] = 0.0
        for nd in self._nodes:
            self._rasterise_scan(
                (nd.x, nd.y, nd.theta),
                nd.ranges, nd.fov, nd.max_range,
                incremental=False,   # write directly into self._grid
            )
 
    # ──────────────────────────────────────────────────────────────────────────
    # Scan rasterisation  (identical ray-casting logic to the original)
    # ──────────────────────────────────────────────────────────────────────────
 
    def _rasterise_scan(
        self,
        pose:       Tuple[float, float, float],
        ranges:     List[float],
        fov:        float,
        max_range:  float,
        incremental: bool = True,
    ) -> None:
        """
        Cast rays from `pose` using the given range scan and update self._grid.
 
        If incremental=False the grid is NOT cleared first (used when rebuilding
        the whole map from scratch — the caller clears before looping).
        """
        rx, ry, rtheta = pose
        num_points = len(ranges)
        if num_points == 0:
            return
 
        ox, oy = self._world_to_grid(rx, ry)
 
        for i, r in enumerate(ranges):
            if r <= 0 or math.isnan(r) or math.isinf(r):
                continue
 
            hit   = r < max_range
            angle = rtheta + fov / 2.0 - i * (fov / num_points)
 
            end_x = rx + r * math.cos(angle)
            end_y = ry + r * math.sin(angle)
            ex, ey = self._world_to_grid(end_x, end_y)
 
            for bx, by in _bresenham(ox, oy, ex, ey):
                if not self._in_bounds(bx, by):
                    continue
                if bx == ex and by == ey:
                    break
                self._grid[by, bx] += FREE_DELTA
                self._grid[by, bx]  = max(self._grid[by, bx], CLAMP_MIN)
 
            if hit and self._in_bounds(ex, ey):
                self._grid[ey, ex] += OCC_DELTA
                self._grid[ey, ex]  = min(self._grid[ey, ex], CLAMP_MAX)
 
    # ──────────────────────────────────────────────────────────────────────────
    # Grid helpers  (same as original)
    # ──────────────────────────────────────────────────────────────────────────
 
    def _world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = int(round((wx - self.world_min[0]) / self.resolution))
        gy = int(round((wy - self.world_min[1]) / self.resolution))
        gy = self.height - gy      # Webots Y-axis inversion
        return gx, gy
 
    def _in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height
 
    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics  (optional – call from main.py if you want to see stats)
    # ──────────────────────────────────────────────────────────────────────────
 
    @property
    def num_nodes(self) -> int:
        return len(self._nodes)
 
    @property
    def num_edges(self) -> int:
        return len(self._edges)
 
    @property
    def loop_closure_count(self) -> int:
        return self._lc_count