import math
import numpy as np
from typing import List, Tuple

from sim.person import Person  # keep parity with the original planner


class TSDWA:
    """Targeted‑Sampling Dynamic Window Approach (TS‑DWA) local planner.

    This class follows the formulation in Shen & Soh, *J. Mechanisms and Robotics* (2024)
    and is a drop‑in replacement for the legacy :class:`DWA` class located in `dwa.py`.

    Core ideas implemented here
    ---------------------------
    1. **Polar translational sampling** biased towards the global‑plan heading.
    2. **Path‑curvature‑coupled angular samples** (ω = v * κ  + α * θ_offset).
    3. A *minimal* set of lateral / reverse samples for escape behaviour.
    4. Hooks to plug into the existing scoring pipeline (goal, clearance, velocity).

    The public API intentionally mirrors the original DWA so existing simulation
    scripts (GUI, visualisers, episode runners) continue to work unchanged.
    """

    # ---------------------------------------------------------------------
    # ─── INITIALISATION ───────────────────────────────────────────────────
    # ---------------------------------------------------------------------

    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        max_speed: float,
        goal: Tuple[float, float],
        radius: float,
        corridor_bounds: dict,
        *,
        look_ahead_idx: int = 7,        # i_look in the paper
        n_heading: int = 7,             # n_asamp   (angular samples in polar space)
        n_speed: int = 9,               # n_vsamp   (speed magnitude samples)
        theta_range: float = math.pi/6,  # θ_range   (±60° cone)
        alpha_ph: float = 1.0,          # α_ph heading‑bias gain
    ) -> None:
        # Save parameters identical to the original DWA planner ------------------
        self.position = position
        self.velocity = velocity
        self.max_speed = max_speed
        self.goal = np.array(goal)
        self.radius = radius
        self.corridor_bounds = corridor_bounds

        # -------  TS‑specific parameters  --------------------------------------
        self.look_ahead_idx = look_ahead_idx
        self.n_heading = n_heading
        self.n_speed = n_speed
        self.theta_range = theta_range
        self.alpha_ph = alpha_ph

        # Dynamics / limits replicate original values so scoring remains valid ----
        self.max_rotation = math.pi
        self.max_accel = 1.0 * 4
        self.max_angular_accel = math.pi * 2
        self.dt = 0.1
        self.predict_time = 2.0

        # Stored for visualisation ----------------------------------------------
        self.trajectories: List[np.ndarray] = []
        self.best_trajectory: np.ndarray | None = None

        # Runtime state ----------------------------------------------------------
        self.v = 0.0
        self.w = 0.0

        # Re‑use original scoring weights for now; user may tune externally
        self.weights = {"goal": 0.2, "clearance": 0.7, "velocity": 0.1}

    # ---------------------------------------------------------------------
    # ─── PUBLIC API (matches original DWA) ────────────────────────────────
    # ---------------------------------------------------------------------

    def set_goal(self, goal: Tuple[float, float]):
        self.goal = np.array(goal)

    def update(self, dt: float, people: List[Person], global_path: List[np.ndarray]):
        """Main planner step.

        Parameters
        ----------
        dt : float
            Simulation time‐step.
        people : list[Person]
            Crowd for clearance scoring.
        global_path : list[np.ndarray]
            Sequence of way‑points produced by a global planner, **in map frame**.
            The first element is assumed to be the robot pose; the last is the goal.
        """
        if self.goal is None:
            return

        # ------------------------------------------------------------------
        # 1. Dynamic Window (same computation as legacy planner)
        dw = self._dynamic_window()

        # ------------------------------------------------------------------
        # 2. Extract path heading θ_ph and curvature κ                    
        theta_ph = self._extract_heading(global_path)
        kappa = self._extract_curvature(global_path)

        # ------------------------------------------------------------------
        # 3. Generate *targeted* samples biased along the global path
        samples = self._generate_ts_samples(dw, theta_ph, kappa)

        # ------------------------------------------------------------------
        # 4. Trajectory rollout + scoring (identical to baseline)
        best_score = -float("inf")
        best_v, best_w = 0.0, 0.0
        self.trajectories.clear()

        for v_sample, w_sample in samples:
            traj = self._predict_trajectory(v_sample, w_sample)
            self.trajectories.append(traj)

            g = self._goal_score(traj)
            c = self._clearance_score(traj, people)
            vel_score = v_sample / self.max_speed

            score = (
                self.weights["goal"] * g
                + self.weights["clearance"] * c
                + self.weights["velocity"] * vel_score
            )
            if score > best_score:
                best_score = score
                best_v, best_w = v_sample, w_sample
                self.best_trajectory = traj

        # ------------------------------------------------------------------
        # 5. Command update & kinematics integration
        self.v, self.w = best_v, best_w
        self.orientation = getattr(self, "orientation", 0.0) + self.w * dt
        self.orientation = self._normalize_angle(self.orientation)
        self.velocity = np.array([
            self.v * math.cos(self.orientation),
            self.v * math.sin(self.orientation),
        ])
        self.position += self.velocity * dt
        return self.velocity, self.position, self.goal

    # ------------------------------------------------------------------
    # ─── INTERNAL UTILITIES ─────────────────────────────────────────────
    # ------------------------------------------------------------------

    def _extract_heading(self, global_path: List[np.ndarray]) -> float:
        """Extract path heading by finding closest point and looking ahead."""
        if len(global_path) < 2:
            return 0.0
        
        # 1. Find the closest point on the path to robot position
        min_dist = float('inf')
        closest_idx = 0
        
        for i, path_point in enumerate(global_path):
            dist = np.linalg.norm(path_point - self.position)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 2. Look ahead by look_ahead_idx steps from the closest point
        look_ahead_idx = min(closest_idx + self.look_ahead_idx, len(global_path) - 1)
        look_pt = global_path[look_ahead_idx]
        
        # 3. Calculate heading from robot to look-ahead point
        rel = look_pt - self.position
        heading = math.atan2(rel[1], rel[0]) - getattr(self, "orientation", 0.0)
        
        return self._normalize_angle(heading)

    def _extract_curvature(self, global_path: List[np.ndarray]) -> float:
        # Algorithm 2 (Menger curvature) — simplified for 2D points
        i, j, k = 0, min(1 + 2, len(global_path) - 2), min(2 + 4, len(global_path) - 1)
        p0, p1, p2 = global_path[i], global_path[j], global_path[k]
        # shift p0 to origin
        p1_shifted = p1.copy() - p0
        p2_shifted = p2.copy() - p0
        area2 = abs(p1_shifted[0]*p2_shifted[1] - p1_shifted[1]*p2_shifted[0])
        denom = np.linalg.norm(p1_shifted) * np.linalg.norm(p2_shifted) * np.linalg.norm(p1_shifted - p2_shifted) + 1e-6
        return 2 * area2 / denom

    def _generate_ts_samples(
        self, dw: Tuple[float, float, float, float], theta_ph: float, kappa: float
    ) -> List[Tuple[float, float]]:
        """Polar velocity generator with path‑aware bias."""
        v_min, v_max, w_min, w_max = dw

        # Translational sampling in polar space ---------------------------
        headings = np.linspace(
            theta_ph - self.theta_range,
            theta_ph + self.theta_range,
            self.n_heading,
        )
        speeds = np.linspace(self.max_speed * 0.1, self.max_speed, self.n_speed)

        samples: list[tuple[float, float]] = []
        for h in headings:
            for s in speeds:
                vx = s * math.cos(h)
                vy = s * math.sin(h)
                v_trans = math.hypot(vx, vy)
                # clamp to dw linear limits
                v_trans = np.clip(v_trans, v_min, v_max)
                # Angular velocity coupling (Eq. 7 in paper)
                omega = v_trans * (kappa + self.alpha_ph * h)
                omega = np.clip(omega, w_min, w_max)
                samples.append((v_trans, omega))

                # Add zero‑omega version (straight motion) for exit corridors
                samples.append((v_trans, 0.0))

        # Escape manoeuvres (left/right/back) -----------------------------
        for omega_bias in (-self.max_rotation * 0.5, self.max_rotation * 0.5):
            samples.append((self.max_speed * 0.3, omega_bias))
        samples.append((self.max_speed * 0.2, 0.0))  # reverse‑like slow move
        return samples

    # -------  Legacy helpers copied verbatim from original planner  -----

    def _dynamic_window(self):
        vs = [0, self.max_speed, -self.max_rotation, self.max_rotation]
        vd = [
            self.v - self.max_accel * self.dt,
            self.v + self.max_accel * self.dt,
            self.w - self.max_angular_accel * self.dt,
            self.w + self.max_angular_accel * self.dt,
        ]
        return [
            max(vs[0], vd[0]),
            min(vs[1], vd[1]),
            max(vs[2], vd[2]),
            min(vs[3], vd[3]),
        ]

    def _predict_trajectory(self, v: float, w: float):
        time = 0.0
        traj = [self.position.copy()]
        pos = self.position.copy()
        theta = getattr(self, "orientation", 0.0)
        while time < self.predict_time:
            theta += w * self.dt
            pos[0] += v * math.cos(theta) * self.dt
            pos[1] += v * math.sin(theta) * self.dt
            traj.append(pos.copy())
            time += self.dt
        return np.array(traj)

    def _goal_score(self, traj):
        final = traj[-1]
        dist_goal = np.linalg.norm(self.goal - final)
        dist_max = np.linalg.norm(self.goal - self.position) + 1e-6
        direction_goal = (self.goal - self.position) / max(dist_max, 1e-6)
        direction_traj = (final - self.position) / max(np.linalg.norm(final - self.position), 1e-6)
        alignment = np.dot(direction_goal, direction_traj)
        return 0.7 * (1 - dist_goal / dist_max) + 0.3 * alignment

    def _clearance_score(self, traj, people):
        min_dist = float("inf")
        for person in people:
            if not person.active:
                continue
            for p in traj:
                d = np.linalg.norm(p - person.position) - self.radius - person.radius
                min_dist = min(min_dist, d)
                if min_dist <= 0:
                    return -float("inf")
        # corridor collisions
        bounds = self.corridor_bounds
        for p in traj[:5]:  # TODO: update to the term for wall check points (Orientation brunch)
            dists = [
                p[0] - bounds["x_min"] - self.radius,
                bounds["x_max"] - p[0] - self.radius,
                p[1] - bounds["y_min"] - self.radius,
                bounds["y_max"] - p[1] - self.radius,
            ]
            min_dist = min(min_dist, *dists)
            if min_dist <= 0:
                return -float("inf")
        return min(min_dist / 1.0, 1.0)

    @staticmethod
    def _normalize_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
