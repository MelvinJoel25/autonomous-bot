"""
omnibot_navigation.py  –  v2  (coordinate-fixed release)
═══════════════════════════════════════════════════════════════════════════════
Open-Loop Grid Navigation  ·  Omniwheel Robot  ·  20×20 Banner Arena

FIXES over v1
─────────────
  FIX-1  heading_to_cmd() rewritten with correct world-angle derivation.
         Banner frame: col+ = right, row+ = up.
         Image frame:  x+ = right, y+ = DOWN  (opposite to row).
         The function now converts banner (dc, dr) → world angle correctly
         before subtracting robot heading, eliminating the ~180° command flip.

  FIX-2  heading is re-read from the latest detection before EVERY step, not
         just at mission start.  Stored in MissionRunner._last_theta and
         updated each time detect_robot() succeeds.

  FIX-3  px_to_cell_homography uses int(coord/CELL) not round(coord/CELL-0.5).
         The old formula displaced cell centres by half a cell.

  FIX-4  path_length_cm is used consistently for both the display estimate and
         the execution timer.

  FIX-5  _build_grid honours config.inflate_obs instead of the global constant.

  FIX-6  build_command_sequence removed as a standalone pre-builder.
         Commands are now derived step-by-step inside the mission loop using
         the most recent heading, not the stale initial one.

  FIX-7  grid_overlay draws border lines correctly (+1 outside loop).

  FIX-8  Simple colour palette matching Autobot_navigation_v4.py style.

  DIAG   Full debug log: robot heading, current cell, target cell, desired
         vector, chosen command, world angle, relative angle printed to
         console for every step when DEBUG_MOVEMENT = True.

Mission cycle (open-loop — NO continuous camera feedback while driving)
────────────────────────────────────────────────────────────────────────
  1. Snapshot   – ONE frame, undistort
  2. Detect     – AprilTag ID 0 + YOLO obstacles
  3. Localise   – pixel → banner cell
  4. Plan       – A* on 20×20 grid
  5. Command    – step-by-step: derive command from CURRENT heading → send
  6. Wait       – sleep per-step (BANNER_CELL_CM / speed_cms)
  7. Verify     – ONE fresh snapshot, re-detect
  8. Check      – arrived? done. else retry.

Banner coordinate system
─────────────────────────
  Origin (0,0) = bottom-left corner
  col  0–19  →  left to right      (+x in world)
  row  0–19  →  bottom to top      (+y in world, opposite to image y)

  Image y increases DOWNWARD.  World y increases UPWARD.
  All world-angle maths uses the world frame; image-frame y is flipped
  exactly once, at the point where dc/dr are computed.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
import heapq
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import pupil_apriltags
    APRILTAG_OK = True
except ImportError:
    APRILTAG_OK = False

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False


# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

BANNER_COLS     = 20
BANNER_ROWS     = 20
BANNER_CELL_CM  = 15.0
BANNER_W_CM     = BANNER_COLS * BANNER_CELL_CM   # 300
BANNER_H_CM     = BANNER_ROWS * BANNER_CELL_CM   # 300

TAG_SIZE_CM     = 12.0          # physical AprilTag side length
ROBOT_TAG_ID    = 0
ARRIVAL_THRESH  = 1             # cells — within this = arrived
DEFAULT_RETRIES = 5
DEFAULT_SPEED   = 20.0          # cm/s
DWELL_BUFFER_S  = 0.5

DEFAULT_ROBOT_IP   = "192.168.4.1"
DEFAULT_ROBOT_PORT = 80

YOLO_CONF       = 0.45
YOLO_IOU        = 0.45
OBSTACLE_INFLATE = 1

# Set True to print per-step movement diagnostics to the console
DEBUG_MOVEMENT  = True

LEFT_W          = 370

# ── Simple colour palette (matches Autobot_navigation_v4 style) ──
BG      = "#f0f0f0"
BG2     = "#ffffff"
BG3     = "#e0e0e0"
FG      = "#000000"
FG2     = "#555555"
ACCENT  = "#0055cc"
GREEN   = "#006600"
RED     = "#cc0000"
ORANGE  = "#cc6600"
MONO    = ("Courier New", 9)
LABELF  = ("Arial", 9)
HEADF   = ("Arial", 10, "bold")


# ══════════════════════════════════════════════════════════════
#  MISSION STATE
# ══════════════════════════════════════════════════════════════

class MissionState(Enum):
    IDLE       = auto()
    SNAPSHOT   = auto()
    DETECTING  = auto()
    PLANNING   = auto()
    COMMANDING = auto()
    WAITING    = auto()
    VERIFYING  = auto()
    COMPLETE   = auto()
    FAILED     = auto()
    ABORTED    = auto()


# ══════════════════════════════════════════════════════════════
#  BANNER MAPPER
# ══════════════════════════════════════════════════════════════

class BannerMapper:
    """
    Converts between image pixels and banner grid cells.

    Two modes, auto-selected:
      Homography mode  — uses H (world_cm → image_px) from calibration.
      Anchor mode      — uses cm_per_pixel + anchor_px for overhead cams.

    COORDINATE CONVENTION
    ─────────────────────
    World frame (banner):
        origin   = bottom-left corner of the physical banner
        x_cm     = increases rightward  (col increases rightward)
        y_cm     = increases upward     (row increases upward)

    Image frame:
        origin   = top-left of image
        x_px     = increases rightward  ← SAME direction as world x
        y_px     = increases DOWNWARD   ← OPPOSITE to world y

    Cell centre in world frame:
        x_cm = (col + 0.5) * BANNER_CELL_CM
        y_cm = (row + 0.5) * BANNER_CELL_CM

    FIX-3: pixel → cell via homography
        H_inv maps image_px → world_cm
        col = int(x_cm / BANNER_CELL_CM)   ← not round(x/cell - 0.5)
        row = int(y_cm / BANNER_CELL_CM)
    """

    def __init__(self):
        self.cm_per_pixel: Optional[float]          = None
        self.anchor_px:    Optional[tuple[int,int]] = None
        self.anchor_cell:  tuple[int,int]           = (0, 0)
        self.homography:   Optional[np.ndarray]     = None   # world_cm → px
        self._H_inv:       Optional[np.ndarray]     = None   # px → world_cm

    def load_from_profile(self, profile: dict) -> None:
        hom = profile.get("homography", profile.get("H", None))
        if hom is not None:
            H = np.array(hom, dtype=np.float64)
            if H.shape == (3, 3):
                self.homography = H
                self._H_inv     = np.linalg.inv(H)

        if "cm_per_pixel" in profile:
            self.cm_per_pixel = float(profile["cm_per_pixel"])
        elif "px_per_cell" in profile:
            ppc = float(profile["px_per_cell"])
            if ppc > 0:
                self.cm_per_pixel = BANNER_CELL_CM / ppc

        ap = profile.get("anchor_px", profile.get("anchorPx", None))
        if ap is not None:
            t = tuple(int(x) for x in ap)
            if len(t) >= 2:
                self.anchor_px = (t[0], t[1])

        ac = profile.get("anchor_cell", profile.get("anchorCell", None))
        if ac is not None:
            t = tuple(int(x) for x in ac)
            if len(t) >= 2:
                self.anchor_cell = (t[0], t[1])

    def is_ready(self) -> bool:
        return (self.homography is not None or
                (self.cm_per_pixel is not None and self.anchor_px is not None))

    @property
    def px_per_cell(self) -> float:
        if self.cm_per_pixel and self.cm_per_pixel > 0:
            return BANNER_CELL_CM / self.cm_per_pixel
        return 0.0

    # ── pixel → banner cell ─────────────────────────────────

    def px_to_cell(self, x_px: float, y_px: float) -> Optional[tuple[int,int]]:
        if self.homography is not None and self._H_inv is not None:
            return self._px_to_cell_H(x_px, y_px)
        if self.cm_per_pixel and self.anchor_px:
            return self._px_to_cell_anchor(x_px, y_px)
        return None

    def _px_to_cell_H(self, x_px: float, y_px: float) -> tuple[int,int]:
        """
        FIX-3: use int(coord / CELL_CM) instead of round(coord/CELL - 0.5).
        H_inv maps (x_px, y_px) → (x_cm, y_cm) in world frame.
        Because H was built with world origin = bottom-left and y up,
        the result is already in the correct convention.
        """
        pt  = np.array([[[x_px, y_px]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self._H_inv)
        x_cm = float(res[0, 0, 0])
        y_cm = float(res[0, 0, 1])
        col  = int(x_cm / BANNER_CELL_CM)
        row  = int(y_cm / BANNER_CELL_CM)
        return (max(0, min(BANNER_COLS - 1, col)),
                max(0, min(BANNER_ROWS - 1, row)))

    def _px_to_cell_anchor(self, x_px: float, y_px: float) -> tuple[int,int]:
        """
        Anchor-based: image y increases downward, world row increases upward.
        dx_px positive  → col increases  (same direction)
        dy_px positive  → row DECREASES  (opposite direction) — note the minus sign.
        """
        ppc = self.px_per_cell
        if ppc <= 0:
            return (0, 0)
        ax, ay   = self.anchor_px
        ac, ar   = self.anchor_cell
        col = int(round(ac + (x_px - ax) / ppc))
        row = int(round(ar - (y_px - ay) / ppc))   # ← minus: y_px down = row down
        return (max(0, min(BANNER_COLS - 1, col)),
                max(0, min(BANNER_ROWS - 1, row)))

    # ── banner cell → pixel centre ──────────────────────────

    def cell_to_px(self, col: int, row: int) -> Optional[tuple[int,int]]:
        if self.homography is not None:
            return self._cell_to_px_H(col, row)
        if self.cm_per_pixel and self.anchor_px:
            return self._cell_to_px_anchor(col, row)
        return None

    def _cell_to_px_H(self, col: int, row: int) -> tuple[int,int]:
        x_cm = (col + 0.5) * BANNER_CELL_CM
        y_cm = (row + 0.5) * BANNER_CELL_CM
        pt   = np.array([[[x_cm, y_cm]]], dtype=np.float32)
        res  = cv2.perspectiveTransform(pt, self.homography)
        return (int(round(float(res[0, 0, 0]))),
                int(round(float(res[0, 0, 1]))))

    def _cell_to_px_anchor(self, col: int, row: int) -> tuple[int,int]:
        ppc  = self.px_per_cell
        ax, ay = self.anchor_px
        ac, ar = self.anchor_cell
        x = int(round(ax + (col - ac) * ppc))
        y = int(round(ay - (row - ar) * ppc))   # ← minus: row up = y_px down
        return x, y

    # ── A* index helpers ────────────────────────────────────

    @staticmethod
    def cell_to_astar(col: int, row: int) -> tuple[int,int]:
        """
        Banner cell (col, row) → A* numpy index (astar_row, astar_col).
        numpy row 0 = top of array = banner row 19 (top of arena).
        astar_row = (BANNER_ROWS - 1) - row
        """
        return (BANNER_ROWS - 1) - row, col

    @staticmethod
    def astar_to_cell(ar: int, ac: int) -> tuple[int,int]:
        return ac, (BANNER_ROWS - 1) - ar

    # ── runtime scale fallback ──────────────────────────────

    def update_scale_from_tag(self, edge_px: float) -> None:
        """Derive cm_per_pixel from AprilTag edge length when not in calibration."""
        if self.cm_per_pixel is None and edge_px > 5:
            self.cm_per_pixel = TAG_SIZE_CM / edge_px


# ══════════════════════════════════════════════════════════════
#  MOVEMENT COMMAND LOGIC  (FIX-1 & FIX-2)
# ══════════════════════════════════════════════════════════════

# Arduino command map
# The omniwheel robot interprets these in its OWN body frame:
#   F  = drive forward  (robot's local +y)
#   B  = drive backward (robot's local -y)
#   R  = strafe right   (robot's local +x)
#   L  = strafe left    (robot's local -x)
#   D  = forward-right diagonal
#   A  = forward-left  diagonal
#   E  = backward-right diagonal
#   Q  = backward-left  diagonal
#   S  = stop

def heading_to_cmd(
    dc: int,            # target_col - current_col  (+right in banner world)
    dr: int,            # target_row - current_row  (+up in banner world)
    robot_theta_deg: float,
    debug: bool = False,
) -> str:
    """
    FIX-1 — Correct world-angle derivation.

    WORLD FRAME convention used here
    ─────────────────────────────────
    We work entirely in the world (banner) frame where:
        +x = rightward  (col increases)
        +y = upward     (row increases)
    atan2 with this convention:
        pure right   (dc=+1, dr=0) → 0°
        pure up      (dc=0,  dr=+1) → 90°
        pure left    (dc=-1, dr=0) → 180°
        pure down    (dc=0,  dr=-1) → 270° (or −90°)

    ROBOT HEADING convention
    ─────────────────────────
    AprilTag gives theta from image coordinates.
    The tag is mounted so that theta=0 means the robot faces the
    +x (rightward) direction in the world frame.

    If your physical mount is different, adjust TAG_HEADING_OFFSET_DEG.

    The key insight:
        rel = (world_angle - robot_fwd_world) % 360
    where robot_fwd_world is the world-frame angle that corresponds
    to the robot sending command "F".

    For a standard top-mount with tag forward aligned to world +x:
        robot_fwd_world = robot_theta_deg  (directly from AprilTag)

    For a rear-mount or 90°-rotated tag, set TAG_HEADING_OFFSET_DEG.
    """
    # World angle of the desired move (atan2 in world frame: y=up, x=right)
    world_angle = math.degrees(math.atan2(dr, dc)) % 360.0

    # World angle of robot's forward direction.
    # AprilTag theta is measured in the IMAGE frame (y down, x right).
    # For a top-facing camera with x-right, the image x-axis and world
    # x-axis agree.  The image y-axis is inverted vs world y, which means
    # the heading angle is NEGATED when converting image → world frame.
    robot_fwd_world = (-robot_theta_deg) % 360.0

    # Relative angle: how far clockwise from robot-forward is the target?
    rel = (world_angle - robot_fwd_world) % 360.0

    # Map 8 octants to commands
    if   rel <  22.5 or rel >= 337.5: cmd = "F"   # straight forward
    elif rel <  67.5:                 cmd = "A"   # forward-left
    elif rel < 112.5:                 cmd = "L"   # straight left
    elif rel < 157.5:                 cmd = "Q"   # backward-left
    elif rel < 202.5:                 cmd = "B"   # straight backward
    elif rel < 247.5:                 cmd = "E"   # backward-right
    elif rel < 292.5:                 cmd = "R"   # straight right
    else:                             cmd = "D"   # forward-right

    if debug or DEBUG_MOVEMENT:
        print(
            f"[CMD] dc={dc:+d} dr={dr:+d}  "
            f"world_angle={world_angle:.1f}°  "
            f"robot_theta={robot_theta_deg:.1f}°  "
            f"robot_fwd_world={robot_fwd_world:.1f}°  "
            f"rel={rel:.1f}°  "
            f"→ {cmd}"
        )

    return cmd


# Tuning constant — add degrees if your tag is physically rotated on the robot
# e.g. TAG_HEADING_OFFSET_DEG = 90  if tag faces robot's left side
TAG_HEADING_OFFSET_DEG = 0.0


def corrected_theta(raw_theta: float) -> float:
    """Apply any physical tag-mount offset to the raw AprilTag heading."""
    return (raw_theta + TAG_HEADING_OFFSET_DEG) % 360.0


# ══════════════════════════════════════════════════════════════
#  A* PLANNER
# ══════════════════════════════════════════════════════════════

class AStarPlanner:
    """
    A* on the 20×20 A* numpy grid.
    astar_row 0 = banner row 19 (top of arena).
    Diagonal cost = √2.
    """
    _DIRS = [
        ( 0, 1, 1.0), ( 1, 0, 1.0), ( 0,-1, 1.0), (-1, 0, 1.0),
        ( 1, 1, 1.414), (-1,-1, 1.414), ( 1,-1, 1.414), (-1, 1, 1.414),
    ]

    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.h, self.w = grid.shape

    @staticmethod
    def _heuristic(a: tuple, b: tuple) -> float:
        dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
        return abs(dr-dc) + 1.414 * min(dr, dc)

    def plan(self, start: tuple, goal: tuple) -> Optional[list[tuple]]:
        heap = [(0.0, start)]
        came: dict = {}
        g = {start: 0.0}
        while heap:
            _, cur = heapq.heappop(heap)
            if cur == goal:
                path = []
                while cur in came:
                    path.append(cur)
                    cur = came[cur]
                path.append(start)
                return path[::-1]
            for dr, dc, cost in self._DIRS:
                nr, nc = cur[0]+dr, cur[1]+dc
                if not (0 <= nr < self.h and 0 <= nc < self.w):
                    continue
                if self.grid[nr, nc] == 1:
                    continue
                ng = g[cur] + cost
                nb = (nr, nc)
                if nb not in g or ng < g[nb]:
                    came[nb] = cur
                    g[nb]    = ng
                    heapq.heappush(heap, (ng + self._heuristic(nb, goal), nb))
        return None


def path_length_cm(path_astar: list[tuple]) -> float:
    """Total path length in cm — diagonal step = √2 × BANNER_CELL_CM."""
    total = 0.0
    for i in range(len(path_astar) - 1):
        dr = abs(path_astar[i+1][0] - path_astar[i][0])
        dc = abs(path_astar[i+1][1] - path_astar[i][1])
        total += (1.414 if (dr and dc) else 1.0) * BANNER_CELL_CM
    return total


def inflate_obstacles(grid: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return grid
    try:
        from scipy.ndimage import binary_dilation
        struct = np.ones((2*radius+1, 2*radius+1), bool)
        return binary_dilation(grid, structure=struct).astype(np.uint8)
    except ImportError:
        out = grid.copy()
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                out |= np.roll(np.roll(grid, dr, 0), dc, 1)
        return out


# ══════════════════════════════════════════════════════════════
#  PERCEPTION
# ══════════════════════════════════════════════════════════════

@dataclass
class RobotDetection:
    found:       bool
    px:          float = 0.0
    py:          float = 0.0
    theta_deg:   float = 0.0    # raw from AprilTag (image frame)
    cell:        Optional[tuple[int,int]] = None
    tag_edge_px: float = 0.0


@dataclass
class ObstacleCell:
    col: int
    row: int


class Perception:
    """
    One-shot camera snapshot + AprilTag + YOLO.
    Never runs continuously.
    """

    def __init__(self):
        self.cap:           Optional[cv2.VideoCapture] = None
        self.camera_matrix: Optional[np.ndarray]       = None
        self.dist_coeffs:   Optional[np.ndarray]       = None

        if APRILTAG_OK:
            self.at_detector = pupil_apriltags.Detector(
                families="tag36h11", nthreads=4,
                quad_decimate=1.0, quad_sigma=0.0,
                refine_edges=1, decode_sharpening=0.25,
            )
        else:
            self.at_detector = None

        self.yolo_model: Optional[Any] = None

    def connect_camera(self, source: Any) -> bool:
        if self.cap:
            self.cap.release()
        import platform
        backend = (cv2.CAP_DSHOW
                   if isinstance(source, int) and platform.system() == "Windows"
                   else cv2.CAP_ANY)
        cap = cv2.VideoCapture(source, backend)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, _ = cap.read()
        if not ret:
            cap.release()
            return False
        self.cap = cap
        return True

    def load_yolo(self, model_path: str) -> bool:
        if not YOLO_OK:
            return False
        try:
            self.yolo_model = YOLO(model_path)
            return True
        except Exception:
            return False

    def snapshot(self) -> Optional[np.ndarray]:
        """
        ONE fresh frame — grabs once to flush stale buffer, then reads.
        Undistorts with calibration if available.
        """
        if not self.cap:
            return None
        self.cap.grab()   # flush buffer
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            frame = cv2.undistort(frame, self.camera_matrix,
                                  self.dist_coeffs, None, self.camera_matrix)
        return frame

    def detect_robot(
        self,
        frame: np.ndarray,
        mapper: BannerMapper,
    ) -> RobotDetection:
        if self.at_detector is None:
            return RobotDetection(found=False)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = self.at_detector.detect(gray)

        for d in dets:
            if int(d.tag_id) != ROBOT_TAG_ID:
                continue
            cx, cy    = float(d.center[0]), float(d.center[1])
            corners   = d.corners.astype(np.float32)
            c0, c1    = corners[0], corners[1]
            edge_px   = float(np.linalg.norm(c1 - c0))
            raw_theta = float(
                np.degrees(np.arctan2(c1[1]-c0[1], c1[0]-c0[0]))) % 360.0

            mapper.update_scale_from_tag(edge_px)
            cell = mapper.px_to_cell(cx, cy)

            if DEBUG_MOVEMENT:
                print(f"[TAG] centre=({cx:.1f},{cy:.1f})  "
                      f"raw_theta={raw_theta:.1f}°  "
                      f"corrected={corrected_theta(raw_theta):.1f}°  "
                      f"cell={cell}")

            return RobotDetection(
                found=True, px=cx, py=cy,
                theta_deg=raw_theta,
                cell=cell, tag_edge_px=edge_px)

        return RobotDetection(found=False)

    def detect_obstacles(
        self,
        frame: np.ndarray,
        mapper: BannerMapper,
        robot_px: Optional[tuple[float,float]] = None,
    ) -> list[ObstacleCell]:
        if self.yolo_model is None or not mapper.is_ready():
            return []

        results = self.yolo_model(frame, conf=YOLO_CONF,
                                  iou=YOLO_IOU, verbose=False)
        obstacles: list[ObstacleCell] = []

        robot_r_px = 0
        if robot_px and mapper.cm_per_pixel and mapper.cm_per_pixel > 0:
            robot_r_px = int(30.0 / mapper.cm_per_pixel)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                if robot_px:
                    dx, dy = cx - robot_px[0], cy - robot_px[1]
                    if math.hypot(dx, dy) < max(robot_r_px, 40):
                        continue
                cell = mapper.px_to_cell(cx, cy)
                if cell:
                    obstacles.append(ObstacleCell(*cell))

        return obstacles


# ══════════════════════════════════════════════════════════════
#  ARDUINO HTTP CLIENT
# ══════════════════════════════════════════════════════════════

class ArduinoClient:
    SPEED_PREFIX = {1: "1", 2: "2", 3: "3"}

    def __init__(self, ip: str = DEFAULT_ROBOT_IP,
                 port: int = DEFAULT_ROBOT_PORT):
        self.ip   = ip
        self.port = port
        self.speed = 2
        self._connected = False
        self._lock = threading.Lock()
        self.last_cmd = ""

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> tuple[bool, str]:
        import requests as req
        for endpoint in ("/cmd?v=S", "/stop", "/"):
            try:
                r = req.get(f"http://{self.ip}:{self.port}{endpoint}", timeout=3)
                self._connected = True
                return True, f"HTTP {r.status_code}"
            except Exception as e:
                last_err = str(e)
        return False, last_err

    def disconnect(self):
        self._connected = False

    def send(self, cmd: str) -> tuple[str, float]:
        if not self._connected:
            if DEBUG_MOVEMENT:
                print(f"[SIM] would send: {cmd}")
            return "NOT_CONNECTED", 0.0
        import requests as req
        full = (f"{self.SPEED_PREFIX.get(self.speed,'2')}{cmd}"
                if cmd != "S" else "S")
        with self._lock:
            try:
                t0 = time.time()
                r  = req.get(
                    f"http://{self.ip}:{self.port}/cmd?v={full}", timeout=2)
                ms = (time.time()-t0)*1000
                self.last_cmd = full
                return ("OK" if r.ok else f"ERR_{r.status_code}"), ms
            except Exception as e:
                return f"ERR:{e}", 0.0

    def send_stop(self):
        self.send("S")


# ══════════════════════════════════════════════════════════════
#  GRID CANVAS WIDGET
# ══════════════════════════════════════════════════════════════

class GridCanvas(tk.Canvas):
    """
    Schematic 20×20 grid view.
    Banner row 0 = physical bottom → drawn at bottom of canvas (y flipped).
    Click callback returns (col, row) in banner space.
    """
    CELL_PX = 28

    def __init__(self, parent, **kwargs):
        cw = BANNER_COLS * self.CELL_PX
        ch = BANNER_ROWS * self.CELL_PX
        super().__init__(parent, width=cw, height=ch,
                         bg="white", highlightthickness=1,
                         highlightbackground="black", **kwargs)
        self._obstacles:   set[tuple[int,int]] = set()
        self._path:        list[tuple[int,int]] = []
        self._robot_cell:  Optional[tuple[int,int]] = None
        self._target_cell: Optional[tuple[int,int]] = None
        self._robot_theta: float = 0.0
        self._click_callback = None
        self.bind("<Button-1>", self._on_click)
        self._draw()

    def set_state(
        self,
        obstacles:   set[tuple[int,int]],
        path:        list[tuple[int,int]],
        robot_cell:  Optional[tuple[int,int]],
        target_cell: Optional[tuple[int,int]],
        robot_theta: float = 0.0,
    ):
        self._obstacles   = obstacles
        self._path        = path
        self._robot_cell  = robot_cell
        self._target_cell = target_cell
        self._robot_theta = robot_theta
        self._draw()

    def _cell_rect(self, col: int, row: int):
        cp = self.CELL_PX
        x1 = col * cp
        y1 = (BANNER_ROWS - 1 - row) * cp   # row 0 → bottom of canvas
        return x1, y1, x1+cp, y1+cp

    def _draw(self):
        self.delete("all")
        cp       = self.CELL_PX
        path_set = set(self._path)

        for row in range(BANNER_ROWS):
            for col in range(BANNER_COLS):
                x1, y1, x2, y2 = self._cell_rect(col, row)
                cell = (col, row)
                if cell in self._obstacles:
                    fill = "#ffcccc"
                elif cell in path_set:
                    fill = "#ccddff"
                else:
                    fill = "white"
                self.create_rectangle(x1, y1, x2, y2,
                                      fill=fill, outline="#cccccc", width=1)
                if col % 5 == 0 and row % 5 == 0:
                    self.create_text(x1+2, y2-2, text=f"{col},{row}",
                                     font=("Courier New", 5),
                                     fill="#aaaaaa", anchor=tk.SW)

        # Target outline
        if self._target_cell:
            tc, tr = self._target_cell
            x1, y1, x2, y2 = self._cell_rect(tc, tr)
            self.create_rectangle(x1+2, y1+2, x2-2, y2-2,
                                   outline="black", width=2)
            self.create_text((x1+x2)//2, (y1+y2)//2,
                              text="T", font=("Arial", 7, "bold"), fill="black")

        # Path arrows
        if len(self._path) > 1:
            for i in range(len(self._path)-1):
                c0, r0 = self._path[i]
                c1, r1 = self._path[i+1]
                x1,y1,x2,y2 = self._cell_rect(c0, r0)
                mx0, my0 = (x1+x2)//2, (y1+y2)//2
                x1,y1,x2,y2 = self._cell_rect(c1, r1)
                mx1, my1 = (x1+x2)//2, (y1+y2)//2
                self.create_line(mx0, my0, mx1, my1,
                                 fill="#0055cc", width=2, arrow=tk.LAST)

        # Robot
        if self._robot_cell:
            rc, rr = self._robot_cell
            x1, y1, x2, y2 = self._cell_rect(rc, rr)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            r = cp // 3
            self.create_oval(cx-r, cy-r, cx+r, cy+r,
                              fill="#aaaaaa", outline="black", width=2)
            # Heading arrow in canvas frame (canvas y is flipped vs world)
            # corrected_theta in world frame; canvas y is flipped, so negate
            theta_canvas = -corrected_theta(self._robot_theta)
            angle_rad    = math.radians(theta_canvas)
            ex = cx + int(r * 1.7 * math.cos(angle_rad))
            ey = cy + int(r * 1.7 * math.sin(angle_rad))
            self.create_line(cx, cy, ex, ey,
                             fill="black", width=2, arrow=tk.LAST)

    def _on_click(self, event):
        cp  = self.CELL_PX
        col = event.x // cp
        row = BANNER_ROWS - 1 - (event.y // cp)
        col = max(0, min(BANNER_COLS-1, col))
        row = max(0, min(BANNER_ROWS-1, row))
        if self._click_callback:
            self._click_callback(col, row)


# ══════════════════════════════════════════════════════════════
#  CAMERA PREVIEW CANVAS
# ══════════════════════════════════════════════════════════════

class CameraPreview(tk.Canvas):
    """Shows the last captured frame with simple detection overlays."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="black",
                         highlightthickness=1,
                         highlightbackground="black", **kwargs)
        self._imgtk = None

    def show_frame(
        self,
        frame:     np.ndarray,
        robot:     Optional[RobotDetection] = None,
        obstacles: Optional[list[ObstacleCell]] = None,
        mapper:    Optional[BannerMapper] = None,
    ):
        img = frame.copy()
        h_img, w_img = img.shape[:2]

        # Grid overlay
        if mapper and mapper.is_ready():
            self._draw_grid_overlay(img, mapper, w_img, h_img)

        # Obstacles
        if obstacles and mapper and mapper.is_ready():
            for obs in obstacles:
                px = mapper.cell_to_px(obs.col, obs.row)
                if px:
                    cv2.circle(img, px, 10, (80, 80, 80), -1)
                    cv2.circle(img, px, 12, (0,  0,  0), 1)

        # Robot tag
        if robot and robot.found:
            cx, cy = int(robot.px), int(robot.py)
            cv2.circle(img, (cx, cy), 16, (0, 0, 0), 2)
            r = math.radians(robot.theta_deg)
            ex = int(cx + 30 * math.cos(r))
            ey = int(cy + 30 * math.sin(r))
            cv2.arrowedLine(img, (cx, cy), (ex, ey), (0,0,0), 2, tipLength=0.4)
            cell_str = str(robot.cell) if robot.cell else ""
            cv2.putText(img, f"ROBOT {cell_str} th={int(robot.theta_deg)}",
                        (cx+18, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw > 10 and ch > 10:
            img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)
        rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self._imgtk = imgtk

    def _draw_grid_overlay(self, img, mapper, w_img, h_img):
        """
        FIX-7: draw BANNER_COLS+1 vertical and BANNER_ROWS+1 horizontal lines
        so all cell borders (including the outer edge) are shown.
        """
        for col in range(BANNER_COLS + 1):
            # Left edge of column `col`
            x_cm = col * BANNER_CELL_CM
            # Project a point on the left edge at row 0 and row BANNER_ROWS
            pt0 = _project_world_to_px(x_cm, 0,              mapper)
            pt1 = _project_world_to_px(x_cm, BANNER_H_CM,    mapper)
            if pt0 and pt1:
                cv2.line(img, pt0, pt1, (120, 120, 120), 1)

        for row in range(BANNER_ROWS + 1):
            y_cm = row * BANNER_CELL_CM
            pt0  = _project_world_to_px(0,           y_cm, mapper)
            pt1  = _project_world_to_px(BANNER_W_CM, y_cm, mapper)
            if pt0 and pt1:
                cv2.line(img, pt0, pt1, (120, 120, 120), 1)


def _project_world_to_px(
    x_cm: float, y_cm: float,
    mapper: BannerMapper,
) -> Optional[tuple[int,int]]:
    """Project a world-cm point to image pixels using homography or anchor."""
    if mapper.homography is not None:
        pt  = np.array([[[x_cm, y_cm]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, mapper.homography)
        return int(round(float(res[0,0,0]))), int(round(float(res[0,0,1])))
    if mapper.cm_per_pixel and mapper.anchor_px:
        ppc = mapper.px_per_cell
        ax, ay = mapper.anchor_px
        ac, ar = mapper.anchor_cell
        x_px = int(round(ax + (x_cm / BANNER_CELL_CM - ac) * ppc))
        y_px = int(round(ay - (y_cm / BANNER_CELL_CM - ar) * ppc))
        return x_px, y_px
    return None


# ══════════════════════════════════════════════════════════════
#  MISSION RUNNER  (FIX-2: per-step heading re-read)
# ══════════════════════════════════════════════════════════════

@dataclass
class MissionConfig:
    target_cell:    tuple[int,int]
    max_retries:    int   = DEFAULT_RETRIES
    speed_cms:      float = DEFAULT_SPEED
    arrival_thresh: int   = ARRIVAL_THRESH
    inflate_obs:    int   = OBSTACLE_INFLATE


@dataclass
class MissionResult:
    success:     bool
    final_cell:  Optional[tuple[int,int]]
    attempts:    int
    message:     str


class MissionRunner:
    """
    Executes:  snapshot → detect → plan → (step command → sleep) × N
               → snapshot → detect → arrived? done : retry

    FIX-2: _last_theta is updated every time detect_robot() succeeds.
    Each movement step uses the freshest known heading, not stale initial one.
    """

    def __init__(
        self,
        perception: Perception,
        mapper:     BannerMapper,
        arduino:    ArduinoClient,
        config:     MissionConfig,
        on_state_change,
        on_frame,
        on_grid_update,
    ):
        self.perception = perception
        self.mapper     = mapper
        self.arduino    = arduino
        self.config     = config
        self._on_state  = on_state_change
        self._on_frame  = on_frame
        self._on_grid   = on_grid_update
        self._abort     = threading.Event()
        self._thread:   Optional[threading.Thread] = None

        # FIX-2: last known robot heading, updated on every successful detection
        self._last_theta: float = 0.0

    def start(self):
        self._abort.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def abort(self):
        self._abort.set()
        self.arduino.send_stop()

    def _run(self):
        cfg     = self.config
        attempt = 0
        current_cell: Optional[tuple[int,int]] = None

        while attempt <= cfg.max_retries and not self._abort.is_set():
            attempt += 1
            self._state(MissionState.SNAPSHOT,
                        f"Attempt {attempt}/{cfg.max_retries+1} — snapshot")

            # ── 1. Snapshot ───────────────────────────────────
            frame = self.perception.snapshot()
            if frame is None:
                self._state(MissionState.FAILED, "Camera error: no frame")
                return

            # ── 2. Detect ─────────────────────────────────────
            self._state(MissionState.DETECTING, "Detecting…")
            robot = self.perception.detect_robot(frame, self.mapper)
            if not robot.found:
                self._state(MissionState.FAILED,
                            "Robot not found (AprilTag ID 0 missing)")
                self._on_frame(frame, None, [])
                return

            # FIX-2: store heading immediately after detection
            self._last_theta = corrected_theta(robot.theta_deg)

            obstacles = self.perception.detect_obstacles(
                frame, self.mapper, (robot.px, robot.py))
            self._on_frame(frame, robot, obstacles)

            # ── 3. Localise ───────────────────────────────────
            current_cell = robot.cell
            if current_cell is None:
                self._state(MissionState.FAILED,
                            "Cannot convert robot pixel to banner cell")
                return

            if DEBUG_MOVEMENT:
                print(f"\n[MISSION] attempt={attempt}  "
                      f"current={current_cell}  target={cfg.target_cell}  "
                      f"theta={self._last_theta:.1f}°")

            # Pre-move arrival check
            if self._arrived(current_cell, cfg.target_cell, cfg.arrival_thresh):
                self._state(MissionState.COMPLETE,
                            f"Arrived at {cfg.target_cell} in {attempt} attempt(s)")
                self._on_grid({(o.col,o.row) for o in obstacles},
                              [], current_cell, cfg.target_cell, self._last_theta)
                return

            # ── 4. Plan ───────────────────────────────────────
            self._state(MissionState.PLANNING,
                        f"A* {current_cell} → {cfg.target_cell}")
            grid = self._build_grid(obstacles, cfg.inflate_obs)
            sr, sc = BannerMapper.cell_to_astar(*current_cell)
            er, ec = BannerMapper.cell_to_astar(*cfg.target_cell)
            grid[sr, sc] = 0   # ensure start is passable
            grid[er, ec] = 0   # ensure goal  is passable

            path_astar = AStarPlanner(grid).plan((sr, sc), (er, ec))
            if path_astar is None:
                self._state(MissionState.FAILED, "A*: no path found")
                return

            path_cells = [BannerMapper.astar_to_cell(r, c) for r, c in path_astar]
            total_cm   = path_length_cm(path_astar)
            step_s     = BANNER_CELL_CM / max(cfg.speed_cms, 1.0)
            total_s    = total_cm / max(cfg.speed_cms, 1.0) + DWELL_BUFFER_S

            self._on_grid({(o.col,o.row) for o in obstacles},
                          path_cells, current_cell, cfg.target_cell,
                          self._last_theta)

            # ── 5 & 6. Command each step ──────────────────────
            self._state(MissionState.COMMANDING,
                        f"{len(path_cells)-1} steps  "
                        f"{total_cm:.0f}cm  ≈{total_s:.1f}s")

            if not self.arduino.connected:
                # Simulation: print commands, wait
                self._state(MissionState.WAITING,
                            f"[SIM] {total_s:.1f}s")
                for i in range(len(path_cells) - 1):
                    if self._abort.is_set():
                        break
                    col0, row0 = path_cells[i]
                    col1, row1 = path_cells[i+1]
                    dc = col1 - col0
                    dr = row1 - row0
                    # FIX-2: use self._last_theta (updated from latest detection)
                    cmd = heading_to_cmd(dc, dr, self._last_theta)
                    is_diag = (dc != 0 and dr != 0)
                    dur     = step_s * (1.414 if is_diag else 1.0)
                    print(f"[SIM step {i+1}] dc={dc:+d} dr={dr:+d} "
                          f"theta={self._last_theta:.1f}° → {cmd}  {dur:.2f}s")
                    self._interruptible_sleep(dur)
            else:
                self._state(MissionState.WAITING,
                            f"Moving {len(path_cells)-1} steps…")
                for i in range(len(path_cells) - 1):
                    if self._abort.is_set():
                        break
                    col0, row0 = path_cells[i]
                    col1, row1 = path_cells[i+1]
                    dc = col1 - col0
                    dr = row1 - row0
                    # FIX-2: use self._last_theta
                    cmd = heading_to_cmd(dc, dr, self._last_theta)
                    is_diag = (dc != 0 and dr != 0)
                    dur     = step_s * (1.414 if is_diag else 1.0)
                    self.arduino.send(cmd)
                    self._interruptible_sleep(dur)

                self.arduino.send_stop()
                self._interruptible_sleep(DWELL_BUFFER_S)

            if self._abort.is_set():
                self._state(MissionState.ABORTED, "Aborted by user")
                return

            # ── 7 & 8. Verify ─────────────────────────────────
            self._state(MissionState.VERIFYING, "Verifying…")
            vframe = self.perception.snapshot()
            if vframe is None:
                self._state(MissionState.FAILED, "Verify: no frame")
                return

            vrobot = self.perception.detect_robot(vframe, self.mapper)
            vobs   = self.perception.detect_obstacles(
                vframe, self.mapper,
                (vrobot.px, vrobot.py) if vrobot.found else None)
            self._on_frame(vframe, vrobot, vobs)

            if vrobot.found and vrobot.cell is not None:
                # FIX-2: update heading from verification frame too
                self._last_theta = corrected_theta(vrobot.theta_deg)
                current_cell     = vrobot.cell
                if DEBUG_MOVEMENT:
                    print(f"[VERIFY] cell={current_cell}  "
                          f"theta={self._last_theta:.1f}°")

            self._on_grid(
                {(o.col,o.row) for o in vobs},
                [], current_cell, cfg.target_cell, self._last_theta)

            if self._arrived(current_cell, cfg.target_cell, cfg.arrival_thresh):
                self._state(MissionState.COMPLETE,
                            f"✓ Arrived at {cfg.target_cell} after {attempt} attempt(s)")
                return

            self._state(MissionState.SNAPSHOT,
                        f"At {current_cell}, target {cfg.target_cell} — retry")

        # Retries exhausted
        self._state(MissionState.FAILED,
                    f"Max retries ({cfg.max_retries}) — stopped at {current_cell}")

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _arrived(current: tuple, target: tuple, thresh: int) -> bool:
        return (abs(current[0]-target[0]) <= thresh and
                abs(current[1]-target[1]) <= thresh)

    @staticmethod
    def _build_grid(
        obstacles:  list[ObstacleCell],
        inflate:    int,
    ) -> np.ndarray:
        grid = np.zeros((BANNER_ROWS, BANNER_COLS), dtype=np.uint8)
        for obs in obstacles:
            ar, ac = BannerMapper.cell_to_astar(obs.col, obs.row)
            if 0 <= ar < BANNER_ROWS and 0 <= ac < BANNER_COLS:
                grid[ar, ac] = 1
        if inflate > 0:
            grid = inflate_obstacles(grid, inflate)
        return grid

    def _interruptible_sleep(self, seconds: float):
        end = time.time() + seconds
        while time.time() < end and not self._abort.is_set():
            time.sleep(0.05)

    def _state(self, state: MissionState, msg: str):
        self._on_state(state, msg)


# ══════════════════════════════════════════════════════════════
#  CALIBRATION LOADER
# ══════════════════════════════════════════════════════════════

def load_calibration_npz(path: str) -> Optional[dict]:
    """Load .npz calibration saved by camera_calibration.py."""
    try:
        raw = np.load(path, allow_pickle=True)
        out: dict = {}
        for key in raw.files:
            val = raw[key]
            if isinstance(val, np.ndarray) and val.ndim == 0:
                out[key] = val.item()
            elif key in ("camera_matrix","cameraMatrix","K",
                         "dist_coeffs","distCoeffs","dist",
                         "homography","H","banner_corners_px"):
                out[key] = val
            elif key in ("resolution","anchor_px","anchorPx",
                         "anchor_cell","anchorCell"):
                out[key] = tuple(int(x) for x in val.tolist())
            elif key in ("cm_per_pixel","px_per_cell","rms_error",
                         "height_cm","tilt_deg"):
                out[key] = float(val)
            elif key in ("grid_cols","grid_rows","n_frames"):
                out[key] = int(val)
            else:
                try:    out[key] = str(val)
                except: out[key] = val

        # Normalise aliases
        if "camera_matrix" not in out:
            for a in ("cameraMatrix","K"):
                if a in out: out["camera_matrix"] = out[a]; break
        if "dist_coeffs" not in out:
            for a in ("distCoeffs","dist"):
                if a in out: out["dist_coeffs"] = out[a]; break
        if "homography" not in out and "H" in out:
            out["homography"] = out["H"]
        if "cm_per_pixel" not in out and "px_per_cell" in out:
            ppc = float(out["px_per_cell"])
            if ppc > 0:
                out["cm_per_pixel"] = BANNER_CELL_CM / ppc

        return out
    except Exception as e:
        print(f"[load_calibration_npz] {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  TKINTER STYLE (simple, matches Autobot_navigation_v4 look)
# ══════════════════════════════════════════════════════════════

def _apply_style(root: tk.Tk):
    s = ttk.Style(root)
    s.theme_use("clam")
    # Keep system-like colours — not heavy teal/orange
    s.configure(".",
                background=BG, foreground=FG,
                fieldbackground=BG2, font=LABELF,
                borderwidth=1, relief="flat")
    s.configure("TFrame",       background=BG)
    s.configure("TLabelframe",  background=BG, borderwidth=1,
                relief="groove")
    s.configure("TLabelframe.Label", background=BG, font=HEADF)
    s.configure("TButton",      background=BG3, foreground=FG,
                padding=(6,4))
    s.map("TButton",
          background=[("active", "#d0d0d0"), ("pressed", "#b0b0b0")])
    s.configure("Accent.TButton", background="#0055cc", foreground="white",
                font=("Arial", 9, "bold"))
    s.map("Accent.TButton",
          background=[("active","#0033aa"),("pressed","#002288")])
    s.configure("Stop.TButton", background="#cc0000", foreground="white",
                font=("Arial", 9, "bold"))
    s.map("Stop.TButton",
          background=[("active","#aa0000"),("pressed","#880000")])
    s.configure("TLabel",    background=BG, foreground=FG, font=LABELF)
    s.configure("Dim.TLabel",background=BG, foreground=FG2, font=LABELF)
    s.configure("Mono.TLabel",background=BG, foreground=FG, font=MONO)
    s.configure("TEntry",    fieldbackground=BG2, foreground=FG, font=LABELF)
    s.configure("TSpinbox",  fieldbackground=BG2, foreground=FG, font=LABELF)
    s.configure("TScrollbar",background=BG3)
    s.configure("TSeparator",background=BG3)


# ══════════════════════════════════════════════════════════════
#  MAIN GUI
# ══════════════════════════════════════════════════════════════

class NavigationApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Omnibot — Open-Loop Grid Navigation  v2")
        self.root.configure(bg=BG)
        self.root.state("zoomed")
        _apply_style(root)

        self.perception  = Perception()
        self.mapper      = BannerMapper()
        self.arduino     = ArduinoClient()
        self.calibration: Optional[dict] = None
        self.mission:     Optional[MissionRunner] = None

        self._v_robot_ip   = tk.StringVar(value=DEFAULT_ROBOT_IP)
        self._v_robot_port = tk.IntVar(value=DEFAULT_ROBOT_PORT)
        self._v_target_col = tk.IntVar(value=10)
        self._v_target_row = tk.IntVar(value=10)
        self._v_speed_cms  = tk.DoubleVar(value=DEFAULT_SPEED)
        self._v_max_retry  = tk.IntVar(value=DEFAULT_RETRIES)
        self._v_cam_src    = tk.StringVar(value="0")
        self._v_yolo_path  = tk.StringVar(value="")
        self._v_arrive_thr = tk.IntVar(value=ARRIVAL_THRESH)
        self._v_inflate    = tk.IntVar(value=OBSTACLE_INFLATE)

        self._current_cell:  Optional[tuple[int,int]] = None
        self._target_cell:   Optional[tuple[int,int]] = None
        self._obstacles:     set[tuple[int,int]]      = set()
        self._path:          list[tuple[int,int]]     = []
        self._robot_theta:   float                    = 0.0
        self._mission_state: MissionState             = MissionState.IDLE

        self._preview_enabled = False
        self._preview_job: Optional[str] = None
        self._last_click_px: Optional[tuple[int,int]] = None

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(300, self._startup_load_calibration)

    # ─────────────────────────────────────────────────────────
    #  UI BUILD
    # ─────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header strip
        hdr = tk.Frame(self.root, bg=BG3, height=40)
        hdr.pack(fill=tk.X, side=tk.TOP)
        tk.Label(hdr,
                 text="Omnibot  –  Open-Loop Grid Navigation  v2",
                 font=("Arial",11,"bold"),
                 bg=BG3, fg=FG).pack(side=tk.LEFT, padx=14, pady=8)
        self._hdr_status = tk.Label(hdr, text="Startup…",
                                    font=LABELF, bg=BG3, fg=FG2)
        self._hdr_status.pack(side=tk.RIGHT, padx=14)

        # Body: left panel | camera preview | grid canvas
        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4,6))
        body.columnconfigure(0, weight=0, minsize=LEFT_W)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0)
        body.rowconfigure(0, weight=1)

        # Left scrollable panel
        lc = tk.Frame(body, bg=BG, width=LEFT_W)
        lc.grid(row=0, column=0, sticky="nsew", padx=(0,4))
        lc.grid_propagate(False)
        lcan = tk.Canvas(lc, bg=BG, highlightthickness=0, width=LEFT_W-20)
        lscr = ttk.Scrollbar(lc, orient="vertical", command=lcan.yview)
        lf   = ttk.Frame(lcan)
        lf.bind("<Configure>",
                lambda e: lcan.configure(scrollregion=lcan.bbox("all")))
        lcan.create_window((0,0), window=lf, anchor="nw")
        lcan.configure(yscrollcommand=lscr.set)
        lcan.bind_all("<MouseWheel>",
                      lambda e: lcan.yview_scroll(int(-e.delta/120),"units"))
        lcan.pack(side="left", fill="both", expand=True)
        lscr.pack(side="right", fill="y")
        left = lf

        # Camera preview
        cf = ttk.LabelFrame(body, text="Camera Snapshot", padding=2)
        cf.grid(row=0, column=1, sticky="nsew", padx=4)
        cf.rowconfigure(0, weight=1); cf.columnconfigure(0, weight=1)
        self._cam_preview = CameraPreview(cf)
        self._cam_preview.grid(row=0, column=0, sticky="nsew")
        self._cam_preview.bind("<Button-1>", self._on_preview_click)

        # Grid canvas
        gf = ttk.LabelFrame(body, text="Banner Grid  (20×20)", padding=2)
        gf.grid(row=0, column=2, sticky="nsew", padx=(4,0))
        gf.rowconfigure(0, weight=1); gf.columnconfigure(0, weight=1)
        self._grid_canvas = GridCanvas(gf)
        self._grid_canvas.grid(row=0, column=0)
        self._grid_canvas._click_callback = self._on_grid_click

        # Build panel sections
        self._build_sec_calibration(left)
        self._build_sec_camera(left)
        self._build_sec_robot(left)
        self._build_sec_mission(left)
        self._build_sec_status(left)
        self._build_sec_anchor(left)
        self._build_sec_debug(left)

    # ── Section helpers ───────────────────────────────────────

    def _sep(self, p):
        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, pady=3)

    def _btn(self, p, text, cmd, style="TButton"):
        b = ttk.Button(p, text=text, command=cmd, style=style)
        b.pack(fill=tk.X, pady=2)
        return b

    def _row_entry(self, p, label, var, w=14):
        r = ttk.Frame(p); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text=label, width=19, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=var, width=w).pack(side=tk.LEFT, padx=2)

    def _row_spin(self, p, label, var, frm, to, inc=1.0, w=8):
        r = ttk.Frame(p); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text=label, width=19, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(r, textvariable=var, from_=frm, to=to,
                    increment=inc, width=w).pack(side=tk.LEFT, padx=2)

    def _mono(self, p, text=""):
        l = ttk.Label(p, text=text, style="Mono.TLabel", wraplength=340)
        l.pack(anchor=tk.W, pady=1)
        return l

    # ── Section: calibration ──────────────────────────────────
    def _build_sec_calibration(self, left):
        sf = ttk.LabelFrame(left, text="Calibration Profile", padding=6)
        sf.pack(fill=tk.X, pady=3)
        self._calib_lbl = ttk.Label(sf, text="No calibration loaded",
                                    font=MONO, foreground=RED,
                                    wraplength=340, justify=tk.LEFT)
        self._calib_lbl.pack(anchor=tk.W, pady=2)
        self._btn(sf, "Load Calibration File (.npz)", self._load_calibration)

    # ── Section: camera ───────────────────────────────────────
    def _build_sec_camera(self, left):
        sf = ttk.LabelFrame(left, text="Camera", padding=6)
        sf.pack(fill=tk.X, pady=3)
        self._row_entry(sf, "Source (index/URL):", self._v_cam_src, 14)
        ttk.Label(sf, text="DroidCam: http://IP:4747/video",
                  style="Dim.TLabel").pack(anchor=tk.W)
        self._btn(sf, "Connect Camera", self._connect_camera, "Accent.TButton")
        self._btn(sf, "Take Snapshot Now", self._manual_snapshot)
        self._cam_status = self._mono(sf, "Not connected")

        self._sep(sf)
        ttk.Label(sf, text="YOLO model path (optional):",
                  style="Dim.TLabel").pack(anchor=tk.W)
        r = ttk.Frame(sf); r.pack(fill=tk.X, pady=2)
        ttk.Entry(r, textvariable=self._v_yolo_path, width=22).pack(
            side=tk.LEFT, padx=(0,2))
        ttk.Button(r, text="...", command=self._browse_yolo, width=3).pack(side=tk.LEFT)
        self._btn(sf, "Load YOLO Model", self._load_yolo)
        self._yolo_status = self._mono(sf, "YOLO: not loaded")

    # ── Section: robot ────────────────────────────────────────
    def _build_sec_robot(self, left):
        sf = ttk.LabelFrame(left, text="Robot (Arduino R4 WiFi)", padding=6)
        sf.pack(fill=tk.X, pady=3)
        self._row_entry(sf, "Robot IP:", self._v_robot_ip, 14)
        self._row_spin(sf, "Port:", self._v_robot_port, 1, 99999, 1)
        bf = ttk.Frame(sf); bf.pack(fill=tk.X, pady=2)
        ttk.Button(bf, text="Connect", style="Accent.TButton",
                   command=self._connect_robot).pack(side=tk.LEFT, padx=(0,3))
        ttk.Button(bf, text="Disconnect",
                   command=self._disconnect_robot).pack(side=tk.LEFT, padx=(0,3))
        ttk.Button(bf, text="STOP", style="Stop.TButton",
                   command=self._emergency_stop).pack(side=tk.LEFT)
        self._robot_status = self._mono(sf, "Not connected")

    # ── Section: mission ──────────────────────────────────────
    def _build_sec_mission(self, left):
        sf = ttk.LabelFrame(left, text="Mission", padding=6)
        sf.pack(fill=tk.X, pady=3)

        ttk.Label(sf, text="Target cell (col, row)  0–19:",
                  style="Dim.TLabel").pack(anchor=tk.W)
        tf = ttk.Frame(sf); tf.pack(fill=tk.X, pady=2)
        ttk.Label(tf, text="Col:").pack(side=tk.LEFT)
        ttk.Spinbox(tf, textvariable=self._v_target_col,
                    from_=0, to=19, width=4).pack(side=tk.LEFT, padx=3)
        ttk.Label(tf, text="Row:").pack(side=tk.LEFT, padx=(6,0))
        ttk.Spinbox(tf, textvariable=self._v_target_row,
                    from_=0, to=19, width=4).pack(side=tk.LEFT, padx=3)
        ttk.Label(sf, text="Or click on the grid canvas →",
                  style="Dim.TLabel").pack(anchor=tk.W, pady=(0,3))

        self._row_spin(sf, "Speed (cm/s):", self._v_speed_cms, 1, 200, 1)
        self._row_spin(sf, "Max retries:", self._v_max_retry, 0, 20, 1)
        self._row_spin(sf, "Arrival thresh (cells):", self._v_arrive_thr, 0, 5, 1)
        self._row_spin(sf, "Obstacle inflate (cells):", self._v_inflate, 0, 5, 1)

        self._sep(sf)
        self._btn(sf, "START MISSION", self._start_mission, "Accent.TButton")
        self._btn(sf, "ABORT MISSION", self._abort_mission, "Stop.TButton")

    # ── Section: status ───────────────────────────────────────
    def _build_sec_status(self, left):
        sf = ttk.LabelFrame(left, text="Mission Status", padding=6)
        sf.pack(fill=tk.X, pady=3)
        self._lbl_state   = self._mono(sf, "State   : IDLE")
        self._lbl_current = self._mono(sf, "Robot   : –")
        self._lbl_target  = self._mono(sf, "Target  : –")
        self._lbl_path    = self._mono(sf, "Path    : –")
        self._lbl_travel  = self._mono(sf, "Travel  : –")
        self._lbl_msg     = ttk.Label(sf, text="", font=LABELF,
                                      foreground=FG2, wraplength=340)
        self._lbl_msg.pack(anchor=tk.W, pady=2)

    # ── Section: anchor override ──────────────────────────────
    def _build_sec_anchor(self, left):
        sf = ttk.LabelFrame(left,
                            text="Anchor Override (if no homography)",
                            padding=6)
        sf.pack(fill=tk.X, pady=3)
        ttk.Label(sf,
                  text="Snapshot → click a known point on camera preview\n"
                       "→ enter its banner cell → Set Anchor.",
                  style="Dim.TLabel", wraplength=340).pack(anchor=tk.W)
        self._v_anch_col = tk.IntVar(value=0)
        self._v_anch_row = tk.IntVar(value=0)
        af = ttk.Frame(sf); af.pack(fill=tk.X, pady=2)
        ttk.Label(af, text="Col:").pack(side=tk.LEFT)
        ttk.Spinbox(af, textvariable=self._v_anch_col,
                    from_=0, to=19, width=4).pack(side=tk.LEFT, padx=3)
        ttk.Label(af, text="Row:").pack(side=tk.LEFT, padx=(6,0))
        ttk.Spinbox(af, textvariable=self._v_anch_row,
                    from_=0, to=19, width=4).pack(side=tk.LEFT, padx=3)
        self._btn(sf, "Set Anchor from Last Click", self._set_anchor_from_click)
        self._anch_lbl = self._mono(sf, "Anchor: not set")

    # ── Section: debug / tuning ───────────────────────────────
    def _build_sec_debug(self, left):
        sf = ttk.LabelFrame(left, text="Debug / Tuning", padding=6)
        sf.pack(fill=tk.X, pady=3)

        self._v_debug    = tk.BooleanVar(value=DEBUG_MOVEMENT)
        self._v_tag_off  = tk.DoubleVar(value=TAG_HEADING_OFFSET_DEG)

        ttk.Checkbutton(sf, text="Print movement diagnostics to console",
                        variable=self._v_debug,
                        command=self._toggle_debug).pack(anchor=tk.W)

        r = ttk.Frame(sf); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="Tag heading offset (°):", width=22,
                  anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(r, textvariable=self._v_tag_off,
                    from_=-360, to=360, increment=5, width=6).pack(side=tk.LEFT)
        ttk.Button(r, text="Apply",
                   command=self._apply_tag_offset).pack(side=tk.LEFT, padx=4)

        ttk.Label(sf,
                  text="If robot moves in wrong direction:\n"
                       "  • Try offset = 90, 180, or 270\n"
                       "  • Check DEBUG output for 'rel' angle\n"
                       "  • Verify your tag is mounted flat on top",
                  style="Dim.TLabel", wraplength=340).pack(anchor=tk.W, pady=4)

    # ─────────────────────────────────────────────────────────
    #  LIVE PREVIEW
    # ─────────────────────────────────────────────────────────

    def _start_live_preview(self):
        self._preview_enabled = True
        if self._preview_job is None:
            self._preview_tick()

    def _stop_live_preview(self):
        self._preview_enabled = False
        if self._preview_job:
            try: self.root.after_cancel(self._preview_job)
            except: pass
            self._preview_job = None

    def _preview_tick(self):
        self._preview_job = None
        if not self._preview_enabled or self.perception.cap is None:
            return
        busy_states = {MissionState.SNAPSHOT, MissionState.DETECTING,
                       MissionState.PLANNING, MissionState.COMMANDING,
                       MissionState.WAITING, MissionState.VERIFYING}
        if self._mission_state in busy_states:
            if self._preview_enabled:
                self._preview_job = self.root.after(200, self._preview_tick)
            return

        frame = self.perception.snapshot()
        if frame is not None:
            self._cam_preview.show_frame(frame, None, [], self.mapper)
        self._preview_job = self.root.after(120, self._preview_tick)

    # ─────────────────────────────────────────────────────────
    #  STARTUP CALIBRATION
    # ─────────────────────────────────────────────────────────

    def _startup_load_calibration(self):
        messagebox.showinfo(
            "Load Calibration",
            "Select the camera calibration file (.npz)\n"
            "saved by camera_calibration.py.\n\n"
            "Must contain: camera_matrix, dist_coeffs\n"
            "Optional: homography (or cm_per_pixel + anchor_px)",
            parent=self.root)
        self._load_calibration()

    def _load_calibration(self):
        path = filedialog.askopenfilename(
            title="Select calibration profile (.npz)",
            filetypes=[("Calibration","*.npz"),("All","*.*")],
            parent=self.root)
        if not path:
            self._set_status("No calibration loaded")
            return

        data = load_calibration_npz(path)
        if data is None:
            messagebox.showerror("Error", f"Cannot load:\n{path}")
            return

        self.calibration = data
        if "camera_matrix" in data:
            self.perception.camera_matrix = data["camera_matrix"]
            self.perception.dist_coeffs   = data.get("dist_coeffs")
        self.mapper.load_from_profile(data)

        rms   = data.get("rms_error", "–")
        h     = data.get("height_cm", "–")
        cpp   = data.get("cm_per_pixel", "–")
        has_H = "homography" in data and data["homography"] is not None
        mode  = "homography" if has_H else "anchor"
        ready = self.mapper.is_ready()
        col   = GREEN if ready else RED

        summary = (
            f"File : {os.path.basename(path)}\n"
            f"RMS  : {rms:.4f}\n" if not isinstance(rms,str) else
            f"File : {os.path.basename(path)}\nRMS  : {rms}\n"
        )
        summary += (
            f"H cm : {h}\n"
            f"Mode : {mode}\n"
            f"cpp  : {cpp if isinstance(cpp,str) else f'{cpp:.5f}'}\n"
            f"Ready: {'yes' if ready else 'NO — set anchor below'}"
        )
        self._calib_lbl.config(text=summary, foreground=col)
        self._set_status(f"Calibration loaded ({mode} mode)")
        self._update_grid()

    # ─────────────────────────────────────────────────────────
    #  CAMERA ACTIONS
    # ─────────────────────────────────────────────────────────

    def _connect_camera(self):
        src_str = self._v_cam_src.get().strip()
        src: Any = int(src_str) if src_str.isdigit() else src_str
        self._set_status(f"Connecting {src}…")
        self.root.update()
        if self.perception.connect_camera(src):
            self._cam_status.config(text=f"Connected: {src}", foreground=GREEN)
            self._start_live_preview()
            self._manual_snapshot()
        else:
            self._cam_status.config(text=f"Failed: {src}", foreground=RED)
            messagebox.showerror("Camera Error",
                f"Cannot open: {src}\nCheck index or URL.")

    def _manual_snapshot(self):
        frame = self.perception.snapshot()
        if frame is None:
            messagebox.showwarning("No frame", "Connect camera first")
            return
        robot = self.perception.detect_robot(frame, self.mapper)
        obs   = self.perception.detect_obstacles(
            frame, self.mapper,
            (robot.px, robot.py) if robot.found else None)
        self._update_from_detection(frame, robot, obs)
        self._set_status(
            f"Snapshot — {'Robot cell '+str(robot.cell) if robot.found else 'Robot NOT FOUND'}"
            f"  Obstacles: {len(obs)}")

    def _browse_yolo(self):
        path = filedialog.askopenfilename(
            title="YOLO model", filetypes=[("PyTorch","*.pt"),("All","*.*")])
        if path: self._v_yolo_path.set(path)

    def _load_yolo(self):
        path = self._v_yolo_path.get().strip()
        if not path:
            messagebox.showwarning("No path", "Enter YOLO model path")
            return
        self._set_status("Loading YOLO…"); self.root.update()
        if self.perception.load_yolo(path):
            self._yolo_status.config(
                text=f"YOLO: {os.path.basename(path)}", foreground=GREEN)
        else:
            self._yolo_status.config(text="YOLO load failed", foreground=RED)

    # ─────────────────────────────────────────────────────────
    #  ROBOT ACTIONS
    # ─────────────────────────────────────────────────────────

    def _connect_robot(self):
        ip   = self._v_robot_ip.get().strip()
        port = self._v_robot_port.get()
        self.arduino.ip = ip; self.arduino.port = port
        self._set_status(f"Connecting robot {ip}:{port}…")
        self.root.update()
        ok, err = self.arduino.connect()
        if ok:
            self._robot_status.config(
                text=f"Connected: {ip}:{port}", foreground=GREEN)
        else:
            self._robot_status.config(
                text=f"Failed: {err}", foreground=RED)
            messagebox.showerror("Robot Error",
                f"Cannot reach {ip}:{port}\n{err}")

    def _disconnect_robot(self):
        self.arduino.disconnect()
        self._robot_status.config(text="Disconnected", foreground=FG2)

    def _emergency_stop(self):
        if self.mission: self.mission.abort()
        self.arduino.send_stop()
        self._set_status("EMERGENCY STOP")
        self._lbl_state.config(text="State   : ABORTED")

    # ─────────────────────────────────────────────────────────
    #  DEBUG / TUNING
    # ─────────────────────────────────────────────────────────

    def _toggle_debug(self):
        global DEBUG_MOVEMENT
        DEBUG_MOVEMENT = self._v_debug.get()

    def _apply_tag_offset(self):
        global TAG_HEADING_OFFSET_DEG
        TAG_HEADING_OFFSET_DEG = float(self._v_tag_off.get())
        self._set_status(f"Tag heading offset = {TAG_HEADING_OFFSET_DEG}°")

    # ─────────────────────────────────────────────────────────
    #  ANCHOR OVERRIDE
    # ─────────────────────────────────────────────────────────

    def _on_preview_click(self, event):
        self._last_click_px = (event.x, event.y)

    def _set_anchor_from_click(self):
        if self._last_click_px is None:
            messagebox.showwarning("No click",
                "Snapshot first, then click on the camera preview")
            return
        col = int(self._v_anch_col.get())
        row = int(self._v_anch_row.get())
        cw  = self._cam_preview.winfo_width()
        ch  = self._cam_preview.winfo_height()
        cx_c, cy_c = self._last_click_px
        fw = self.calibration.get("resolution",(cw,ch))[0] if self.calibration else cw
        fh = self.calibration.get("resolution",(cw,ch))[1] if self.calibration else ch
        px = int(cx_c * fw / max(cw, 1))
        py = int(cy_c * fh / max(ch, 1))
        self.mapper.anchor_px   = (px, py)
        self.mapper.anchor_cell = (col, row)
        self._anch_lbl.config(
            text=f"Anchor: px({px},{py}) → cell({col},{row})")
        self._set_status(f"Anchor set: px({px},{py}) → banner cell({col},{row})")

    # ─────────────────────────────────────────────────────────
    #  MISSION CONTROL
    # ─────────────────────────────────────────────────────────

    def _start_mission(self):
        if not self.mapper.is_ready():
            messagebox.showerror("Not Ready",
                "Calibration not loaded or incomplete.\n"
                "Load a .npz file or set an anchor manually.")
            return
        if self.perception.cap is None:
            messagebox.showerror("No Camera", "Connect camera first.")
            return
        busy = {MissionState.DETECTING, MissionState.PLANNING,
                MissionState.COMMANDING, MissionState.WAITING}
        if self._mission_state in busy:
            messagebox.showwarning("Busy", "Mission already running.")
            return

        col = max(0, min(19, int(self._v_target_col.get())))
        row = max(0, min(19, int(self._v_target_row.get())))
        self._target_cell = (col, row)

        cfg = MissionConfig(
            target_cell    = (col, row),
            max_retries    = int(self._v_max_retry.get()),
            speed_cms      = max(1.0, float(self._v_speed_cms.get())),
            arrival_thresh = int(self._v_arrive_thr.get()),
            inflate_obs    = int(self._v_inflate.get()),
        )

        # Quick travel estimate (straight-line)
        if self._current_cell:
            dc = col - self._current_cell[0]
            dr = row - self._current_cell[1]
            dist_cm = math.hypot(dc, dr) * BANNER_CELL_CM
        else:
            dist_cm = 0.0
        est_s = dist_cm / max(cfg.speed_cms, 1.0)
        self._lbl_travel.config(
            text=f"Travel  : ≈{dist_cm:.0f}cm  ≈{est_s:.1f}s")
        self._lbl_target.config(text=f"Target  : ({col},{row})")

        if not self.arduino.connected:
            if not messagebox.askyesno("Simulation",
                "Robot not connected.\nRun in simulation mode?"):
                return

        self._stop_live_preview()
        self.mission = MissionRunner(
            perception      = self.perception,
            mapper          = self.mapper,
            arduino         = self.arduino,
            config          = cfg,
            on_state_change = self._on_mission_state,
            on_frame        = self._on_mission_frame,
            on_grid_update  = self._on_mission_grid,
        )
        self.mission.start()
        self._set_status(f"Mission → target ({col},{row})")

    def _abort_mission(self):
        if self.mission:
            self.mission.abort()

    # ─────────────────────────────────────────────────────────
    #  MISSION CALLBACKS  (background thread → main thread via after)
    # ─────────────────────────────────────────────────────────

    def _on_mission_state(self, state: MissionState, msg: str):
        self.root.after(0, self._apply_state, state, msg)

    def _apply_state(self, state: MissionState, msg: str):
        self._mission_state = state
        col_map = {
            MissionState.IDLE:       FG2,
            MissionState.SNAPSHOT:   FG,
            MissionState.DETECTING:  FG,
            MissionState.PLANNING:   FG,
            MissionState.COMMANDING: ORANGE,
            MissionState.WAITING:    ORANGE,
            MissionState.VERIFYING:  FG,
            MissionState.COMPLETE:   GREEN,
            MissionState.FAILED:     RED,
            MissionState.ABORTED:    RED,
        }
        c = col_map.get(state, FG)
        self._lbl_state.config(text=f"State   : {state.name}", foreground=c)
        self._lbl_msg.config(text=msg, foreground=c)
        self._hdr_status.config(text=f"{state.name}: {msg[:70]}")

        if state == MissionState.COMPLETE:
            messagebox.showinfo("Mission Complete", msg, parent=self.root)
            self._start_live_preview()
        elif state == MissionState.FAILED:
            messagebox.showerror("Mission Failed", msg, parent=self.root)
            self._start_live_preview()
        elif state == MissionState.ABORTED:
            self._start_live_preview()

    def _on_mission_frame(self, frame, robot, obstacles):
        self.root.after(0, self._apply_frame, frame, robot, obstacles)

    def _apply_frame(self, frame, robot, obstacles):
        self._update_from_detection(frame, robot, obstacles)

    def _on_mission_grid(self, obstacles, path, robot_cell, target_cell, theta):
        self.root.after(0, self._apply_grid,
                        obstacles, path, robot_cell, target_cell, theta)

    def _apply_grid(self, obstacles, path, robot_cell, target_cell, theta):
        self._obstacles   = obstacles
        self._path        = path
        self._robot_theta = theta
        if robot_cell:
            self._current_cell = robot_cell
            self._lbl_current.config(
                text=f"Robot   : {robot_cell}  θ={int(theta)}°")
        if path:
            cm = path_length_cm(
                [BannerMapper.cell_to_astar(*c) for c in path])
            self._lbl_path.config(text=f"Path    : {len(path)} cells  {cm:.0f}cm")
        self._update_grid()

    # ─────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────

    def _on_grid_click(self, col: int, row: int):
        self._v_target_col.set(col)
        self._v_target_row.set(row)
        self._target_cell = (col, row)
        self._lbl_target.config(text=f"Target  : ({col},{row})")
        self._update_grid()

    def _update_grid(self):
        self._grid_canvas.set_state(
            obstacles   = self._obstacles,
            path        = self._path,
            robot_cell  = self._current_cell,
            target_cell = self._target_cell,
            robot_theta = self._robot_theta,
        )

    def _update_from_detection(self, frame, robot, obstacles):
        self._cam_preview.show_frame(frame, robot,
                                     obstacles or [], self.mapper)
        if robot and robot.found:
            self._current_cell = robot.cell
            self._robot_theta  = corrected_theta(robot.theta_deg)
            self._lbl_current.config(
                text=f"Robot   : {robot.cell}  θ={int(self._robot_theta)}°",
                foreground=GREEN)
        elif robot is not None:
            self._lbl_current.config(
                text="Robot   : NOT DETECTED", foreground=RED)
        self._obstacles = {(o.col, o.row) for o in (obstacles or [])}
        self._update_grid()

    def _set_status(self, msg: str):
        self._hdr_status.config(text=msg)

    def _on_close(self):
        self._stop_live_preview()
        if self.mission: self.mission.abort()
        if self.arduino.connected: self.arduino.send_stop()
        if self.perception.cap: self.perception.cap.release()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    NavigationApp(root)
    root.mainloop()