"""
omnibot_navigation.py
═══════════════════════════════════════════════════════════════════════════════
Open-Loop Grid Navigation  ·  Omniwheel Robot  ·  20×20 Banner Arena

Mission cycle (open-loop — NO continuous camera feedback while driving)
────────────────────────────────────────────────────────────────────────
  1. Snapshot   – capture ONE frame, undistort with calibration
  2. Detect     – find AprilTag ID 0 (robot) + YOLO obstacles
  3. Localise   – convert tag pixel centre → banner grid cell
  4. Plan       – A* on 20×20 grid from current cell to target cell
  5. Command    – convert path to movement commands, send to Arduino
  6. Wait       – sleep for estimated travel time (path_cm / speed_cms)
  7. Verify     – take ONE fresh snapshot, detect robot again
  8. Check      – if current_cell == target_cell → mission complete
  9. Retry      – if not reached and retries remain, go to step 1

Banner coordinate system
─────────────────────────
  Origin (0,0) = bottom-left corner of the 20×20 banner
  col  0–19  → left to right
  row  0–19  → bottom to top   (image y is flipped: higher row = lower y_px)

  BannerMapper uses:
    • homography H  (world_cm → image_px)   from calibration file
    • anchor fallback (cm_per_pixel + anchor_px) for overhead cameras

Calibration file (.npz from camera_calibration.py)
────────────────────────────────────────────────────
  Required keys: camera_matrix, dist_coeffs
  Optional keys: homography, cm_per_pixel, anchor_px, anchor_cell,
                 resolution

Arduino HTTP protocol
──────────────────────
  GET http://<ip>:<port>/cmd?v=<cmd>
  Commands: F B L R A(FL) D(FR) Q(BL) E(BR) S(stop)
  Speed prefix: 1=slow 2=normal 3=fast  e.g. "2F"
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional
import heapq
import tkinter as tk

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
#  CONSTANTS
# ══════════════════════════════════════════════════════════════

BANNER_COLS     = 20
BANNER_ROWS     = 20
BANNER_CELL_CM  = 15.0          # physical size of each grid cell
BANNER_W_CM     = BANNER_COLS * BANNER_CELL_CM   # 300 cm
BANNER_H_CM     = BANNER_ROWS * BANNER_CELL_CM   # 300 cm

TAG_SIZE_CM     = 12.0          # AprilTag physical side length
ROBOT_TAG_ID    = 0             # AprilTag ID attached to the robot
ARRIVAL_THRESH  = 1             # cells — within this = "arrived"
DEFAULT_RETRIES = 5
DEFAULT_SPEED   = 20.0          # cm/s  (user-editable in GUI)
DWELL_BUFFER_S  = 0.5           # extra seconds added to travel time

DISPLAY_W       = 900
DISPLAY_H       = 680

DEFAULT_ROBOT_IP   = "192.168.4.1"
DEFAULT_ROBOT_PORT = 80

YOLO_CONF       = 0.45
YOLO_IOU        = 0.45
DEFAULT_YOLO_PATH = ""          # filled from GUI

# Obstacle inflation (cells)
OBSTACLE_INFLATE = 1

# UI palette
BG       = "#1a1a1e"
BG2      = "#25252b"
BG3      = "#2e2e36"
ACCENT   = "#00d4aa"
FG       = "#e8e8e8"
FG2      = "#aaaaaa"
GREEN    = "#00cc77"
RED      = "#ff4444"
ORANGE   = "#ff9933"
MONO     = ("Courier New", 9)
LABELF   = ("Segoe UI", 9)
HEADF    = ("Segoe UI", 10, "bold")
LEFT_W   = 380


# ══════════════════════════════════════════════════════════════
#  MISSION  STATE  MACHINE
# ══════════════════════════════════════════════════════════════

class MissionState(Enum):
    IDLE          = auto()
    SNAPSHOT      = auto()
    DETECTING     = auto()
    PLANNING      = auto()
    COMMANDING    = auto()
    WAITING       = auto()
    VERIFYING     = auto()
    COMPLETE      = auto()
    FAILED        = auto()
    ABORTED       = auto()


# ══════════════════════════════════════════════════════════════
#  BANNER  MAPPER
# ══════════════════════════════════════════════════════════════

class BannerMapper:
    """
    Converts between pixel coordinates and banner grid cells.

    Supports two modes (automatically selected):
      1. Homography mode  — uses H matrix from calibration file.
                            Most accurate; handles any camera angle.
      2. Anchor mode      — uses cm_per_pixel + anchor_px.
                            Works for perfectly overhead cameras.
                            Falls back to this when no homography saved.
    """

    def __init__(self):
        self.cm_per_pixel: Optional[float]           = None
        self.anchor_px:    Optional[tuple[int,int]]  = None
        self.anchor_cell:  tuple[int,int]            = (0, 0)
        self.homography:   Optional[np.ndarray]      = None   # world_cm→px
        self._H_inv:       Optional[np.ndarray]      = None   # px→world_cm

    def load_from_profile(self, profile: dict[str, Any]) -> None:
        """Populate mapper from a loaded calibration profile dict."""
        if "homography" in profile and profile["homography"] is not None:
            H = np.array(profile["homography"], dtype=np.float64)
            if H.shape == (3, 3):
                self.homography = H
                self._H_inv     = np.linalg.inv(H)

        if "cm_per_pixel" in profile:
            self.cm_per_pixel = float(profile["cm_per_pixel"])

        if "anchor_px" in profile:
            ap = profile["anchor_px"]
            self.anchor_px = (int(ap[0]), int(ap[1]))

        if "anchor_cell" in profile:
            ac = profile["anchor_cell"]
            self.anchor_cell = (int(ac[0]), int(ac[1]))

    def is_ready(self) -> bool:
        return (self.homography is not None or
                (self.cm_per_pixel is not None and self.anchor_px is not None))

    @property
    def px_per_cell(self) -> float:
        if self.cm_per_pixel and self.cm_per_pixel > 0:
            return BANNER_CELL_CM / self.cm_per_pixel
        return 0.0

    # ── Pixel → banner cell ───────────────────────────────────
    def px_to_cell(self, x_px: float, y_px: float) -> Optional[tuple[int,int]]:
        """Convert pixel position to (col, row) banner cell."""
        if self.homography is not None and self._H_inv is not None:
            return self._px_to_cell_homography(x_px, y_px)
        if self.cm_per_pixel and self.anchor_px:
            return self._px_to_cell_anchor(x_px, y_px)
        return None

    def _px_to_cell_homography(self, x_px: float, y_px: float) -> tuple[int,int]:
        """Use inverse homography to get world_cm then convert to cell."""
        pt  = np.array([[[x_px, y_px]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self._H_inv)
        x_cm, y_cm = float(res[0,0,0]), float(res[0,0,1])
        col = int(round(x_cm / BANNER_CELL_CM - 0.5))
        row = int(round(y_cm / BANNER_CELL_CM - 0.5))
        col = max(0, min(BANNER_COLS - 1, col))
        row = max(0, min(BANNER_ROWS - 1, row))
        return col, row

    def _px_to_cell_anchor(self, x_px: float, y_px: float) -> tuple[int,int]:
        """Linear anchor-based mapping (overhead camera only)."""
        ppc = self.px_per_cell
        if ppc <= 0:
            return (0, 0)
        ax, ay = self.anchor_px
        ac, ar = self.anchor_cell
        col = int(round(ac + (x_px - ax) / ppc))
        row = int(round(ar - (y_px - ay) / ppc))
        col = max(0, min(BANNER_COLS - 1, col))
        row = max(0, min(BANNER_ROWS - 1, row))
        return col, row

    # ── Banner cell → pixel centre ────────────────────────────
    def cell_to_px(self, col: int, row: int) -> Optional[tuple[int,int]]:
        """Convert banner (col, row) cell to pixel centre."""
        if self.homography is not None:
            return self._cell_to_px_homography(col, row)
        if self.cm_per_pixel and self.anchor_px:
            return self._cell_to_px_anchor(col, row)
        return None

    def _cell_to_px_homography(self, col: int, row: int) -> tuple[int,int]:
        x_cm = (col + 0.5) * BANNER_CELL_CM
        y_cm = (row + 0.5) * BANNER_CELL_CM
        pt   = np.array([[[x_cm, y_cm]]], dtype=np.float32)
        res  = cv2.perspectiveTransform(pt, self.homography)
        return int(round(float(res[0,0,0]))), int(round(float(res[0,0,1])))

    def _cell_to_px_anchor(self, col: int, row: int) -> tuple[int,int]:
        ppc = self.px_per_cell
        ax, ay = self.anchor_px
        ac, ar = self.anchor_cell
        x = int(round(ax + (col - ac) * ppc))
        y = int(round(ay - (row - ar) * ppc))
        return x, y

    # ── A* index helpers ─────────────────────────────────────
    @staticmethod
    def cell_to_astar(col: int, row: int) -> tuple[int,int]:
        """Banner cell → A* grid index (row 0 = top of image = row 19 banner)."""
        return (BANNER_ROWS - 1) - row, col

    @staticmethod
    def astar_to_cell(ar: int, ac: int) -> tuple[int,int]:
        return ac, (BANNER_ROWS - 1) - ar

    # ── cm_per_pixel from AprilTag (runtime fallback) ─────────
    def update_scale_from_tag(self, tag_edge_px: float) -> None:
        """If no calibration scale, derive cm/pixel from visible tag size."""
        if self.cm_per_pixel is None and tag_edge_px > 5:
            self.cm_per_pixel = TAG_SIZE_CM / tag_edge_px


# ══════════════════════════════════════════════════════════════
#  A*  PLANNER
# ══════════════════════════════════════════════════════════════

class AStarPlanner:
    """
    A* on the 20×20 banner grid.
    Grid indices: [astar_row, astar_col] where astar_row 0 = banner row 19.
    Cell value 0 = free, 1 = blocked.
    Diagonal moves cost √2.
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
        heap  = [(0.0, start)]
        came  = {}
        g     = {start: 0.0}
        while heap:
            _, cur = heapq.heappop(heap)
            if cur == goal:
                path = []
                while cur in came:
                    path.append(cur); cur = came[cur]
                path.append(start)
                return path[::-1]
            for dr, dc, cost in self._DIRS:
                nr, nc = cur[0]+dr, cur[1]+dc
                if not (0 <= nr < self.h and 0 <= nc < self.w): continue
                if self.grid[nr, nc] == 1: continue
                ng = g[cur] + cost
                nb = (nr, nc)
                if nb not in g or ng < g[nb]:
                    came[nb] = cur
                    g[nb]    = ng
                    f        = ng + self._heuristic(nb, goal)
                    heapq.heappush(heap, (f, nb))
        return None


def path_length_cm(path_astar: list[tuple]) -> float:
    """Total path length in cm (diagonal = √2 × cell_cm)."""
    total = 0.0
    for i in range(len(path_astar) - 1):
        dr = abs(path_astar[i+1][0] - path_astar[i][0])
        dc = abs(path_astar[i+1][1] - path_astar[i][1])
        total += (1.414 if dr and dc else 1.0) * BANNER_CELL_CM
    return total


def inflate_obstacles(grid: np.ndarray, radius: int) -> np.ndarray:
    """Dilate obstacle cells by radius to add safety margin."""
    if radius <= 0:
        return grid
    try:
        from scipy.ndimage import binary_dilation
        struct  = np.ones((2*radius+1, 2*radius+1), bool)
        return binary_dilation(grid, structure=struct).astype(np.uint8)
    except ImportError:
        out = grid.copy()
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                out |= np.roll(np.roll(grid, dr, 0), dc, 1)
        return out


# ══════════════════════════════════════════════════════════════
#  PERCEPTION  MODULE
# ══════════════════════════════════════════════════════════════

@dataclass
class RobotDetection:
    found:      bool
    px:         float = 0.0      # pixel x of tag centre
    py:         float = 0.0      # pixel y of tag centre
    theta_deg:  float = 0.0      # heading in degrees
    cell:       Optional[tuple[int,int]] = None   # banner (col, row)
    tag_edge_px: float = 0.0     # edge length in pixels (for scale)


@dataclass
class ObstacleCell:
    col: int
    row: int


class Perception:
    """
    Handles camera snapshot + AprilTag detection + YOLO obstacle detection.
    All detection is ONE-SHOT (single frame), never continuous.
    """

    def __init__(self):
        self.cap:         Optional[cv2.VideoCapture] = None
        self.camera_matrix: Optional[np.ndarray]    = None
        self.dist_coeffs:   Optional[np.ndarray]    = None

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
        """Open camera from index or URL. Returns True on success."""
        if self.cap:
            self.cap.release()
        import platform
        backend = cv2.CAP_DSHOW if (isinstance(source, int) and
                                     platform.system() == "Windows") else cv2.CAP_ANY
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
        Grab ONE fresh frame and undistort it.
        Reads twice to flush the buffer (avoids stale frame).
        """
        if not self.cap:
            return None
        self.cap.grab()          # discard buffered frame
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
        """Detect AprilTag ID 0, return RobotDetection."""
        if self.at_detector is None:
            return RobotDetection(found=False)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = self.at_detector.detect(gray)

        for d in dets:
            if int(d.tag_id) != ROBOT_TAG_ID:
                continue
            cx, cy = float(d.center[0]), float(d.center[1])
            corners = d.corners.astype(np.float32)
            c0, c1  = corners[0], corners[1]
            edge_px = float(np.linalg.norm(c1 - c0))
            theta   = float(np.degrees(
                np.arctan2(c1[1]-c0[1], c1[0]-c0[0]))) % 360.0

            # Update scale from tag if no calibration scale exists
            mapper.update_scale_from_tag(edge_px)

            cell = mapper.px_to_cell(cx, cy)
            return RobotDetection(
                found=True, px=cx, py=cy,
                theta_deg=theta, cell=cell, tag_edge_px=edge_px)

        return RobotDetection(found=False)

    def detect_obstacles(
        self,
        frame: np.ndarray,
        mapper: BannerMapper,
        robot_px: Optional[tuple[float,float]] = None,
    ) -> list[ObstacleCell]:
        """Run YOLO on frame; map obstacle bounding-box centres to banner cells."""
        if self.yolo_model is None or not mapper.is_ready():
            return []

        results = self.yolo_model(frame, conf=YOLO_CONF,
                                  iou=YOLO_IOU, verbose=False)
        obstacles: list[ObstacleCell] = []

        # Keep-out radius for robot mask (avoid marking robot as obstacle)
        robot_r_px = 0
        if robot_px and mapper.cm_per_pixel and mapper.cm_per_pixel > 0:
            robot_r_px = int(30.0 / mapper.cm_per_pixel)   # 30 cm radius

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1+x2)/2.0, (y1+y2)/2.0

                # Skip boxes overlapping the robot tag
                if robot_px:
                    dx, dy = cx - robot_px[0], cy - robot_px[1]
                    if math.hypot(dx, dy) < max(robot_r_px, 40):
                        continue

                cell = mapper.px_to_cell(cx, cy)
                if cell:
                    obstacles.append(ObstacleCell(*cell))

        return obstacles


# ══════════════════════════════════════════════════════════════
#  ARDUINO  HTTP  CLIENT
# ══════════════════════════════════════════════════════════════

class ArduinoClient:
    """
    Sends HTTP GET commands to the Arduino R4 WiFi.
    Non-blocking version posts to a daemon thread.
    """
    SPEED_PREFIX = {1: "1", 2: "2", 3: "3"}

    def __init__(self, ip: str = DEFAULT_ROBOT_IP,
                 port: int = DEFAULT_ROBOT_PORT):
        self.ip    = ip
        self.port  = port
        self.speed = 2          # 1=slow 2=normal 3=fast
        self._connected = False
        self._lock = threading.Lock()
        self.last_cmd  = ""
        self.last_ms   = 0.0

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> tuple[bool, str]:
        import requests as req
        try:
            r = req.get(f"http://{self.ip}:{self.port}/cmd?v=S", timeout=3)
            if r.status_code == 200:
                self._connected = True
                return True, ""
            return False, f"HTTP {r.status_code}"
        except Exception as e:
            return False, str(e)

    def disconnect(self):
        self._connected = False

    def send(self, cmd: str) -> tuple[str, float]:
        """Send command synchronously. Returns (status, ms)."""
        if not self._connected:
            return "NOT_CONNECTED", 0.0
        import requests as req
        full = f"{self.SPEED_PREFIX.get(self.speed,'2')}{cmd}" if cmd != "S" else "S"
        with self._lock:
            try:
                t0 = time.time()
                r  = req.get(f"http://{self.ip}:{self.port}/cmd?v={full}",
                             timeout=2)
                ms = (time.time()-t0)*1000
                self.last_cmd = full
                self.last_ms  = ms
                return ("OK" if r.status_code == 200
                        else f"ERR_{r.status_code}"), ms
            except Exception as e:
                return f"ERR:{e}", 0.0

    def send_stop(self):
        self.send("S")


# ══════════════════════════════════════════════════════════════
#  MOVEMENT  COMMAND  BUILDER
# ══════════════════════════════════════════════════════════════

def heading_to_cmd(dx_cell: int, dy_cell: int, robot_theta_deg: float) -> str:
    """
    Convert a grid step (dx_cell, dy_cell) relative to the banner into
    a robot-frame movement command, accounting for heading.

    Banner frame:  dx > 0 = rightward, dy > 0 = upward (row increases up)
    Image frame:   dy > 0 = downward — we flip dy for world angle.

    Commands: F B L R A(FL) D(FR) Q(BL) E(BR)
    """
    # World angle of desired movement (0° = right, 90° = up)
    world_angle = math.degrees(math.atan2(dy_cell, dx_cell)) % 360.0
    # Robot forward direction in world frame
    robot_fwd   = (-robot_theta_deg) % 360.0
    # Relative angle
    rel = (world_angle - robot_fwd) % 360.0

    if   rel <  22.5 or rel >= 337.5: return "F"
    elif rel <  67.5:                 return "D"   # forward-right
    elif rel < 112.5:                 return "R"
    elif rel < 157.5:                 return "E"   # backward-right
    elif rel < 202.5:                 return "B"
    elif rel < 247.5:                 return "Q"   # backward-left
    elif rel < 292.5:                 return "L"
    else:                             return "A"   # forward-left


def build_command_sequence(
    path_cells: list[tuple[int,int]],    # banner (col, row) list
    robot_theta_deg: float,
) -> list[tuple[str, float]]:
    """
    Convert a path in banner cells into a list of (command, duration_s) pairs.
    Each step takes BANNER_CELL_CM / speed_cms seconds (caller adjusts speed).

    Returns list of (cmd, fraction_of_step) — duration is normalised to 1
    step each; caller multiplies by actual step_time_s.

    Diagonal steps are treated as 1 combined move (not split into H+V).
    """
    cmds: list[tuple[str, float]] = []
    for i in range(len(path_cells) - 1):
        col0, row0 = path_cells[i]
        col1, row1 = path_cells[i+1]
        dx = col1 - col0   # +right
        dy = row1 - row0   # +up (banner frame)
        cmd = heading_to_cmd(dx, dy, robot_theta_deg)
        is_diag = (dx != 0 and dy != 0)
        cost    = 1.414 if is_diag else 1.0
        cmds.append((cmd, cost))
    return cmds


# ══════════════════════════════════════════════════════════════
#  GRID  CANVAS  WIDGET
# ══════════════════════════════════════════════════════════════

class GridCanvas(tk.Canvas):
    """
    Renders the 20×20 banner grid with:
      • obstacle cells (red fill)
      • path cells (cyan fill)
      • robot position (magenta dot + heading arrow)
      • target cell (green outline)
    Cells are mapped to fixed pixel grid — no live camera feed here.
    """

    CELL_PX = 32   # pixels per grid cell in the canvas view

    def __init__(self, parent, **kwargs):
        cw = BANNER_COLS * self.CELL_PX
        ch = BANNER_ROWS * self.CELL_PX
        super().__init__(parent, width=cw, height=ch,
                         bg="#0a0a0c", highlightthickness=1,
                         highlightbackground=ACCENT, **kwargs)
        self._obstacles: set[tuple[int,int]] = set()
        self._path:      list[tuple[int,int]] = []
        self._robot_cell: Optional[tuple[int,int]] = None
        self._target_cell: Optional[tuple[int,int]] = None
        self._robot_theta: float = 0.0

        self.bind("<Button-1>", self._on_click)
        self._click_callback = None   # called with (col, row)
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

    def _cell_rect(self, col: int, row: int) -> tuple[int,int,int,int]:
        """Return (x1,y1,x2,y2) for a banner cell in canvas pixels.
        Banner row 0 = bottom → canvas y near bottom."""
        cp = self.CELL_PX
        x1 = col * cp
        y1 = (BANNER_ROWS - 1 - row) * cp   # flip y
        return x1, y1, x1+cp, y1+cp

    def _draw(self):
        self.delete("all")
        cp = self.CELL_PX

        path_set = set(self._path)

        for row in range(BANNER_ROWS):
            for col in range(BANNER_COLS):
                x1, y1, x2, y2 = self._cell_rect(col, row)
                cell = (col, row)

                if cell in self._obstacles:
                    fill = "#cc2222"
                elif cell in path_set:
                    fill = "#004455"
                else:
                    fill = "#141418"

                self.create_rectangle(x1, y1, x2, y2,
                                      fill=fill, outline="#2a2a32", width=1)

                # Coordinate label (small, every 5 cells)
                if col % 5 == 0 and row % 5 == 0:
                    self.create_text(x1+3, y2-4, text=f"{col},{row}",
                                     font=("Courier New", 6), fill="#444455",
                                     anchor=tk.SW)

        # Target cell outline
        if self._target_cell:
            tc, tr = self._target_cell
            x1, y1, x2, y2 = self._cell_rect(tc, tr)
            self.create_rectangle(x1+2, y1+2, x2-2, y2-2,
                                   outline="#00ff88", width=3)
            self.create_text((x1+x2)//2, (y1+y2)//2, text="T",
                              font=("Segoe UI", 8, "bold"), fill="#00ff88")

        # Path waypoints
        if len(self._path) > 1:
            for i in range(len(self._path)-1):
                c0, r0 = self._path[i]
                c1, r1 = self._path[i+1]
                x1,y1,x2,y2 = self._cell_rect(c0, r0)
                mx0, my0 = (x1+x2)//2, (y1+y2)//2
                x1,y1,x2,y2 = self._cell_rect(c1, r1)
                mx1, my1 = (x1+x2)//2, (y1+y2)//2
                self.create_line(mx0, my0, mx1, my1,
                                 fill="#00d4aa", width=2, arrow=tk.LAST)

        # Robot position
        if self._robot_cell:
            rc, rr = self._robot_cell
            x1, y1, x2, y2 = self._cell_rect(rc, rr)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            r = cp // 3
            self.create_oval(cx-r, cy-r, cx+r, cy+r,
                              fill="#ff00ff", outline="#ffffff", width=2)
            # Heading arrow
            angle_rad = math.radians(-self._robot_theta)   # canvas y flipped
            ex = cx + int(r * 1.6 * math.cos(angle_rad))
            ey = cy + int(r * 1.6 * math.sin(angle_rad))
            self.create_line(cx, cy, ex, ey,
                             fill="#ffffff", width=2, arrow=tk.LAST)

    def _on_click(self, event):
        cp = self.CELL_PX
        col = event.x // cp
        row = BANNER_ROWS - 1 - (event.y // cp)
        col = max(0, min(BANNER_COLS-1, col))
        row = max(0, min(BANNER_ROWS-1, row))
        if self._click_callback:
            self._click_callback(col, row)


# ══════════════════════════════════════════════════════════════
#  CAMERA  PREVIEW  CANVAS
# ══════════════════════════════════════════════════════════════

class CameraPreview(tk.Canvas):
    """Shows the last captured frame with detection overlays."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#0a0a0c",
                         highlightthickness=0, **kwargs)
        self._imgtk = None

    def show_frame(
        self,
        frame: np.ndarray,
        robot: Optional[RobotDetection] = None,
        obstacles: Optional[list[ObstacleCell]] = None,
        mapper: Optional["BannerMapper"] = None,
    ):
        """Render frame with overlay annotations."""
        img = frame.copy()
        h_img, w_img = img.shape[:2]

        # Draw banner grid overlay if mapper ready
        if mapper and mapper.is_ready():
            self._draw_grid_overlay(img, mapper)

        # Draw obstacles
        if obstacles and mapper and mapper.is_ready():
            for obs in obstacles:
                px = mapper.cell_to_px(obs.col, obs.row)
                if px:
                    cv2.circle(img, px, 12, (0, 60, 220), -1)
                    cv2.circle(img, px, 14, (0, 0, 180), 2)

        # Draw robot tag
        if robot and robot.found:
            cx, cy = int(robot.px), int(robot.py)
            cv2.circle(img, (cx, cy), 18, (255, 0, 255), 2)
            cv2.drawMarker(img, (cx, cy), (255, 0, 255),
                           cv2.MARKER_DIAMOND, 14, 2)
            # Heading arrow
            r = math.radians(robot.theta_deg)
            ex = int(cx + 35 * math.cos(r))
            ey = int(cy + 35 * math.sin(r))
            cv2.arrowedLine(img, (cx, cy), (ex, ey),
                            (255, 100, 255), 2, tipLength=0.4)
            cell_str = f"cell({robot.cell[0]},{robot.cell[1]})" if robot.cell else ""
            cv2.putText(img, f"ROBOT  {cell_str}  θ={int(robot.theta_deg)}°",
                        (cx+20, cy-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 255), 2)

        # Resize to canvas
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw > 10 and ch > 10:
            img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)
        rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self._imgtk = imgtk

    def _draw_grid_overlay(self, img: np.ndarray, mapper: "BannerMapper"):
        h_img, w_img = img.shape[:2]
        for col in range(BANNER_COLS + 1):
            pt0 = mapper.cell_to_px(col, 0)
            pt1 = mapper.cell_to_px(col, BANNER_ROWS - 1)
            if pt0 and pt1:
                if (0 <= pt0[0] < w_img or 0 <= pt1[0] < w_img):
                    cv2.line(img, pt0, pt1, (0, 100, 50), 1)
        for row in range(BANNER_ROWS + 1):
            pt0 = mapper.cell_to_px(0, row)
            pt1 = mapper.cell_to_px(BANNER_COLS - 1, row)
            if pt0 and pt1:
                cv2.line(img, pt0, pt1, (0, 100, 50), 1)


# ══════════════════════════════════════════════════════════════
#  MISSION  RUNNER
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
    success:        bool
    final_cell:     Optional[tuple[int,int]]
    attempts:       int
    message:        str
    path_history:   list[list[tuple[int,int]]] = field(default_factory=list)


class MissionRunner:
    """
    Executes the open-loop capture → detect → plan → command → wait → verify
    cycle.  Runs in a background thread; reports progress via callbacks.
    """

    def __init__(
        self,
        perception: Perception,
        mapper:     BannerMapper,
        arduino:    ArduinoClient,
        config:     MissionConfig,
        on_state_change,    # callable(MissionState, str)
        on_frame,           # callable(frame, robot, obstacles)
        on_grid_update,     # callable(obstacles, path, robot_cell, target_cell, theta)
    ):
        self.perception   = perception
        self.mapper       = mapper
        self.arduino      = arduino
        self.config       = config
        self._on_state    = on_state_change
        self._on_frame    = on_frame
        self._on_grid     = on_grid_update
        self._abort       = threading.Event()
        self._thread:     Optional[threading.Thread] = None

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
        result  = MissionResult(success=False, final_cell=None,
                                 attempts=0, message="Not started")

        while attempt <= cfg.max_retries and not self._abort.is_set():
            attempt += 1
            self._state(MissionState.SNAPSHOT,
                        f"Attempt {attempt}/{cfg.max_retries+1} — capturing frame…")

            # ── Step 1: Snapshot ───────────────────────────────
            frame = self.perception.snapshot()
            if frame is None:
                result = MissionResult(False, None, attempt,
                                       "Camera error: no frame captured")
                self._state(MissionState.FAILED, result.message)
                return

            # ── Step 2: Detect ─────────────────────────────────
            self._state(MissionState.DETECTING, "Detecting robot and obstacles…")
            robot = self.perception.detect_robot(frame, self.mapper)

            if not robot.found:
                result = MissionResult(False, None, attempt,
                                       "Robot not detected (AprilTag ID 0 missing)")
                self._state(MissionState.FAILED, result.message)
                self._on_frame(frame, None, [])
                return

            obstacles = self.perception.detect_obstacles(
                frame, self.mapper,
                robot_px=(robot.px, robot.py) if robot.found else None,
            )
            self._on_frame(frame, robot, obstacles)

            # ── Step 3: Localise ───────────────────────────────
            current_cell = robot.cell
            if current_cell is None:
                result = MissionResult(False, None, attempt,
                                       "Cannot map robot pixel to banner cell")
                self._state(MissionState.FAILED, result.message)
                return

            # Check arrival before planning (could already be there)
            if self._arrived(current_cell, cfg.target_cell, cfg.arrival_thresh):
                result = MissionResult(True, current_cell, attempt,
                                       f"Arrived at {cfg.target_cell} in {attempt} attempt(s)")
                self._state(MissionState.COMPLETE, result.message)
                self._on_grid(
                    {(o.col, o.row) for o in obstacles},
                    [], current_cell, cfg.target_cell, robot.theta_deg)
                return

            # ── Step 4: Plan ───────────────────────────────────
            self._state(MissionState.PLANNING,
                        f"Planning A* from {current_cell} → {cfg.target_cell}…")
            grid = self._build_grid(obstacles)

            # Clear robot's own cell and target cell in grid
            sr, sc = BannerMapper.cell_to_astar(*current_cell)
            er, ec = BannerMapper.cell_to_astar(*cfg.target_cell)
            grid[sr, sc] = 0
            grid[er, ec] = 0

            if grid[er, ec] == 1:
                result = MissionResult(False, current_cell, attempt,
                                       f"Target cell {cfg.target_cell} is blocked")
                self._state(MissionState.FAILED, result.message)
                return

            path_astar = AStarPlanner(grid).plan((sr, sc), (er, ec))
            if path_astar is None:
                result = MissionResult(False, current_cell, attempt,
                                       "A*: no path found — all routes blocked")
                self._state(MissionState.FAILED, result.message)
                return

            # Convert A* indices back to banner cells for display
            path_cells = [BannerMapper.astar_to_cell(r, c)
                          for r, c in path_astar]

            total_cm  = path_length_cm(path_astar)
            travel_s  = total_cm / max(cfg.speed_cms, 1.0) + DWELL_BUFFER_S
            step_s    = BANNER_CELL_CM / max(cfg.speed_cms, 1.0)

            self._on_grid(
                {(o.col, o.row) for o in obstacles},
                path_cells, current_cell, cfg.target_cell, robot.theta_deg)

            # ── Step 5 & 6: Command → Wait ─────────────────────
            self._state(MissionState.COMMANDING,
                        f"Sending {len(path_cells)-1} moves  "
                        f"({total_cm:.0f} cm  ≈{travel_s:.1f}s)…")

            if not self.arduino.connected:
                # Simulation mode — just wait
                self._state(MissionState.WAITING,
                            f"[SIM] Waiting {travel_s:.1f}s…")
                self._interruptible_sleep(travel_s)
            else:
                # Send each command with per-step timing
                cmds = build_command_sequence(path_cells, robot.theta_deg)
                for cmd, cost_factor in cmds:
                    if self._abort.is_set():
                        break
                    self.arduino.send(cmd)
                    self._interruptible_sleep(step_s * cost_factor)
                # Stop robot
                self.arduino.send_stop()
                # Extra dwell
                self._interruptible_sleep(DWELL_BUFFER_S)

            if self._abort.is_set():
                result = MissionResult(False, current_cell, attempt,
                                       "Mission aborted by user")
                self._state(MissionState.ABORTED, result.message)
                return

            # ── Step 7 & 8: Verify ─────────────────────────────
            self._state(MissionState.VERIFYING, "Verifying position…")
            vframe = self.perception.snapshot()
            if vframe is None:
                result = MissionResult(False, current_cell, attempt,
                                       "Verification failed: no frame")
                self._state(MissionState.FAILED, result.message)
                return

            vrobot = self.perception.detect_robot(vframe, self.mapper)
            vobs   = self.perception.detect_obstacles(
                vframe, self.mapper,
                robot_px=(vrobot.px, vrobot.py) if vrobot.found else None)
            self._on_frame(vframe, vrobot, vobs)

            if not vrobot.found or vrobot.cell is None:
                # Cannot verify — assume moved, retry
                self._state(MissionState.SNAPSHOT,
                            "Cannot verify (tag lost) — retrying…")
                continue

            current_cell = vrobot.cell
            self._on_grid(
                {(o.col, o.row) for o in vobs},
                [], current_cell, cfg.target_cell, vrobot.theta_deg)

            # ── Step 9: Check / Retry ──────────────────────────
            if self._arrived(current_cell, cfg.target_cell, cfg.arrival_thresh):
                result = MissionResult(True, current_cell, attempt,
                                       f"✓ Arrived at {cfg.target_cell} "
                                       f"after {attempt} attempt(s)")
                self._state(MissionState.COMPLETE, result.message)
                return

            # Not arrived — loop back for retry
            self._state(MissionState.SNAPSHOT,
                        f"Not at target yet (at {current_cell}) — replanning…")

        # Exhausted retries
        result = MissionResult(False, current_cell if 'current_cell' in dir()
                               else None, attempt,
                               f"Max retries ({cfg.max_retries}) exhausted — "
                               f"stopped at {current_cell}")
        self._state(MissionState.FAILED, result.message)

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _arrived(current: tuple, target: tuple, thresh: int) -> bool:
        return (abs(current[0]-target[0]) <= thresh and
                abs(current[1]-target[1]) <= thresh)

    @staticmethod
    def _build_grid(obstacles: list[ObstacleCell]) -> np.ndarray:
        grid = np.zeros((BANNER_ROWS, BANNER_COLS), dtype=np.uint8)
        for obs in obstacles:
            ar, ac = BannerMapper.cell_to_astar(obs.col, obs.row)
            if 0 <= ar < BANNER_ROWS and 0 <= ac < BANNER_COLS:
                grid[ar, ac] = 1
        if OBSTACLE_INFLATE > 0:
            grid = inflate_obstacles(grid, OBSTACLE_INFLATE)
        return grid

    def _interruptible_sleep(self, seconds: float):
        """Sleep that can be interrupted by abort event."""
        end = time.time() + seconds
        while time.time() < end and not self._abort.is_set():
            time.sleep(0.05)

    def _state(self, state: MissionState, msg: str):
        self._on_state(state, msg)


# ══════════════════════════════════════════════════════════════
#  CALIBRATION  LOADER
# ══════════════════════════════════════════════════════════════

def load_calibration_npz(path: str) -> Optional[dict[str, Any]]:
    """
    Load a .npz calibration file saved by camera_calibration.py.
    Returns a plain dict of all keys, or None on error.
    """
    try:
        raw = np.load(path, allow_pickle=True)
        out: dict[str, Any] = {}
        for key in raw.files:
            val = raw[key]
            if val.ndim == 0:
                out[key] = val.item()
            elif key in ("camera_matrix", "dist_coeffs",
                         "homography", "banner_corners_px"):
                out[key] = val
            elif key in ("resolution", "anchor_px", "anchor_cell"):
                out[key] = tuple(int(x) for x in val.tolist())
            elif key in ("cm_per_pixel", "px_per_cell", "rms_error",
                         "height_cm", "tilt_deg"):
                out[key] = float(val)
            elif key in ("grid_cols", "grid_rows", "n_frames"):
                out[key] = int(val)
            else:
                try:    out[key] = str(val)
                except: out[key] = val
        return out
    except Exception as e:
        print(f"[load_calibration_npz] {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  MAIN  GUI
# ══════════════════════════════════════════════════════════════

def _apply_style(root: tk.Tk):
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=BG, foreground=FG,
                fieldbackground=BG2, troughcolor=BG2,
                borderwidth=0, relief="flat")
    s.configure("TFrame",        background=BG)
    s.configure("TLabelframe",   background=BG2, foreground=ACCENT,
                borderwidth=1, relief="solid")
    s.configure("TLabelframe.Label", background=BG2, foreground=ACCENT,
                font=HEADF)
    s.configure("TButton",       background=BG3, foreground=FG,
                padding=(8,5), font=LABELF)
    s.map("TButton",
          background=[("active", ACCENT), ("pressed", BG2)],
          foreground=[("active", BG)])
    s.configure("Accent.TButton", background=ACCENT, foreground=BG,
                font=("Segoe UI", 9, "bold"))
    s.map("Accent.TButton",
          background=[("active","#00ffcc"),("pressed","#009977")])
    s.configure("Stop.TButton",  background=RED, foreground=FG,
                font=("Segoe UI", 9, "bold"))
    s.map("Stop.TButton",
          background=[("active","#ff8888"),("pressed","#aa0000")])
    s.configure("TLabel",  background=BG,  foreground=FG,  font=LABELF)
    s.configure("Dim.TLabel", background=BG, foreground=FG2, font=LABELF)
    s.configure("Mono.TLabel", background=BG, foreground=ACCENT, font=MONO)
    s.configure("TEntry",  fieldbackground=BG2, foreground=FG,
                insertcolor=ACCENT, font=LABELF)
    s.configure("TSpinbox", fieldbackground=BG2, foreground=FG,
                arrowcolor=ACCENT, font=LABELF)
    s.configure("TScrollbar", background=BG2, arrowcolor=ACCENT)
    s.configure("TSeparator", background=BG3)


class NavigationApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Omnibot  ·  Open-Loop Grid Navigation")
        self.root.configure(bg=BG)
        self.root.state("zoomed")
        _apply_style(root)

        # ── Core modules ──────────────────────────────────────
        self.perception  = Perception()
        self.mapper      = BannerMapper()
        self.arduino     = ArduinoClient()
        self.calibration: Optional[dict] = None
        self.mission:     Optional[MissionRunner] = None

        # ── GUI state ─────────────────────────────────────────
        self._v_robot_ip   = tk.StringVar(value=DEFAULT_ROBOT_IP)
        self._v_robot_port = tk.IntVar(value=DEFAULT_ROBOT_PORT)
        self._v_target_col = tk.IntVar(value=10)
        self._v_target_row = tk.IntVar(value=10)
        self._v_speed_cms  = tk.DoubleVar(value=DEFAULT_SPEED)
        self._v_max_retry  = tk.IntVar(value=DEFAULT_RETRIES)
        self._v_cam_src    = tk.StringVar(value="0")
        self._v_yolo_path  = tk.StringVar(value=DEFAULT_YOLO_PATH)
        self._v_arrive_thr = tk.IntVar(value=ARRIVAL_THRESH)

        self._current_cell:  Optional[tuple[int,int]] = None
        self._target_cell:   Optional[tuple[int,int]] = None
        self._obstacles:     set[tuple[int,int]]      = set()
        self._path:          list[tuple[int,int]]     = []
        self._robot_theta:   float                    = 0.0
        self._mission_state: MissionState             = MissionState.IDLE

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Startup: load calibration file ────────────────────
        root.after(200, self._startup_load_calibration)

    # ─────────────────────────────────────────────────────────
    #  UI  BUILD
    # ─────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=0, minsize=LEFT_W)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=0)
        self.root.rowconfigure(0, weight=1)

        # ── Header ────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=BG3, height=46)
        hdr.grid(row=0, column=0, columnspan=3,
                 sticky="new")
        tk.Label(hdr, text="⬡  OMNIBOT  OPEN-LOOP NAVIGATOR",
                 font=("Segoe UI",13,"bold"),
                 bg=BG3, fg=ACCENT).pack(side=tk.LEFT, padx=18, pady=10)
        self._header_status = tk.Label(hdr, text="Startup…",
                                       font=LABELF, bg=BG3, fg=FG2)
        self._header_status.pack(side=tk.RIGHT, padx=18)

        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)

        # ── Left panel (scrollable) ───────────────────────────
        lc = tk.Frame(self.root, bg=BG, width=LEFT_W)
        lc.grid(row=1, column=0, sticky="nsew", padx=(6,4), pady=6)
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

        # ── Centre: camera preview ────────────────────────────
        cf = ttk.LabelFrame(self.root, text="Camera Snapshot", padding=4)
        cf.grid(row=1, column=1, sticky="nsew", padx=4, pady=6)
        cf.rowconfigure(0, weight=1)
        cf.columnconfigure(0, weight=1)
        self._cam_preview = CameraPreview(cf)
        self._cam_preview.grid(row=0, column=0, sticky="nsew")

        # ── Right: grid canvas ────────────────────────────────
        gf = ttk.LabelFrame(self.root, text="Banner Grid  (20×20)", padding=4)
        gf.grid(row=1, column=2, sticky="nsew", padx=(4,6), pady=6)
        gf.rowconfigure(0, weight=1)
        gf.columnconfigure(0, weight=1)
        self._grid_canvas = GridCanvas(gf)
        self._grid_canvas.grid(row=0, column=0, sticky="nsew")
        self._grid_canvas._click_callback = self._on_grid_click

        # ── Build left panel sections ─────────────────────────
        self._build_sec_calibration(left)
        self._build_sec_camera(left)
        self._build_sec_robot(left)
        self._build_sec_mission(left)
        self._build_sec_status(left)
        self._build_sec_manual(left)

    # ── Section builders ──────────────────────────────────────

    def _sep(self, p):
        ttk.Separator(p, orient="horizontal").pack(fill=tk.X, pady=4)

    def _btn(self, p, text, cmd, style="TButton"):
        b = ttk.Button(p, text=text, command=cmd, style=style)
        b.pack(fill=tk.X, pady=2)
        return b

    def _row_entry(self, p, label, var, w=14):
        r = ttk.Frame(p); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text=label, width=18, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=var, width=w).pack(side=tk.LEFT, padx=4)

    def _row_spin(self, p, label, var, frm, to, inc=1.0, w=8):
        r = ttk.Frame(p); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text=label, width=18, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(r, textvariable=var, from_=frm, to=to,
                    increment=inc, width=w).pack(side=tk.LEFT, padx=4)

    def _mono_lbl(self, p, text="", **kw):
        l = ttk.Label(p, text=text, style="Mono.TLabel",
                      wraplength=340, **kw)
        l.pack(anchor=tk.W, pady=1)
        return l

    # ── Section: calibration ──────────────────────────────────
    def _build_sec_calibration(self, left):
        sf = ttk.LabelFrame(left, text="📐  Calibration Profile", padding=8)
        sf.pack(fill=tk.X, pady=4)
        self._calib_lbl = ttk.Label(
            sf, text="No calibration loaded",
            font=MONO, background=BG2, foreground=RED,
            wraplength=340, justify=tk.LEFT)
        self._calib_lbl.pack(anchor=tk.W, pady=2)
        self._btn(sf, "📂  Load Calibration File (.npz)",
                  self._load_calibration)

    # ── Section: camera ───────────────────────────────────────
    def _build_sec_camera(self, left):
        sf = ttk.LabelFrame(left, text="📷  Camera", padding=8)
        sf.pack(fill=tk.X, pady=4)
        self._row_entry(sf, "Source (index/URL):", self._v_cam_src, 14)
        ttk.Label(sf, text="DroidCam: http://IP:4747/video",
                  style="Dim.TLabel").pack(anchor=tk.W)
        self._btn(sf, "▶  Connect Camera", self._connect_camera, "Accent.TButton")
        self._btn(sf, "📸  Take Snapshot Now", self._manual_snapshot)
        self._cam_status = self._mono_lbl(sf, "Not connected")

        self._sep(sf)
        ttk.Label(sf, text="YOLO model path (optional):",
                  style="Dim.TLabel").pack(anchor=tk.W)
        r = ttk.Frame(sf); r.pack(fill=tk.X, pady=2)
        ttk.Entry(r, textvariable=self._v_yolo_path, width=22).pack(
            side=tk.LEFT, padx=(0,4))
        ttk.Button(r, text="📁", command=self._browse_yolo,
                   width=3).pack(side=tk.LEFT)
        self._btn(sf, "Load YOLO Model", self._load_yolo)
        self._yolo_status = self._mono_lbl(sf, "YOLO: not loaded")

    # ── Section: robot / arduino ──────────────────────────────
    def _build_sec_robot(self, left):
        sf = ttk.LabelFrame(left, text="🤖  Robot (Arduino R4 WiFi)", padding=8)
        sf.pack(fill=tk.X, pady=4)
        self._row_entry(sf, "Robot IP:", self._v_robot_ip, 14)
        self._row_spin(sf, "Port:", self._v_robot_port, 1, 99999, 1)
        bf = ttk.Frame(sf); bf.pack(fill=tk.X, pady=2)
        ttk.Button(bf, text="Connect", style="Accent.TButton",
                   command=self._connect_robot).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(bf, text="Disconnect",
                   command=self._disconnect_robot).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(bf, text="⛔ STOP",
                   style="Stop.TButton",
                   command=self._emergency_stop).pack(side=tk.LEFT)
        self._robot_status = self._mono_lbl(sf, "Not connected")

        self._sep(sf)
        ttk.Label(sf, text="Speed note: set Arduino speed via prefix (1/2/3).",
                  style="Dim.TLabel", wraplength=330).pack(anchor=tk.W)

    # ── Section: mission ──────────────────────────────────────
    def _build_sec_mission(self, left):
        sf = ttk.LabelFrame(left, text="🎯  Mission", padding=8)
        sf.pack(fill=tk.X, pady=4)

        ttk.Label(sf, text="Target cell (col, row)  0–19:",
                  style="Dim.TLabel").pack(anchor=tk.W)
        tf = ttk.Frame(sf); tf.pack(fill=tk.X, pady=3)
        ttk.Label(tf, text="Col:").pack(side=tk.LEFT)
        ttk.Spinbox(tf, textvariable=self._v_target_col,
                    from_=0, to=19, width=4).pack(side=tk.LEFT, padx=4)
        ttk.Label(tf, text="Row:").pack(side=tk.LEFT, padx=(6,0))
        ttk.Spinbox(tf, textvariable=self._v_target_row,
                    from_=0, to=19, width=4).pack(side=tk.LEFT, padx=4)
        ttk.Label(sf, text="Or click directly on the grid →",
                  style="Dim.TLabel").pack(anchor=tk.W, pady=(0,4))

        self._row_spin(sf, "Speed (cm/s):", self._v_speed_cms, 1, 200, 1)
        self._row_spin(sf, "Max retries:", self._v_max_retry, 0, 20, 1)
        self._row_spin(sf, "Arrival thresh (cells):",
                       self._v_arrive_thr, 0, 5, 1)

        self._sep(sf)
        self._btn(sf, "🚀  START MISSION", self._start_mission, "Accent.TButton")
        self._btn(sf, "⛔  ABORT MISSION", self._abort_mission, "Stop.TButton")

    # ── Section: status dashboard ─────────────────────────────
    def _build_sec_status(self, left):
        sf = ttk.LabelFrame(left, text="📊  Mission Status", padding=8)
        sf.pack(fill=tk.X, pady=4)

        self._lbl_state    = self._mono_lbl(sf, "State   : IDLE")
        self._lbl_current  = self._mono_lbl(sf, "Robot   : –")
        self._lbl_target   = self._mono_lbl(sf, "Target  : –")
        self._lbl_path     = self._mono_lbl(sf, "Path    : –")
        self._lbl_travel   = self._mono_lbl(sf, "Travel  : –")
        self._lbl_attempt  = self._mono_lbl(sf, "Attempt : –")
        self._lbl_msg      = ttk.Label(sf, text="",
                                       font=LABELF, background=BG2,
                                       foreground=FG2, wraplength=340,
                                       justify=tk.LEFT)
        self._lbl_msg.pack(anchor=tk.W, pady=2)

    # ── Section: manual snapshot controls ─────────────────────
    def _build_sec_manual(self, left):
        sf = ttk.LabelFrame(left, text="🔧  Anchor Override (if no homography)",
                            padding=8)
        sf.pack(fill=tk.X, pady=4)
        ttk.Label(sf,
                  text="If calibration has no homography, set anchor manually:\n"
                       "Snapshot → click a known cell on the preview → enter its col/row.",
                  style="Dim.TLabel", wraplength=340).pack(anchor=tk.W)
        self._v_anch_col = tk.IntVar(value=0)
        self._v_anch_row = tk.IntVar(value=0)
        af = ttk.Frame(sf); af.pack(fill=tk.X, pady=3)
        ttk.Label(af, text="Anchor col:").pack(side=tk.LEFT)
        ttk.Spinbox(af, textvariable=self._v_anch_col,
                    from_=0, to=19, width=4).pack(side=tk.LEFT, padx=4)
        ttk.Label(af, text="Row:").pack(side=tk.LEFT, padx=(6,0))
        ttk.Spinbox(af, textvariable=self._v_anch_row,
                    from_=0, to=19, width=4).pack(side=tk.LEFT, padx=4)
        self._btn(sf, "📌  Set Anchor from Last Snapshot Click",
                  self._set_anchor_from_click)
        self._anch_lbl = self._mono_lbl(sf, "Anchor: not set")
        self._last_click_px: Optional[tuple[int,int]] = None
        self._cam_preview.bind("<Button-1>", self._on_preview_click)

    # ─────────────────────────────────────────────────────────
    #  STARTUP  CALIBRATION  LOAD
    # ─────────────────────────────────────────────────────────

    def _startup_load_calibration(self):
        """Prompt user to select a .npz calibration file at startup."""
        messagebox.showinfo(
            "Load Calibration",
            "Please select your camera calibration (.npz) file.\n\n"
            "This was saved by camera_calibration.py and contains\n"
            "the camera matrix, distortion coefficients, and banner\n"
            "mapping (homography / anchor / cm_per_pixel).",
            parent=self.root)
        self._load_calibration()

    def _load_calibration(self):
        path = filedialog.askopenfilename(
            title="Select calibration profile (.npz)",
            filetypes=[("Calibration profile","*.npz"),
                       ("NumPy archive","*.npz"),
                       ("All","*.*")],
            parent=self.root)
        if not path:
            self._set_status("No calibration file selected")
            return

        data = load_calibration_npz(path)
        if data is None:
            messagebox.showerror("Error",
                f"Cannot load calibration file:\n{path}")
            return

        self.calibration = data

        # ── Apply to Perception (undistortion) ────────────────
        if "camera_matrix" in data:
            self.perception.camera_matrix = data["camera_matrix"]
            self.perception.dist_coeffs   = data.get("dist_coeffs")

        # ── Apply to BannerMapper ─────────────────────────────
        self.mapper.load_from_profile(data)

        # ── Build summary label ───────────────────────────────
        rms  = data.get("rms_error", "–")
        h    = data.get("height_cm", "–")
        cpp  = data.get("cm_per_pixel", "–")
        has_H = "homography" in data and data["homography"] is not None
        mode  = "homography" if has_H else "anchor"
        fname = os.path.basename(path)

        summary = (f"File     : {fname}\n"
                   f"RMS      : {rms:.4f}\n"
                   f"Height   : {h} cm\n"
                   f"Mapping  : {mode}\n"
                   f"cm/pixel : {cpp if isinstance(cpp,str) else f'{cpp:.5f}'}\n"
                   f"Mapper   : {'✓ ready' if self.mapper.is_ready() else '⚠ incomplete'}")
        col = GREEN if self.mapper.is_ready() else ORANGE
        self._calib_lbl.config(text=summary, foreground=col)
        self._set_status(f"Calibration loaded: {fname}  [{mode} mode]")
        self._update_grid_canvas()

    # ─────────────────────────────────────────────────────────
    #  CAMERA  ACTIONS
    # ─────────────────────────────────────────────────────────

    def _connect_camera(self):
        src_str = self._v_cam_src.get().strip()
        src: Any = int(src_str) if src_str.isdigit() else src_str
        self._set_status(f"Connecting camera: {src}…")
        self.root.update()
        ok = self.perception.connect_camera(src)
        if ok:
            self._cam_status.config(text=f"✓ Connected: {src}", foreground=GREEN)
            self._set_status(f"Camera connected: {src}")
        else:
            self._cam_status.config(text=f"✗ Failed: {src}", foreground=RED)
            messagebox.showerror("Camera Error",
                f"Cannot open camera: {src}\n"
                "Check index or stream URL.")

    def _manual_snapshot(self):
        frame = self.perception.snapshot()
        if frame is None:
            messagebox.showwarning("No frame",
                "Connect a camera first"); return
        robot = self.perception.detect_robot(frame, self.mapper)
        obs   = self.perception.detect_obstacles(
            frame, self.mapper,
            robot_px=(robot.px, robot.py) if robot.found else None)
        self._update_from_detection(frame, robot, obs)
        self._set_status(
            f"Snapshot — Robot: {'found '+str(robot.cell) if robot.found else 'NOT FOUND'}  "
            f"Obstacles: {len(obs)}")

    def _browse_yolo(self):
        path = filedialog.askopenfilename(
            title="YOLO model",
            filetypes=[("PyTorch model","*.pt"),("All","*.*")])
        if path:
            self._v_yolo_path.set(path)

    def _load_yolo(self):
        path = self._v_yolo_path.get().strip()
        if not path:
            messagebox.showwarning("No path", "Enter YOLO model path"); return
        self._set_status("Loading YOLO model…"); self.root.update()
        ok = self.perception.load_yolo(path)
        if ok:
            self._yolo_status.config(text=f"✓ YOLO loaded: {os.path.basename(path)}",
                                     foreground=GREEN)
        else:
            self._yolo_status.config(text="✗ YOLO load failed", foreground=RED)

    # ─────────────────────────────────────────────────────────
    #  ROBOT  ACTIONS
    # ─────────────────────────────────────────────────────────

    def _connect_robot(self):
        ip   = self._v_robot_ip.get().strip()
        port = self._v_robot_port.get()
        self.arduino.ip   = ip
        self.arduino.port = port
        self._set_status(f"Connecting to robot {ip}:{port}…")
        self.root.update()
        ok, err = self.arduino.connect()
        if ok:
            self._robot_status.config(
                text=f"✓ Connected: {ip}:{port}", foreground=GREEN)
            self._set_status(f"Robot connected: {ip}:{port}")
        else:
            self._robot_status.config(
                text=f"✗ Failed: {err}", foreground=RED)
            messagebox.showerror("Robot Connection Failed",
                f"Cannot connect to {ip}:{port}\n\n{err}\n\n"
                "Check: robot powered on, connected to OMNI_ROBOT WiFi, "
                "correct IP.")

    def _disconnect_robot(self):
        self.arduino.disconnect()
        self._robot_status.config(text="Disconnected", foreground=FG2)

    def _emergency_stop(self):
        if self.mission:
            self.mission.abort()
        self.arduino.send_stop()
        self._set_status("⛔ EMERGENCY STOP")
        self._lbl_state.config(text="State   : ABORTED ⛔", foreground=RED)

    # ─────────────────────────────────────────────────────────
    #  ANCHOR  OVERRIDE
    # ─────────────────────────────────────────────────────────

    def _on_preview_click(self, event):
        """Record click pixel on camera preview for anchor override."""
        cw = self._cam_preview.winfo_width()
        ch = self._cam_preview.winfo_height()
        # We don't know original frame size here — store raw canvas coords
        # They'll be rescaled when anchor is set if mapper has resolution info
        self._last_click_px = (event.x, event.y)

    def _set_anchor_from_click(self):
        if self._last_click_px is None:
            messagebox.showwarning("No click",
                "Take a snapshot and click on a known cell first")
            return
        col = int(self._v_anch_col.get())
        row = int(self._v_anch_row.get())
        # Scale canvas click back to frame coords if possible
        cw  = self._cam_preview.winfo_width()
        ch  = self._cam_preview.winfo_height()
        cx_canvas, cy_canvas = self._last_click_px
        # Assume frame was displayed at full canvas size
        frame_w = self.calibration.get("resolution", (cw, ch))[0] if self.calibration else cw
        frame_h = self.calibration.get("resolution", (cw, ch))[1] if self.calibration else ch
        px = int(cx_canvas * frame_w / max(cw, 1))
        py = int(cy_canvas * frame_h / max(ch, 1))
        self.mapper.anchor_px   = (px, py)
        self.mapper.anchor_cell = (col, row)
        self._anch_lbl.config(
            text=f"Anchor: pixel ({px},{py}) → cell ({col},{row})")
        self._set_status(
            f"Anchor set: ({px},{py}) → banner cell ({col},{row})")

    # ─────────────────────────────────────────────────────────
    #  MISSION  CONTROL
    # ─────────────────────────────────────────────────────────

    def _start_mission(self):
        if not self.mapper.is_ready():
            messagebox.showerror("Not Ready",
                "Banner mapper not calibrated.\n"
                "Load a calibration file with cm_per_pixel or homography.")
            return
        if self.perception.cap is None:
            messagebox.showerror("No Camera",
                "Connect a camera first.")
            return
        if self._mission_state in (MissionState.DETECTING,
                                   MissionState.PLANNING,
                                   MissionState.COMMANDING,
                                   MissionState.WAITING):
            messagebox.showwarning("Busy", "A mission is already running.")
            return

        col = int(self._v_target_col.get())
        row = int(self._v_target_row.get())
        col = max(0, min(BANNER_COLS-1, col))
        row = max(0, min(BANNER_ROWS-1, row))
        self._target_cell = (col, row)

        cfg = MissionConfig(
            target_cell    = (col, row),
            max_retries    = int(self._v_max_retry.get()),
            speed_cms      = max(1.0, float(self._v_speed_cms.get())),
            arrival_thresh = int(self._v_arrive_thr.get()),
            inflate_obs    = OBSTACLE_INFLATE,
        )

        # Estimate travel time for display
        total_cm_est = math.hypot(
            col - (self._current_cell[0] if self._current_cell else 10),
            row - (self._current_cell[1] if self._current_cell else 10)
        ) * BANNER_CELL_CM
        est_s = total_cm_est / max(cfg.speed_cms, 1.0)
        self._lbl_travel.config(
            text=f"Travel  : ≈{total_cm_est:.0f} cm  ≈{est_s:.1f}s (rough)")
        self._lbl_target.config(
            text=f"Target  : ({col}, {row})")

        if not self.arduino.connected:
            ans = messagebox.askyesno(
                "Robot not connected",
                "Robot is not connected.\n\n"
                "Run in SIMULATION mode?\n"
                "(Commands will be logged but not sent)")
            if not ans:
                return

        self.mission = MissionRunner(
            perception   = self.perception,
            mapper       = self.mapper,
            arduino      = self.arduino,
            config       = cfg,
            on_state_change = self._on_mission_state,
            on_frame        = self._on_mission_frame,
            on_grid_update  = self._on_mission_grid,
        )
        self.mission.start()
        self._set_status(f"Mission started → target ({col},{row})")

    def _abort_mission(self):
        if self.mission:
            self.mission.abort()
            self._set_status("Mission aborted")

    # ─────────────────────────────────────────────────────────
    #  MISSION  CALLBACKS  (called from background thread)
    # ─────────────────────────────────────────────────────────

    def _on_mission_state(self, state: MissionState, msg: str):
        """Thread-safe state update via root.after."""
        self.root.after(0, self._apply_state_update, state, msg)

    def _apply_state_update(self, state: MissionState, msg: str):
        self._mission_state = state
        state_colors = {
            MissionState.IDLE:       FG2,
            MissionState.SNAPSHOT:   ACCENT,
            MissionState.DETECTING:  ACCENT,
            MissionState.PLANNING:   ACCENT,
            MissionState.COMMANDING: ORANGE,
            MissionState.WAITING:    ORANGE,
            MissionState.VERIFYING:  ACCENT,
            MissionState.COMPLETE:   GREEN,
            MissionState.FAILED:     RED,
            MissionState.ABORTED:    RED,
        }
        col = state_colors.get(state, FG)
        self._lbl_state.config(text=f"State   : {state.name}", foreground=col)
        self._lbl_msg.config(text=msg, foreground=col)
        self._header_status.config(text=f"{state.name}: {msg[:60]}")

        if state == MissionState.COMPLETE:
            messagebox.showinfo("Mission Complete", msg, parent=self.root)
        elif state == MissionState.FAILED:
            messagebox.showerror("Mission Failed", msg, parent=self.root)
        elif state == MissionState.ABORTED:
            self._header_status.config(text="ABORTED", fg=RED)

    def _on_mission_frame(
        self,
        frame: np.ndarray,
        robot: Optional[RobotDetection],
        obstacles: list[ObstacleCell],
    ):
        """Thread-safe frame display update."""
        self.root.after(0, self._apply_frame_update, frame, robot, obstacles)

    def _apply_frame_update(
        self,
        frame: np.ndarray,
        robot: Optional[RobotDetection],
        obstacles: list[ObstacleCell],
    ):
        self._update_from_detection(frame, robot, obstacles)

    def _on_mission_grid(
        self,
        obstacles:   set[tuple[int,int]],
        path:        list[tuple[int,int]],
        robot_cell:  Optional[tuple[int,int]],
        target_cell: Optional[tuple[int,int]],
        theta:       float,
    ):
        self.root.after(0, self._apply_grid_update,
                        obstacles, path, robot_cell, target_cell, theta)

    def _apply_grid_update(self, obstacles, path, robot_cell, target_cell, theta):
        self._obstacles    = obstacles
        self._path         = path
        self._robot_theta  = theta
        if robot_cell:
            self._current_cell = robot_cell
            self._lbl_current.config(
                text=f"Robot   : {robot_cell}  θ={int(theta)}°")
        if path:
            length_cm = len(path) * BANNER_CELL_CM
            self._lbl_path.config(
                text=f"Path    : {len(path)} cells  ≈{length_cm:.0f}cm")
        self._update_grid_canvas()

    # ─────────────────────────────────────────────────────────
    #  GRID / CANVAS  HELPERS
    # ─────────────────────────────────────────────────────────

    def _on_grid_click(self, col: int, row: int):
        """Handle click on grid canvas → set target."""
        self._v_target_col.set(col)
        self._v_target_row.set(row)
        self._target_cell = (col, row)
        self._lbl_target.config(text=f"Target  : ({col}, {row})")
        self._update_grid_canvas()

    def _update_grid_canvas(self):
        self._grid_canvas.set_state(
            obstacles   = self._obstacles,
            path        = self._path,
            robot_cell  = self._current_cell,
            target_cell = self._target_cell,
            robot_theta = self._robot_theta,
        )

    def _update_from_detection(
        self,
        frame: np.ndarray,
        robot: Optional[RobotDetection],
        obstacles: list[ObstacleCell],
    ):
        """Update preview and status labels from a detection result."""
        self._cam_preview.show_frame(frame, robot,
                                     obstacles if obstacles else [],
                                     self.mapper)
        if robot and robot.found:
            self._current_cell = robot.cell
            self._robot_theta  = robot.theta_deg
            self._lbl_current.config(
                text=f"Robot   : {robot.cell}  θ={int(robot.theta_deg)}°",
                foreground=GREEN)
        elif robot is not None:
            self._lbl_current.config(
                text="Robot   : NOT DETECTED", foreground=RED)

        self._obstacles = {(o.col, o.row) for o in obstacles}
        self._update_grid_canvas()

    # ─────────────────────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._header_status.config(text=msg)

    def _on_close(self):
        if self.mission:
            self.mission.abort()
        if self.arduino.connected:
            self.arduino.send_stop()
        if self.perception.cap:
            self.perception.cap.release()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════
#  ENTRY  POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = NavigationApp(root)
    root.mainloop()