"""
autobot_openloop_test.py  –  Open-Loop Photo Navigation Test
═══════════════════════════════════════════════════════════════════════════════
Separate, self-contained script for testing Autobot navigation WITHOUT
continuous camera feedback.

Workflow
─────────
  1.  Capture OR load a single photo of the arena.
  2.  Detect robot position (AprilTag ID 0) and YOLO obstacles.
  3.  User clicks target on the preview image.
  4.  A* path planning with optional RDP simplification.
  5.  Calculate expected travel time from calibration constant.
  6.  POST the path to the Arduino /path endpoint.
  7.  Wait for the estimated motion duration (+ settle margin).
  8.  Capture a fresh verification photo.
  9.  Check if robot reached the target (within tolerance).
  10. If not reached → re-plan from the new photo and retry.

Calibration (measured empirically)
────────────────────────────────────
  PWM = 170  →  1 metre in 2000 ms
  Therefore:   SPEED_M_PER_S  = 1.0 / 2.0  = 0.50 m/s
               SPEED_CM_PER_S = 50.0 cm/s

  Travel time formula
  ───────────────────
  distance_cm  = pixel_distance × cm_per_pixel
  duration_ms  = (distance_cm / SPEED_CM_PER_S) × 1000
              = (distance_cm / 50.0) × 1000

  This is computed per path segment and summed for total wait.

AprilTag IDs used
──────────────────
  0 = AUTOBOT (the main robot being controlled)

YOLO classes (same trained model as navigation_v3)
───────────────────────────────────────────────────
  0=box  1=sticky_note  2=bottle  3=plastic  4=table  5=tape

Arduino endpoints used
───────────────────────
  POST /path    – send waypoint list + cm_per_pixel scale
  GET  /stop    – emergency abort
  GET  /cmd?v=S – stop motors

NOTE: This script does NOT touch or break Autobot_navigation_v3.py.
      It is standalone and imports nothing from it.
═══════════════════════════════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────────────
#  STANDARD IMPORTS
# ─────────────────────────────────────────────────────────────
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import heapq
import threading
import time
import math
import requests
import os
from collections import defaultdict, deque
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
#  ╔══════════════════════════════╗
#  ║  CONFIGURATION  –  EDIT HERE ║
#  ╚══════════════════════════════╝
# ═══════════════════════════════════════════════════════════════

# ── Robot network ────────────────────────────────────────────
ROBOT_IP         = "192.168.4.1"
ROBOT_PORT       = 80

# ── AprilTag ─────────────────────────────────────────────────
TAG_SIZE_CM      = 12.0          # physical side of the AprilTag in cm
AUTOBOT_TAG_ID   = 0             # tag ID stuck on the robot

# ── Robot physical dimensions ────────────────────────────────
ROBOT_DIAMETER_CM  = 40.0
WHEEL_MARGIN_CM    = 5.0         # extra buffer beyond body radius
ROBOT_KEEPOUT_CM   = (ROBOT_DIAMETER_CM / 2.0) + WHEEL_MARGIN_CM  # = 25 cm

# ── Arrival check tolerance ──────────────────────────────────
# The robot is considered "arrived" if its centre is within this
# many pixels of the target after the motion completes.
ARRIVAL_THRESHOLD_PX = 40        # tune based on your arena / camera height

# ── Calibration constants ────────────────────────────────────
# Measured: PWM=170 → robot travels 1 m in 2000 ms
CALIBRATION_PWM         = 170    # PWM value this calibration is valid for
CALIBRATION_DISTANCE_M  = 1.0    # metres travelled
CALIBRATION_TIME_MS     = 2000   # time taken
#   Derived speed – do NOT change unless you re-measure:
SPEED_M_PER_S   = CALIBRATION_DISTANCE_M / (CALIBRATION_TIME_MS / 1000.0)  # 0.50
SPEED_CM_PER_S  = SPEED_M_PER_S * 100.0                                     # 50.0

# Extra milliseconds added after all segments finish, to let the
# robot decelerate and settle before the verification photo.
SETTLE_MARGIN_MS = 400

# Maximum retry attempts before giving up
MAX_RETRIES = 5

# ── YOLO ─────────────────────────────────────────────────────
YOLO_MODEL_PATH = r"C:\Users\meljo\krab-3\runs\obstacle_v1\weights\best.pt"
YOLO_CONF       = 0.45
YOLO_IOU        = 0.45

YOLO_CLASSES = {
    0: "box", 1: "sticky_note", 2: "bottle",
    3: "plastic", 4: "table", 5: "tape",
}
CLASS_COLORS = {
    0: (0,  100, 255),   # box         – orange-red
    1: (0,  220, 220),   # sticky_note – yellow
    2: (255,  60,  60),  # bottle      – blue
    3: (0,  200,  60),   # plastic     – green
    4: (130, 130, 130),  # table       – grey
    5: (200,   0, 200),  # tape        – magenta
}
DEFAULT_IGNORE_CLASSES = {4}     # table is the arena surface

# ── Grid / heat map ──────────────────────────────────────────
EDGE_MARGIN_PX   = 30
BOX_SHRINK_PX    = 4
OBSTACLE_INFLATE = 1
HEAT_THRESHOLD   = 2             # frames before cell is hard-blocked
                                 # (with single-photo mode, always stamp directly)
RDP_EPSILON      = 8.0           # Ramer-Douglas-Peucker tolerance (px); 0=off

# ── Display ──────────────────────────────────────────────────
DISPLAY_W = 1024
DISPLAY_H = 720


# ═══════════════════════════════════════════════════════════════
#  A*  PLANNER  (diagonal √2 cost – identical to navigation_v3)
# ═══════════════════════════════════════════════════════════════

class AStarPlanner:
    _DIRS = [
        ( 0, 1, 1.000), ( 1, 0, 1.000), ( 0,-1, 1.000), (-1, 0, 1.000),
        ( 1, 1, 1.414), (-1,-1, 1.414), ( 1,-1, 1.414), (-1, 1, 1.414),
    ]

    def __init__(self, grid: np.ndarray):
        self.grid    = grid
        self.h, self.w = grid.shape

    @staticmethod
    def _heuristic(a, b):
        dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
        return abs(dr-dc) + 1.414 * min(dr, dc)

    def _neighbors(self, pos):
        out = []
        for dr, dc, cost in self._DIRS:
            nr, nc = pos[0]+dr, pos[1]+dc
            if 0 <= nr < self.h and 0 <= nc < self.w and self.grid[nr,nc] == 0:
                out.append(((nr, nc), cost))
        return out

    def plan(self, start, goal):
        """Returns list of (row, col) grid nodes, or None if no path."""
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
            for nb, cost in self._neighbors(cur):
                tg = g[cur] + cost
                if nb not in g or tg < g[nb]:
                    came[nb] = cur; g[nb] = tg
                    f = tg + self._heuristic(nb, goal)
                    heapq.heappush(heap, (f, nb))
        return None


# ═══════════════════════════════════════════════════════════════
#  PATH UTILITIES
# ═══════════════════════════════════════════════════════════════

def rdp_simplify(points: list, epsilon: float) -> list:
    """Ramer-Douglas-Peucker path simplification."""
    if len(points) < 3 or epsilon <= 0:
        return points
    start, end = np.array(points[0]), np.array(points[-1])
    line = end - start
    line_len = np.linalg.norm(line)
    if line_len == 0:
        return [points[0], points[-1]]
    unit = line / line_len
    vecs  = np.array(points) - start
    proj  = np.dot(vecs, unit)
    perp  = vecs - np.outer(proj, unit)
    dists = np.linalg.norm(perp, axis=1)
    idx   = int(np.argmax(dists))
    if dists[idx] > epsilon:
        left  = rdp_simplify(points[:idx+1], epsilon)
        right = rdp_simplify(points[idx:],   epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def total_path_length_px(path: list) -> float:
    """Sum of Euclidean distances between consecutive waypoints (pixels)."""
    return sum(
        math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        for i in range(len(path)-1)
    )


# ═══════════════════════════════════════════════════════════════
#  TRAVEL TIME  CALCULATOR
#
#  Formula (from calibration above):
#    speed_cm_per_s  = 50.0  (at PWM=170)
#    distance_cm     = pixel_distance × cm_per_pixel
#    duration_ms     = (distance_cm / speed_cm_per_s) × 1000
#
#  For a multi-segment path the segments are summed, then
#  SETTLE_MARGIN_MS is added for deceleration / mechanical settling.
# ═══════════════════════════════════════════════════════════════

def estimate_travel_time_ms(path: list, cm_per_pixel: float) -> float:
    """
    Returns estimated total motion time in milliseconds.

    path          – list of (x_px, y_px) waypoints
    cm_per_pixel  – calibrated scale from AprilTag detection
    """
    total_ms = 0.0
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        dist_px = math.hypot(dx, dy)
        dist_cm = dist_px * cm_per_pixel
        seg_ms  = (dist_cm / SPEED_CM_PER_S) * 1000.0
        total_ms += seg_ms
    return total_ms + SETTLE_MARGIN_MS


# ═══════════════════════════════════════════════════════════════
#  ARDUINO  HTTP  CLIENT  (minimal – only what open-loop needs)
# ═══════════════════════════════════════════════════════════════

class RobotClient:
    def __init__(self, ip: str = ROBOT_IP, port: int = ROBOT_PORT):
        self.ip   = ip
        self.port = port

    def _get(self, path: str, timeout: float = 3.0):
        url = f"http://{self.ip}:{self.port}{path}"
        return requests.get(url, timeout=timeout)

    def ping(self) -> tuple[bool, str]:
        """Returns (ok, message)."""
        try:
            r = self._get("/cmd?v=S", timeout=2)
            return (r.status_code == 200), f"HTTP {r.status_code}"
        except Exception as e:
            return False, str(e)

    def stop(self):
        """Immediately stop all motors."""
        try:
            self._get("/cmd?v=S", timeout=1)
        except:
            pass

    def abort_path(self):
        """Abort any in-progress path execution."""
        try:
            self._get("/stop", timeout=1)
        except:
            pass

    def send_path(self, waypoints: list, cm_per_pixel: float) -> bool:
        """
        POST a path to the Arduino.

        Body format (text/plain):
          <cm_per_pixel>;<x0>,<y0>;<x1>,<y1>;…

        Example:
          0.0423;640,480;600,420;540,340

        Returns True if accepted, False on error.
        """
        tokens = [f"{cm_per_pixel:.5f}"]
        tokens += [f"{int(x)},{int(y)}" for (x, y) in waypoints]
        body = ";".join(tokens)

        try:
            url = f"http://{self.ip}:{self.port}/path"
            r = requests.post(
                url,
                data=body,
                headers={"Content-Type": "text/plain"},
                timeout=5,
            )
            return r.status_code == 200
        except Exception as e:
            print(f"[RobotClient.send_path] ERROR: {e}")
            return False


# ═══════════════════════════════════════════════════════════════
#  DETECTOR  (AprilTag + YOLO in one class)
# ═══════════════════════════════════════════════════════════════

class Detector:
    def __init__(self):
        # AprilTag
        self.at_ok = False
        try:
            import pupil_apriltags
            self.at_detector = pupil_apriltags.Detector(
                families="tag36h11", nthreads=4,
                quad_decimate=1.0, quad_sigma=0.0,
                refine_edges=1, decode_sharpening=0.25,
            )
            self.at_ok = True
        except ImportError:
            self.at_detector = None

        # YOLO
        self.yolo_ok    = False
        self.yolo_model = None
        try:
            from ultralytics import YOLO
            try:
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                self.yolo_ok    = True
                print(f"✓ YOLO loaded: {YOLO_MODEL_PATH}")
            except Exception as e:
                print(f"Custom model failed ({e}), trying yolo11n.pt …")
                try:
                    self.yolo_model = YOLO("yolo11n.pt")
                    self.yolo_ok    = True
                    print("✓ YOLO loaded: yolo11n.pt (fallback)")
                except Exception as e2:
                    print(f"✗ YOLO load failed: {e2}")
        except ImportError:
            print("ultralytics not installed – YOLO disabled")

    # ── AprilTag ─────────────────────────────────────────────

    def detect_apriltag(self, frame: np.ndarray) -> dict:
        """
        Returns dict  {tag_id: (cx, cy, theta_deg, corners, cm_per_pixel)}
        Only fills cm_per_pixel for the AUTOBOT tag (ID 0).
        """
        result = {}
        if not self.at_ok or self.at_detector is None:
            return result
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = self.at_detector.detect(gray)
        for d in dets:
            tid      = int(d.tag_id)
            cx, cy   = float(d.center[0]), float(d.center[1])
            corners  = d.corners.astype(np.float32).copy()
            c0, c1   = corners[0], corners[1]
            theta    = float(np.degrees(math.atan2(
                c1[1]-c0[1], c1[0]-c0[0]))) % 360.0
            edge_px  = float(np.linalg.norm(c1 - c0))
            cmp      = (TAG_SIZE_CM / edge_px) if edge_px > 5 else None
            result[tid] = (cx, cy, theta, corners, cmp)
        return result

    # ── YOLO ─────────────────────────────────────────────────

    def detect_yolo(self, frame: np.ndarray,
                    ignore_classes: set = DEFAULT_IGNORE_CLASSES,
                    conf: float = YOLO_CONF) -> list:
        """
        Returns list of  (x1, y1, x2, y2, class_id, confidence)
        Boxes inside edge margin or below conf floor are dropped.
        """
        if not self.yolo_ok or self.yolo_model is None:
            return []
        results  = self.yolo_model(frame, conf=conf, iou=YOLO_IOU, verbose=False)
        boxes    = []
        h, w     = frame.shape[:2]
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cid  = int(box.cls[0].item())
                cf   = float(box.conf[0].item())
                if cid in ignore_classes:
                    continue
                # Edge margin filter
                if (x1 < EDGE_MARGIN_PX or y1 < EDGE_MARGIN_PX
                        or x2 > w - EDGE_MARGIN_PX
                        or y2 > h - EDGE_MARGIN_PX):
                    continue
                # Box shrink
                x1s = x1 + BOX_SHRINK_PX; y1s = y1 + BOX_SHRINK_PX
                x2s = x2 - BOX_SHRINK_PX; y2s = y2 - BOX_SHRINK_PX
                if x2s <= x1s or y2s <= y1s:
                    continue
                boxes.append((x1s, y1s, x2s, y2s, cid, cf))
        return boxes


# ═══════════════════════════════════════════════════════════════
#  GRID  BUILDER
# ═══════════════════════════════════════════════════════════════

def build_grid(obstacles: list,
               frame_w: int, frame_h: int,
               grid_size_px: int,
               robot_pos,
               robot_keepout_px: int,
               inflate: int = OBSTACLE_INFLATE) -> np.ndarray:
    """
    Builds a binary occupancy grid (0=free, 1=blocked).

    Obstacles are stamped directly (no heat-map delay needed for
    a single-photo snapshot).  The robot's own footprint is cleared
    using a circular keep-out zone so A* is never confused by it.
    """
    rows = frame_h // grid_size_px
    cols = frame_w  // grid_size_px
    grid = np.zeros((rows, cols), dtype=np.uint8)

    # Stamp each obstacle into the grid
    for x1, y1, x2, y2, *_ in obstacles:
        # Skip implausibly large detections (whole-table shadows etc.)
        if (y2 - y1) > frame_h * 0.60 or (x2 - x1) > frame_w * 0.60:
            continue
        r1 = max(0, min(rows-1,  y1 // grid_size_px))
        r2 = max(0, min(rows-1,  y2 // grid_size_px))
        c1 = max(0, min(cols-1,  x1 // grid_size_px))
        c2 = max(0, min(cols-1,  x2 // grid_size_px))
        grid[r1:r2+1, c1:c2+1] = 1

    # Inflate obstacles for safety margin
    if inflate > 0:
        try:
            from scipy.ndimage import binary_dilation
            struct = np.ones((2*inflate+1, 2*inflate+1), dtype=bool)
            grid   = binary_dilation(grid, structure=struct).astype(np.uint8)
        except ImportError:
            # Manual dilation fallback
            dilated = grid.copy()
            for dr in range(-inflate, inflate+1):
                for dc in range(-inflate, inflate+1):
                    dilated |= np.roll(np.roll(grid, dr, axis=0), dc, axis=1)
            grid = dilated

    # Clear robot's own circular footprint from the grid
    if robot_pos is not None:
        rcx, rcy = robot_pos
        r_sq = robot_keepout_px ** 2
        for row in range(rows):
            for col in range(cols):
                cx = col * grid_size_px + grid_size_px // 2
                cy = row * grid_size_px + grid_size_px // 2
                if (cx-rcx)**2 + (cy-rcy)**2 <= r_sq:
                    grid[row, col] = 0

    return grid


# ═══════════════════════════════════════════════════════════════
#  COORDINATE  HELPERS  (matching navigation_v3 convention)
# ═══════════════════════════════════════════════════════════════

def px_to_grid(px: int, py: int, grid_size: int,
               rows: int, cols: int) -> tuple:
    return (max(0, min(rows-1, py // grid_size)),
            max(0, min(cols-1, px // grid_size)))

def grid_to_px(r: int, c: int, grid_size: int) -> tuple:
    return (c * grid_size + grid_size // 2,
            r * grid_size + grid_size // 2)


# ═══════════════════════════════════════════════════════════════
#  ANNOTATE  FRAME  (draws boxes, tags, path, keep-out circle)
# ═══════════════════════════════════════════════════════════════

def annotate_frame(frame: np.ndarray,
                   all_tags:    dict,
                   obstacles:   list,
                   robot_pos,
                   target_pos,
                   path:        list,
                   keepout_px:  int,
                   cm_per_pixel,
                   grid=None,
                   grid_size_px: int = 20,
                   show_grid:   bool = False) -> np.ndarray:
    img = frame.copy()

    # Optional grid overlay
    if show_grid and grid is not None:
        gs = grid_size_px
        h, w = img.shape[:2]
        rows, cols = grid.shape
        ov = img.copy()
        for x in range(0, w, gs): cv2.line(ov, (x,0), (x,h), (50,50,50), 1)
        for y in range(0, h, gs): cv2.line(ov, (0,y), (w,y), (50,50,50), 1)
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1:
                    x1, y1 = c*gs, r*gs
                    sub = ov[y1:y1+gs, x1:x1+gs]
                    sub[:] = (sub * 0.4 + np.array([0, 0, 180]) * 0.6).astype(np.uint8)
        img = cv2.addWeighted(ov, 0.4, img, 0.6, 0)

    # YOLO obstacle boxes
    for x1, y1, x2, y2, cid, conf in obstacles:
        col   = CLASS_COLORS.get(cid, (0, 0, 220))
        name  = YOLO_CLASSES.get(cid, f"cls{cid}")
        label = f"{name} {conf:.2f}"
        cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lx, ly = x1, max(0, y1-4)
        cv2.rectangle(img, (lx, ly-th-4), (lx+tw+4, ly), col, -1)
        cv2.putText(img, label, (lx+2, ly-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # AprilTag overlays
    for tid, (cx, cy, theta, corners, _) in all_tags.items():
        col  = (255, 0, 255) if tid == 0 else (0, 255, 0)
        name = f"ID{tid}" + (" AUTOBOT" if tid == 0 else "")
        icx, icy = int(cx), int(cy)
        cv2.polylines(img, [corners.astype(int)], True, col, 2)
        cv2.circle(img, (icx, icy), 20, col, 2)
        r  = math.radians(theta)
        ex = int(icx + 35*math.cos(r))
        ey = int(icy + 35*math.sin(r))
        cv2.arrowedLine(img, (icx, icy), (ex, ey), col, 2, tipLength=0.4)
        cv2.putText(img, name, (icx+26, icy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        cv2.putText(img, f"({icx},{icy}) θ={int(theta)}°",
                    (icx+26, icy+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    # Keep-out circle
    if robot_pos is not None and keepout_px > 0:
        kcx, kcy = robot_pos
        ov = img.copy()
        cv2.circle(ov, (kcx, kcy), keepout_px, (255, 0, 255), -1)
        img = cv2.addWeighted(ov, 0.12, img, 0.88, 0)
        cv2.circle(img, (kcx, kcy), keepout_px, (255, 0, 255), 2)

    # Target marker
    if target_pos:
        tx, ty = target_pos
        cv2.drawMarker(img, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 32, 3)
        cv2.circle(img, (tx, ty), 22, (0, 255, 255), 3)
        cv2.putText(img, "TARGET", (tx+26, ty-26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Path overlay
    if path and len(path) > 1:
        for i in range(len(path)-1):
            cv2.line(img, path[i], path[i+1], (0, 220, 255), 2)
        for i, pt in enumerate(path):
            if   i == 0:               cv2.circle(img, pt, 8, (0,255,0),  -1)
            elif i == len(path)-1:     cv2.circle(img, pt, 8, (0,0,255),  -1)
            else:                      cv2.circle(img, pt, 5, (255,220,0), -1)

    # Scale info
    if cm_per_pixel:
        cv2.putText(img,
            f"Scale: {cm_per_pixel:.4f} cm/px | "
            f"Speed: {SPEED_CM_PER_S:.1f} cm/s (PWM={CALIBRATION_PWM})",
            (10, img.shape[0]-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,0), 1)

    return img


# ═══════════════════════════════════════════════════════════════
#  MAIN  GUI
# ═══════════════════════════════════════════════════════════════

class OpenLoopTestGUI:
    """
    Self-contained GUI for the open-loop photo-based navigation test.

    Panels
    ───────
    Left  – controls, settings, status log
    Right – preview canvas (click to set target)
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(
            "Autobot  –  Open-Loop Photo Navigation Test  "
            f"[PWM={CALIBRATION_PWM} → {SPEED_CM_PER_S:.0f} cm/s]")
        self.root.geometry("1520x900")
        self.root.state("zoomed")

        # ── Detection / planning state ────────────────────────
        self.detector      = Detector()
        self.robot_client  = RobotClient(ROBOT_IP, ROBOT_PORT)

        self.current_frame  : np.ndarray | None = None  # raw capture
        self.display_frame  : np.ndarray | None = None  # resized for canvas
        self.all_tags       : dict  = {}
        self.obstacles      : list  = []
        self.robot_pos      : tuple | None = None
        self.robot_theta    : float = 0.0
        self.cm_per_pixel   : float | None = None
        self.grid_size_px   : int   = 20
        self.keepout_px     : int   = 50     # updated after calibration
        self.target_pos     : tuple | None = None
        self.planned_path   : list | None = None
        self.grid           : np.ndarray | None = None

        # ── Retry state ───────────────────────────────────────
        self._running_mission = False
        self._retry_count     = 0
        self._mission_thread  : threading.Thread | None = None

        # ── Webcam ────────────────────────────────────────────
        self.cap = None

        # ── Tkinter vars ──────────────────────────────────────
        self.robot_ip_var     = tk.StringVar(value=ROBOT_IP)
        self.robot_port_var   = tk.StringVar(value=str(ROBOT_PORT))
        self.webcam_idx_var   = tk.StringVar(value="0")
        self.conf_var         = tk.DoubleVar(value=YOLO_CONF)
        self.inflate_var      = tk.IntVar(value=OBSTACLE_INFLATE)
        self.rdp_var          = tk.BooleanVar(value=True)
        self.show_grid_var    = tk.BooleanVar(value=False)
        self.arrival_thr_var  = tk.IntVar(value=ARRIVAL_THRESHOLD_PX)
        self.max_retry_var    = tk.IntVar(value=MAX_RETRIES)
        self.settle_var       = tk.IntVar(value=SETTLE_MARGIN_MS)
        self.manual_cmpx_var  = tk.DoubleVar(value=0.04)
        self.use_manual_var   = tk.BooleanVar(value=False)
        self.wheel_margin_var = tk.DoubleVar(value=WHEEL_MARGIN_CM)
        self.ignore_cls_vars  = {
            cid: tk.BooleanVar(value=(cid in DEFAULT_IGNORE_CLASSES))
            for cid in YOLO_CLASSES
        }

        # Ignore-class set (re-computed on change)
        self._ignored_classes : set = set(DEFAULT_IGNORE_CLASSES)

        self._build_ui()
        self.root.bind("<KeyPress>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────
    #  UI  BUILD
    # ─────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── Left scrollable panel ─────────────────────────────
        lc = ttk.Frame(self.root, width=360)
        lc.grid(row=0, column=0, sticky="nsw", padx=(6,4), pady=6)
        lc.grid_propagate(False)

        lcan = tk.Canvas(lc, bg="#f4f4f4", highlightthickness=0, width=340)
        lscr = ttk.Scrollbar(lc, orient="vertical", command=lcan.yview)
        lf   = ttk.Frame(lcan)
        lf.bind("<Configure>",
                lambda e: lcan.configure(scrollregion=lcan.bbox("all")))
        lcan.create_window((0, 0), window=lf, anchor="nw")
        lcan.configure(yscrollcommand=lscr.set)
        lcan.bind_all("<MouseWheel>",
                      lambda e: lcan.yview_scroll(int(-1*(e.delta/120)), "units"))
        lcan.pack(side="left", fill="both", expand=True)
        lscr.pack(side="right", fill="y")
        left = lf

        # ── Right canvas panel ────────────────────────────────
        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, sticky="nsew", padx=(0,6), pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        cf = ttk.LabelFrame(right, text="Arena Preview  (click to set target)",
                            padding="4")
        cf.grid(row=0, column=0, sticky="nsew")
        cf.columnconfigure(0, weight=1)
        cf.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(cf, bg="#1a1a2e", cursor="crosshair",
                                width=DISPLAY_W, height=DISPLAY_H)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Status bar
        self.status_bar = ttk.Label(
            self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W,
            font=("Courier", 9))
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

        # ════════════════════════════════════
        #  1.  PHOTO SOURCE
        # ════════════════════════════════════
        sf = ttk.LabelFrame(left, text="📷  Photo Source", padding="8")
        sf.pack(fill=tk.X, pady=4)

        ttk.Button(sf, text="📂  Load Photo from File",
                   command=self._load_photo).pack(fill=tk.X, pady=2)

        ttk.Separator(sf, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Label(sf, text="Webcam index:").pack(anchor=tk.W)
        ttk.Entry(sf, textvariable=self.webcam_idx_var, width=5).pack(anchor=tk.W)
        ttk.Button(sf, text="📸  Capture from Webcam",
                   command=self._capture_webcam).pack(fill=tk.X, pady=2)

        # ════════════════════════════════════
        #  2.  DETECTION
        # ════════════════════════════════════
        df = ttk.LabelFrame(left, text="🔍  Detection", padding="8")
        df.pack(fill=tk.X, pady=4)

        self.at_lbl = ttk.Label(
            df,
            text=f"AprilTag: {'✓ ready' if self.detector.at_ok else '❌ missing'}",
            font=("Arial", 9))
        self.at_lbl.pack(anchor=tk.W)
        self.yolo_lbl = ttk.Label(
            df,
            text=f"YOLO: {'✓ loaded' if self.detector.yolo_ok else '⚠ not loaded'}",
            font=("Arial", 9))
        self.yolo_lbl.pack(anchor=tk.W, pady=(0,4))

        ttk.Label(df, text="YOLO confidence:").pack(anchor=tk.W)
        ttk.Scale(df, from_=0.30, to=0.90, orient=tk.HORIZONTAL,
                  variable=self.conf_var).pack(fill=tk.X)

        ttk.Label(df, text="🚫  Ignore classes (not obstacles):").pack(
            anchor=tk.W, pady=(4,0))
        for cid, cname in YOLO_CLASSES.items():
            r, g, b = (CLASS_COLORS[cid][2], CLASS_COLORS[cid][1],
                       CLASS_COLORS[cid][0])
            hex_col = f"#{r:02x}{g:02x}{b:02x}"
            tk.Checkbutton(df, text=f"  {cname}",
                           variable=self.ignore_cls_vars[cid],
                           fg=hex_col, font=("Arial", 8, "bold"),
                           anchor=tk.W,
                           command=self._update_ignored_classes).pack(anchor=tk.W)

        ttk.Button(df, text="🔬  Run Detection on Current Photo",
                   command=self._detect_current).pack(fill=tk.X, pady=4)

        self.det_result_lbl = ttk.Label(
            df, text="Tags: –   Obstacles: –", font=("Arial", 9, "bold"))
        self.det_result_lbl.pack()

        # ════════════════════════════════════
        #  3.  CALIBRATION
        # ════════════════════════════════════
        gf = ttk.LabelFrame(left, text="📐  Calibration & Keep-Out", padding="8")
        gf.pack(fill=tk.X, pady=4)

        self.calib_lbl = ttk.Label(
            gf, text="cm/pixel: pending AprilTag",
            font=("Arial", 9, "bold"), foreground="#888888")
        self.calib_lbl.pack(anchor=tk.W)
        self.grid_lbl = ttk.Label(gf, text="Grid cell: – px",
                                  font=("Arial", 9))
        self.grid_lbl.pack(anchor=tk.W)
        self.keepout_lbl = ttk.Label(gf, text="Keep-out: – px",
                                     font=("Arial", 9), foreground="#cc4400")
        self.keepout_lbl.pack(anchor=tk.W, pady=(0,4))

        ttk.Checkbutton(gf, text="Manual calibration override",
                        variable=self.use_manual_var,
                        command=self._on_manual_toggle).pack(anchor=tk.W)
        mcf = ttk.Frame(gf); mcf.pack(fill=tk.X)
        ttk.Label(mcf, text="cm/px:").pack(side=tk.LEFT)
        self.manual_entry = ttk.Entry(
            mcf, textvariable=self.manual_cmpx_var, width=8, state=tk.DISABLED)
        self.manual_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(mcf, text="Apply",
                   command=self._apply_manual_calib).pack(side=tk.LEFT)

        ttk.Separator(gf, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Label(gf, text="Wheel margin (cm):").pack(anchor=tk.W)
        ttk.Spinbox(gf, from_=0.0, to=15.0, increment=0.5,
                    textvariable=self.wheel_margin_var,
                    width=6).pack(anchor=tk.W)

        # ════════════════════════════════════
        #  4.  PATH PLANNING
        # ════════════════════════════════════
        pf = ttk.LabelFrame(left, text="🗺️  Path Planning", padding="8")
        pf.pack(fill=tk.X, pady=4)

        ttk.Label(pf, text="Click canvas to set target →",
                  font=("Arial", 9, "italic")).pack(anchor=tk.W)
        self.robot_pos_lbl = ttk.Label(
            pf, text="🤖 Robot: not detected", font=("Arial", 9))
        self.robot_pos_lbl.pack(anchor=tk.W, pady=2)
        self.target_pos_lbl = ttk.Label(
            pf, text="🎯 Target: not set", font=("Arial", 9))
        self.target_pos_lbl.pack(anchor=tk.W, pady=2)

        ttk.Button(pf, text="Clear Target",
                   command=self._clear_target).pack(fill=tk.X, pady=2)
        ttk.Button(pf, text="🧭  Plan A* Path",
                   command=self._plan_path).pack(fill=tk.X, pady=4)

        ttk.Checkbutton(pf, text="RDP path simplification",
                        variable=self.rdp_var).pack(anchor=tk.W)
        ttk.Label(pf, text="Obstacle inflate (cells):").pack(anchor=tk.W)
        ttk.Spinbox(pf, from_=0, to=5,
                    textvariable=self.inflate_var, width=5).pack(anchor=tk.W)
        ttk.Checkbutton(pf, text="Show grid overlay",
                        variable=self.show_grid_var,
                        command=self._redraw).pack(anchor=tk.W)

        self.path_info_lbl = ttk.Label(
            pf, text="No path", font=("Arial", 9))
        self.path_info_lbl.pack(pady=4)

        # ════════════════════════════════════
        #  5.  ROBOT CONTROL
        # ════════════════════════════════════
        rf = ttk.LabelFrame(left, text="🤖  Robot Connection", padding="8")
        rf.pack(fill=tk.X, pady=4)

        rif = ttk.Frame(rf); rif.pack(fill=tk.X, pady=2)
        ttk.Label(rif, text="IP:").pack(side=tk.LEFT)
        ttk.Entry(rif, textvariable=self.robot_ip_var,
                  width=14).pack(side=tk.LEFT, padx=4)
        ttk.Label(rif, text="Port:").pack(side=tk.LEFT)
        ttk.Entry(rif, textvariable=self.robot_port_var,
                  width=5).pack(side=tk.LEFT)

        bf = ttk.Frame(rf); bf.pack(fill=tk.X, pady=2)
        ttk.Button(bf, text="Ping Robot",
                   command=self._ping_robot).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="⛔  STOP",
                   command=self._emergency_stop).pack(side=tk.LEFT, padx=2)

        self.robot_conn_lbl = ttk.Label(
            rf, text="Not tested", foreground="#888888", font=("Arial", 9))
        self.robot_conn_lbl.pack()

        # ════════════════════════════════════
        #  6.  CALIBRATION SPEED DISPLAY
        # ════════════════════════════════════
        cf2 = ttk.LabelFrame(left, text="⏱️  Speed Calibration", padding="8")
        cf2.pack(fill=tk.X, pady=4)
        ttk.Label(cf2,
            text=f"PWM {CALIBRATION_PWM}  →  {CALIBRATION_DISTANCE_M} m  "
                 f"in {CALIBRATION_TIME_MS} ms",
            font=("Courier", 8)).pack(anchor=tk.W)
        ttk.Label(cf2,
            text=f"Speed = {SPEED_M_PER_S:.2f} m/s  = {SPEED_CM_PER_S:.1f} cm/s",
            font=("Courier", 8, "bold"), foreground="#007700").pack(anchor=tk.W)
        ttk.Label(cf2,
            text="Formula:\n"
                 "  dist_cm = px_dist × cm_per_px\n"
                 "  time_ms = (dist_cm / speed_cm_s) × 1000",
            font=("Courier", 7), foreground="#555555").pack(anchor=tk.W, pady=2)

        self.est_time_lbl = ttk.Label(
            cf2, text="Est. travel: – ms", font=("Courier", 9, "bold"),
            foreground="#004488")
        self.est_time_lbl.pack(pady=2)

        ttk.Label(cf2, text="Settle margin (ms):").pack(anchor=tk.W)
        ttk.Spinbox(cf2, from_=0, to=2000, increment=50,
                    textvariable=self.settle_var, width=6).pack(anchor=tk.W)

        # ════════════════════════════════════
        #  7.  MISSION CONTROL
        # ════════════════════════════════════
        mf = ttk.LabelFrame(left, text="🚀  Mission Control", padding="8")
        mf.pack(fill=tk.X, pady=4)

        ttk.Label(mf, text="Arrival threshold (px):").pack(anchor=tk.W)
        ttk.Spinbox(mf, from_=10, to=200,
                    textvariable=self.arrival_thr_var, width=5).pack(anchor=tk.W)
        ttk.Label(mf, text="Max retries:").pack(anchor=tk.W)
        ttk.Spinbox(mf, from_=1, to=20,
                    textvariable=self.max_retry_var, width=5).pack(anchor=tk.W, pady=2)

        self.start_btn = ttk.Button(
            mf, text="▶  START MISSION",
            command=self._start_mission)
        self.start_btn.pack(fill=tk.X, pady=4)
        self.abort_btn = ttk.Button(
            mf, text="⛔  ABORT MISSION",
            command=self._abort_mission, state=tk.DISABLED)
        self.abort_btn.pack(fill=tk.X, pady=2)

        self.mission_lbl = ttk.Label(
            mf, text="No mission running",
            font=("Arial", 9, "bold"), foreground="#888888")
        self.mission_lbl.pack(pady=2)
        self.retry_lbl = ttk.Label(mf, text="", font=("Arial", 8))
        self.retry_lbl.pack()

        # ════════════════════════════════════
        #  8.  LOG
        # ════════════════════════════════════
        lf2 = ttk.LabelFrame(left, text="📋  Log", padding="6")
        lf2.pack(fill=tk.BOTH, expand=True, pady=4)
        self.log_text = tk.Text(lf2, height=14, width=38,
                                font=("Courier", 7), state=tk.DISABLED,
                                bg="#f8f8f8")
        lscr2 = ttk.Scrollbar(lf2, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=lscr2.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lscr2.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Button(left, text="Clear Log",
                   command=self._clear_log).pack(fill=tk.X, pady=2)

    # ─────────────────────────────────────────────────────────
    #  LOGGING
    # ─────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        full = f"[{ts}] {msg}"
        print(full)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, full + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.status_bar.config(text=full)

    def _clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ─────────────────────────────────────────────────────────
    #  PHOTO ACQUISITION
    # ─────────────────────────────────────────────────────────

    def _load_photo(self):
        """Load a single photo from disk."""
        path = filedialog.askopenfilename(
            title="Select Arena Photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("All", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Cannot read image:\n{path}")
            return
        self.current_frame = img
        self.display_frame = cv2.resize(img, (DISPLAY_W, DISPLAY_H))
        self._log(f"Loaded photo: {os.path.basename(path)}")
        self._redraw()

    def _capture_webcam(self) -> bool:
        """
        Capture a single frame from the webcam.
        Returns True on success.
        """
        try:
            idx = int(self.webcam_idx_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid webcam index")
            return False

        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open webcam {idx}")
            return False

        # Warm-up: discard a few frames so auto-exposure settles
        for _ in range(5):
            cap.read()
            time.sleep(0.05)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            messagebox.showerror("Error", "Failed to read from webcam")
            return False

        self.current_frame = frame
        self.display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        self._log(f"Captured frame from webcam {idx}")
        self._redraw()
        return True

    # ─────────────────────────────────────────────────────────
    #  DETECTION
    # ─────────────────────────────────────────────────────────

    def _update_ignored_classes(self):
        self._ignored_classes = {
            cid for cid, var in self.ignore_cls_vars.items() if var.get()
        }

    def _detect_current(self) -> bool:
        """
        Run AprilTag + YOLO on self.display_frame.
        Populates:  all_tags, robot_pos, robot_theta,
                    cm_per_pixel, obstacles, grid_size_px, keepout_px
        Returns True if robot was found.
        """
        if self.display_frame is None:
            messagebox.showwarning("No photo", "Load or capture a photo first")
            return False

        self._update_ignored_classes()

        # ── AprilTag ──────────────────────────────────────────
        self.all_tags = self.detector.detect_apriltag(self.display_frame)

        # Extract robot position and calibration from tag 0
        self.robot_pos   = None
        self.robot_theta = 0.0
        if AUTOBOT_TAG_ID in self.all_tags:
            cx, cy, theta, corners, cmp = self.all_tags[AUTOBOT_TAG_ID]
            self.robot_pos   = (int(cx), int(cy))
            self.robot_theta = theta
            if cmp and not self.use_manual_var.get():
                self._apply_calibration(cmp)

        # Manual calibration fallback
        if self.cm_per_pixel is None or self.use_manual_var.get():
            self._apply_manual_calib()

        # ── YOLO ──────────────────────────────────────────────
        # Mask robot before YOLO so its body doesn't show as obstacle
        masked = self._mask_robot(self.display_frame)
        self.obstacles = self.detector.detect_yolo(
            masked,
            ignore_classes=self._ignored_classes,
            conf=float(self.conf_var.get()),
        )
        # Filter out detections inside robot keep-out circle
        self.obstacles = self._filter_obstacles_by_keepout(self.obstacles)

        n_tags = len(self.all_tags)
        n_obs  = len(self.obstacles)
        self.det_result_lbl.config(
            text=f"Tags: {n_tags}   Obstacles: {n_obs}",
            foreground="#007700" if n_tags > 0 else "#cc6600")

        if self.robot_pos:
            self.robot_pos_lbl.config(
                text=f"🤖 Robot: {self.robot_pos} θ={int(self.robot_theta)}°",
                foreground="#007700")
            self._log(
                f"Detected robot at {self.robot_pos}, θ={int(self.robot_theta)}°, "
                f"{n_obs} obstacle(s)")
        else:
            self.robot_pos_lbl.config(
                text="🤖 Robot: NOT FOUND", foreground="#cc0000")
            self._log("⚠  Robot (AprilTag ID 0) not found in this photo")

        self._redraw()
        return self.robot_pos is not None

    def _apply_calibration(self, cmp: float):
        """Update all calibration-derived values from a cm/px measurement."""
        self.cm_per_pixel  = cmp
        self.grid_size_px  = max(10, int((ROBOT_DIAMETER_CM / 2.0) / cmp))
        margin_cm          = float(self.wheel_margin_var.get())
        radius_cm          = (ROBOT_DIAMETER_CM / 2.0) + margin_cm
        self.keepout_px    = max(10, int(radius_cm / cmp))
        self.calib_lbl.config(
            text=f"cm/pixel: {cmp:.4f}", foreground="#007700")
        self.grid_lbl.config(text=f"Grid cell: {self.grid_size_px} px")
        self.keepout_lbl.config(
            text=f"Keep-out: {self.keepout_px} px  "
                 f"(r={radius_cm:.0f} cm)")

    def _mask_robot(self, frame: np.ndarray) -> np.ndarray:
        """Black out the robot's circular keep-out zone before YOLO."""
        if self.robot_pos is None:
            return frame
        masked = frame.copy()
        cv2.circle(masked, self.robot_pos, self.keepout_px, (0, 0, 0), -1)
        return masked

    def _filter_obstacles_by_keepout(self, obstacles: list) -> list:
        """Remove any detection that overlaps the robot keep-out circle."""
        if self.robot_pos is None:
            return obstacles
        cx, cy = self.robot_pos
        r      = self.keepout_px
        out    = []
        for box in obstacles:
            x1, y1, x2, y2 = box[:4]
            nx = max(x1, min(cx, x2))
            ny = max(y1, min(cy, y2))
            if (nx-cx)**2 + (ny-cy)**2 > r*r:
                out.append(box)
        return out

    # ─────────────────────────────────────────────────────────
    #  CALIBRATION CONTROLS
    # ─────────────────────────────────────────────────────────

    def _on_manual_toggle(self):
        if self.use_manual_var.get():
            self.manual_entry.config(state=tk.NORMAL)
            self._apply_manual_calib()
        else:
            self.manual_entry.config(state=tk.DISABLED)

    def _apply_manual_calib(self):
        try:
            val = float(self.manual_cmpx_var.get())
            if val <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            if self.use_manual_var.get():
                messagebox.showerror("Error", "Enter a positive cm/pixel value")
            return
        self._apply_calibration(val)
        self.calib_lbl.config(
            text=f"cm/pixel: {val:.4f} (manual)", foreground="#e06000")

    # ─────────────────────────────────────────────────────────
    #  PATH PLANNING
    # ─────────────────────────────────────────────────────────

    def _plan_path(self) -> bool:
        if self.display_frame is None:
            messagebox.showwarning("No photo", "Load a photo and run detection first")
            return False
        if self.robot_pos is None:
            messagebox.showwarning("No robot",
                "Robot not detected – run detection or load a photo with AprilTag ID 0")
            return False
        if self.target_pos is None:
            messagebox.showwarning("No target",
                "Click on the arena preview to set a target point")
            return False
        if self.cm_per_pixel is None:
            messagebox.showwarning("No calibration",
                "cm/pixel unknown – enable manual calibration or show an AprilTag")
            return False

        gs   = self.grid_size_px
        rows = DISPLAY_H // gs
        cols = DISPLAY_W  // gs

        self.grid = build_grid(
            self.obstacles,
            DISPLAY_W, DISPLAY_H, gs,
            self.robot_pos, self.keepout_px,
            inflate=self.inflate_var.get(),
        )

        sg = px_to_grid(*self.robot_pos, gs, rows, cols)
        eg = px_to_grid(*self.target_pos, gs, rows, cols)

        # Force start and goal cells free
        self.grid[sg[0], sg[1]] = 0
        self.grid[eg[0], eg[1]] = 0

        if self.grid[eg[0], eg[1]] == 1:
            messagebox.showerror("Blocked", "Target cell is inside an obstacle!")
            return False

        path_g = AStarPlanner(self.grid).plan(sg, eg)
        if path_g is None:
            messagebox.showerror("No path", "A* could not find a path to the target!")
            self._log("A* failed – no path found")
            return False

        pts = [grid_to_px(r, c, gs) for r, c in path_g]
        if self.rdp_var.get():
            pts = rdp_simplify(pts, RDP_EPSILON)
        self.planned_path = pts

        # Travel time estimate
        total_ms = estimate_travel_time_ms(pts, self.cm_per_pixel)
        total_ms_with_settle = total_ms  # settle already included inside function
        self.est_time_lbl.config(
            text=f"Est. travel: {total_ms:.0f} ms "
                 f"(+{self.settle_var.get()} ms settle)",
            foreground="#004488")

        length_px = total_path_length_px(pts)
        length_cm = length_px * self.cm_per_pixel
        self.path_info_lbl.config(
            text=f"✓ {len(pts)} waypoints  |  {length_cm:.1f} cm  |  "
                 f"≈{total_ms:.0f} ms",
            foreground="#007700")
        self._log(f"Path planned: {len(pts)} waypoints, "
                  f"{length_cm:.1f} cm, est. {total_ms:.0f} ms")
        self._redraw()
        return True

    # ─────────────────────────────────────────────────────────
    #  CANVAS  INTERACTION
    # ─────────────────────────────────────────────────────────

    def _on_canvas_click(self, event):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        cx = int(event.x * DISPLAY_W / cw) if cw > 10 else event.x
        cy = int(event.y * DISPLAY_H / ch) if ch > 10 else event.y
        self.target_pos = (cx, cy)
        self.target_pos_lbl.config(
            text=f"🎯 Target: ({cx}, {cy})", foreground="#007700")
        self.planned_path = None
        self.path_info_lbl.config(text="Target set – plan a path", foreground="#555")
        self._log(f"Target set: ({cx}, {cy})")
        self._redraw()

    def _clear_target(self):
        self.target_pos   = None
        self.planned_path = None
        self.target_pos_lbl.config(text="🎯 Target: not set", foreground="#888888")
        self.path_info_lbl.config(text="No path")
        self._redraw()

    # ─────────────────────────────────────────────────────────
    #  REDRAW
    # ─────────────────────────────────────────────────────────

    def _redraw(self):
        if self.display_frame is None:
            return
        img = annotate_frame(
            self.display_frame,
            self.all_tags,
            self.obstacles,
            self.robot_pos,
            self.target_pos,
            self.planned_path,
            self.keepout_px,
            self.cm_per_pixel,
            grid=self.grid,
            grid_size_px=self.grid_size_px,
            show_grid=self.show_grid_var.get(),
        )
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw > 10 and ch > 10 and (cw != DISPLAY_W or ch != DISPLAY_H):
            img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)
        rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk

    # ─────────────────────────────────────────────────────────
    #  ROBOT CONNECTION
    # ─────────────────────────────────────────────────────────

    def _ping_robot(self):
        self.robot_client.ip   = self.robot_ip_var.get().strip()
        self.robot_client.port = int(self.robot_port_var.get().strip())
        ok, msg = self.robot_client.ping()
        color   = "#007700" if ok else "#cc0000"
        text    = f"{'✓ Reachable' if ok else '✗ Unreachable'}  ({msg})"
        self.robot_conn_lbl.config(text=text, foreground=color)
        self._log(f"Ping {self.robot_client.ip}: {text}")

    def _emergency_stop(self):
        self._abort_mission()
        self.robot_client.abort_path()
        self.robot_client.stop()
        self._log("⛔  Emergency stop sent")
        messagebox.showinfo("Emergency Stop", "Stop command sent to robot")

    # ─────────────────────────────────────────────────────────
    #  MISSION  EXECUTION  (runs in a background thread)
    # ─────────────────────────────────────────────────────────

    def _start_mission(self):
        if self._running_mission:
            self._log("Mission already running")
            return
        # Pre-flight checks
        if self.target_pos is None:
            messagebox.showwarning("No target", "Set a target first")
            return
        if not self.detector.at_ok:
            messagebox.showerror("Missing library",
                "pupil_apriltags not installed – cannot detect robot position")
            return

        self._running_mission = True
        self._retry_count     = 0
        self.start_btn.config(state=tk.DISABLED)
        self.abort_btn.config(state=tk.NORMAL)
        self.mission_lbl.config(text="Mission running…", foreground="#004488")
        self._log("═══ Mission started ═══")

        self._mission_thread = threading.Thread(
            target=self._mission_loop, daemon=True)
        self._mission_thread.start()

    def _abort_mission(self):
        self._running_mission = False
        self.robot_client.abort_path()
        self.robot_client.stop()
        self._ui_mission_done("⛔ Mission aborted")

    def _ui_mission_done(self, msg: str):
        """Update UI labels from any thread."""
        def _do():
            self.start_btn.config(state=tk.NORMAL)
            self.abort_btn.config(state=tk.DISABLED)
            self.mission_lbl.config(text=msg,
                foreground="#007700" if "reached" in msg.lower() else "#cc0000")
            self._running_mission = False
        self.root.after(0, _do)

    def _mission_loop(self):
        """
        Background thread that implements the photo → plan → move → verify cycle.

        Pseudocode
        ──────────
        while not reached and retries < max_retries:
            1. Capture photo
            2. Detect robot + obstacles
            3. If robot not found: abort
            4. Plan A* path to target
            5. Estimate travel time from calibration
            6. Send path to Arduino via POST /path
            7. Wait estimated_time + settle_margin ms
            8. Capture verification photo
            9. Detect robot position in verification photo
           10. Check distance(robot_pos, target_pos) < arrival_threshold
        """
        max_retries     = self.max_retry_var.get()
        arrival_thr_px  = self.arrival_thr_var.get()

        while self._running_mission and self._retry_count <= max_retries:
            attempt = self._retry_count + 1
            self._log(f"─── Attempt {attempt}/{max_retries+1} ───")
            self.root.after(0, lambda a=attempt, m=max_retries:
                self.retry_lbl.config(text=f"Attempt {a}/{m+1}"))

            # ── STEP 1: Capture photo ──────────────────────────
            self._log("Step 1 – Capturing arena photo…")
            ok = self._thread_safe_capture()
            if not ok:
                self._log("✗ Failed to capture photo – aborting")
                self._ui_mission_done("✗ Capture failed")
                return

            # ── STEP 2: Detect ────────────────────────────────
            self._log("Step 2 – Running detection…")
            found = self._thread_safe_detect()
            if not found:
                self._log("✗ Robot not found – aborting mission")
                self._ui_mission_done("✗ Robot not detected")
                return

            # ── STEP 3: Check if already at target ────────────
            dist = math.hypot(
                self.target_pos[0] - self.robot_pos[0],
                self.target_pos[1] - self.robot_pos[1])
            self._log(f"Distance to target: {dist:.1f} px  "
                      f"(threshold: {arrival_thr_px} px)")
            if dist <= arrival_thr_px:
                self._log("✓ Robot already at target – mission complete!")
                self._ui_mission_done("✓ Target reached!")
                return

            # ── STEP 4: Plan path ──────────────────────────────
            self._log("Step 4 – Planning A* path…")
            ok = self._thread_safe_plan()
            if not ok or not self.planned_path:
                self._log("✗ Path planning failed – aborting")
                self._ui_mission_done("✗ Path planning failed")
                return

            # ── STEP 5: Estimate travel time ───────────────────
            settle_ms  = self.settle_var.get()
            travel_ms  = estimate_travel_time_ms(
                self.planned_path, self.cm_per_pixel)
            # Override settle from UI
            travel_ms  = travel_ms - SETTLE_MARGIN_MS + settle_ms
            total_wait = travel_ms
            self._log(
                f"Step 5 – Estimated travel: {travel_ms:.0f} ms  "
                f"(settle: {settle_ms} ms)")

            # ── STEP 6: Send path to Arduino ───────────────────
            self._log("Step 6 – Sending path to Arduino…")
            self.robot_client.ip   = self.robot_ip_var.get().strip()
            self.robot_client.port = int(self.robot_port_var.get().strip())
            ok = self.robot_client.send_path(
                self.planned_path, self.cm_per_pixel)
            if not ok:
                self._log("✗ Failed to send path – aborting")
                self._ui_mission_done("✗ Path send failed")
                return
            self._log(f"Path accepted by Arduino ({len(self.planned_path)} waypoints)")

            # ── STEP 7: Wait for motion ────────────────────────
            self._log(f"Step 7 – Waiting {total_wait:.0f} ms for motion…")
            t_end = time.time() + total_wait / 1000.0
            while time.time() < t_end and self._running_mission:
                remaining = t_end - time.time()
                self.root.after(0, lambda r=remaining:
                    self.mission_lbl.config(
                        text=f"Moving… {r:.1f}s remaining",
                        foreground="#004488"))
                time.sleep(0.1)

            if not self._running_mission:
                return

            # ── STEP 8: Verification photo ─────────────────────
            self._log("Step 8 – Capturing verification photo…")
            ok = self._thread_safe_capture()
            if not ok:
                self._log("⚠  Verification capture failed – retrying")
                self._retry_count += 1
                continue

            # ── STEP 9: Detect robot position ──────────────────
            self._log("Step 9 – Detecting robot in verification photo…")
            found = self._thread_safe_detect()
            if not found:
                self._log("⚠  Robot not detected in verification photo")
                self._retry_count += 1
                continue

            # ── STEP 10: Check arrival ─────────────────────────
            dist = math.hypot(
                self.target_pos[0] - self.robot_pos[0],
                self.target_pos[1] - self.robot_pos[1])
            dist_cm = dist * self.cm_per_pixel if self.cm_per_pixel else 0
            self._log(
                f"Step 10 – Post-move distance: {dist:.1f} px  "
                f"({dist_cm:.1f} cm)  threshold: {arrival_thr_px} px")

            if dist <= arrival_thr_px:
                self._log("✓ Target reached! Mission complete.")
                self._ui_mission_done("✓ Target reached!")
                return
            else:
                self._log(f"⚠  Not yet at target – retry {attempt}/{max_retries}")
                self._retry_count += 1

        # Exhausted retries
        self._log("✗ Max retries reached without reaching target")
        self._ui_mission_done("✗ Failed – retries exhausted")

    # ── Thread-safe wrappers (schedule on main thread, block until done) ──

    def _thread_safe_capture(self) -> bool:
        result = [False]
        event  = threading.Event()
        def _do():
            result[0] = self._capture_webcam()
            event.set()
        self.root.after(0, _do)
        event.wait(timeout=15)
        return result[0]

    def _thread_safe_detect(self) -> bool:
        result = [False]
        event  = threading.Event()
        def _do():
            result[0] = self._detect_current()
            event.set()
        self.root.after(0, _do)
        event.wait(timeout=10)
        return result[0]

    def _thread_safe_plan(self) -> bool:
        result = [False]
        event  = threading.Event()
        def _do():
            result[0] = self._plan_path()
            event.set()
        self.root.after(0, _do)
        event.wait(timeout=10)
        return result[0]

    # ─────────────────────────────────────────────────────────
    #  KEYBOARD  SHORTCUT
    # ─────────────────────────────────────────────────────────

    def _on_key(self, event):
        key = event.keysym.upper()
        if key == "SPACE":
            self._emergency_stop()
        elif key == "D" and event.state & 0x4:   # Ctrl+D = detect
            self._detect_current()
        elif key == "P" and event.state & 0x4:   # Ctrl+P = plan
            self._plan_path()
        elif key == "RETURN" and event.state & 0x4:  # Ctrl+Enter = start mission
            self._start_mission()

    # ─────────────────────────────────────────────────────────
    #  CLEANUP
    # ─────────────────────────────────────────────────────────

    def _on_close(self):
        self._running_mission = False
        self.robot_client.stop()
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════
#  ENTRY  POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = OpenLoopTestGUI(root)
    root.mainloop()