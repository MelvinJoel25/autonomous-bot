"""
swarm1_navigation.py  –  AUTOBOT Navigation GUI
═══════════════════════════════════════════════════════════════════════════════
Navigation GUI for AUTOBOT Ackermann-style robot (27.5 cm × 20.5 cm)

Communication Path
───────────────────
  PC  ──TCP──►  ESP32 @ 192.168.0.51:5000
  ESP32 ──UART──► STM32 (+XXX-YYY\r format)

AprilTag IDs
─────────────
  0 = AUTOBOT   (magenta)   – treated as obstacle / other agent
  1 = SWARM-1   (green)     – treated as obstacle / other agent
  2 = SWARM-2   (orange)
  3 = SWARM-3   (cyan)

Command Format to ESP32 (TCP, then forwarded as UART to STM32)
───────────────────────────────────────────────────────────────
  "+XXX-YYY\n"  where XXX = |x| cm, YYY = |y| cm
  Signs: + = right / forward,  - = left / backward
  Max value per axis: ±999 cm

YOLO classes (shared trained model)
──────────────────────────────────────
  0=box  1=sticky_note  2=bottle  3=plastic  4=table  5=tape

Robot dimensions
─────────────────
  Length : 27.5 cm  (along Y / forward axis)
  Width  : 20.5 cm  (along X / lateral axis)
  AprilTag size : 12 cm × 12 cm

Coordinate system
──────────────────
  World frame origin = robot's position at detection time
  +X = right,  +Y = forward (up on screen)
  The STM32 maintains its own worldX/worldY and computes
  the delta from the last commanded position automatically.
  This GUI sends ABSOLUTE world coordinates; the STM32
  subtracts its remembered position to get the relative move.
═══════════════════════════════════════════════════════════════════════════════
"""

import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import heapq
import os
import threading
import time
import socket
import math
from pathlib import Path
from collections import defaultdict, deque

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

TAG_SIZE_CM         = 12.0          # AprilTag physical side length
ROBOT_LENGTH_CM     = 27.5          # AUTOBOT body length (Y axis)
ROBOT_WIDTH_CM      = 20.5          # AUTOBOT body width  (X axis)
WHEEL_MARGIN_CM     = 4.0           # extra clearance beyond body edge
STOP_THRESHOLD_CM   = 8.0           # auto-follow: stop within this distance

YOLO_MODEL_PATH     = r"C:\Users\meljo\krab-3\runs\obstacle_v1\weights\best.pt"
YOLO_CONF           = 0.45
YOLO_MIN_CONF       = 0.40
YOLO_IOU            = 0.45

YOLO_CLASSES = {
    0: "box",
    1: "sticky_note",
    2: "bottle",
    3: "plastic",
    4: "table",
    5: "tape",
}
DEFAULT_IGNORE_CLASSES = {4}   # table

CLASS_COLORS = {
    0: (0,  100, 255),
    1: (0,  220, 220),
    2: (255, 60,  60),
    3: (0,  200,  60),
    4: (130,130, 130),
    5: (200, 0,  200),
}

DISPLAY_W          = 1280
DISPLAY_H          = 860
LEFT_PANEL_W       = 360
TARGET_FPS         = 30
DETECT_EVERY_N     = 2

# ESP32 TCP bridge
DEFAULT_ESP32_IP   = "192.168.0.51"
DEFAULT_ESP32_PORT = 5000

ROBOT_COLORS = {0:(255,0,255), 1:(0,255,0), 2:(0,165,255), 3:(255,255,0)}
ROBOT_NAMES  = {0:"AUTOBOT",  1:"SWARM-1", 2:"SWARM-2",   3:"SWARM-3"}

EDGE_MARGIN_PX     = 30
HEAT_THRESHOLD     = 3
HEAT_DECAY         = 1
BOX_SHRINK_PX      = 4
OBSTACLE_INFLATE   = 1
RDP_EPSILON        = 8.0
SHOW_KEEPOUT_BOX   = True
DEBUG_GRID         = False


# ══════════════════════════════════════════════════════════════
#  A* PLANNER  (diagonal √2 cost)
# ══════════════════════════════════════════════════════════════

class AStarPlanner:
    _DIRS = [(0,1,1.0),(1,0,1.0),(0,-1,1.0),(-1,0,1.0),
             (1,1,1.414),(-1,-1,1.414),(1,-1,1.414),(-1,1,1.414)]

    def __init__(self, grid):
        self.grid = grid
        self.h, self.w = grid.shape

    @staticmethod
    def _heuristic(a, b):
        dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
        return 1.0*(abs(dr-dc)) + 1.414*min(dr, dc)

    def _neighbors(self, pos):
        out = []
        for dr, dc, cost in self._DIRS:
            nr, nc = pos[0]+dr, pos[1]+dc
            if 0 <= nr < self.h and 0 <= nc < self.w and self.grid[nr,nc] == 0:
                out.append(((nr, nc), cost))
        return out

    def plan(self, start, goal):
        heap  = [(0.0, start)]
        came  = {}
        g     = {start: 0.0}
        while heap:
            _, cur = heapq.heappop(heap)
            if cur == goal:
                path = []
                while cur in came:
                    path.append(cur)
                    cur = came[cur]
                path.append(start)
                return path[::-1]
            for nb, cost in self._neighbors(cur):
                tg = g[cur] + cost
                if nb not in g or tg < g[nb]:
                    came[nb] = cur
                    g[nb]    = tg
                    heapq.heappush(heap, (tg + self._heuristic(nb, goal), nb))
        return None


# ══════════════════════════════════════════════════════════════
#  PATH UTILITIES
# ══════════════════════════════════════════════════════════════

def rdp_simplify(points, epsilon):
    if len(points) < 3 or epsilon <= 0:
        return points
    start, end = np.array(points[0]), np.array(points[-1])
    line = end - start
    line_len = np.linalg.norm(line)
    if line_len == 0:
        return [points[0], points[-1]]
    line_unit = line / line_len
    vecs  = np.array(points) - start
    proj  = np.dot(vecs, line_unit)
    perp  = vecs - np.outer(proj, line_unit)
    dists = np.linalg.norm(perp, axis=1)
    idx   = int(np.argmax(dists))
    if dists[idx] > epsilon:
        left  = rdp_simplify(points[:idx+1], epsilon)
        right = rdp_simplify(points[idx:],   epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


# ══════════════════════════════════════════════════════════════
#  ESP32 TCP CLIENT
# ══════════════════════════════════════════════════════════════

class Swarm1Client:
    """
    Sends navigation commands to AUTOBOT via ESP32 TCP bridge.

    The ESP32 expects: "+XXX-YYY\n"  (variable length, padded to 3 digits each)
    It then forwards "+XXX-YYY\r" to the STM32 over UART.

    We send ABSOLUTE world coordinates (cm).  The STM32 tracks its own
    worldX / worldY and computes the delta internally.
    """

    def __init__(self, ip=DEFAULT_ESP32_IP, port=DEFAULT_ESP32_PORT):
        self.ip        = ip
        self.port      = port
        self.connected = False
        self._sock     = None
        self._lock     = threading.Lock()
        self.last_cmd  = None
        # Tracked absolute position sent to STM32 (mirrors STM32 worldX/worldY)
        self.world_x   = 0   # cm
        self.world_y   = 0   # cm

    def connect(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((self.ip, self.port))
            s.settimeout(None)
            self._sock     = s
            self.connected = True
            return True, ""
        except Exception as e:
            return False, str(e)

    def disconnect(self):
        with self._lock:
            self.connected = False
            if self._sock:
                try: self._sock.close()
                except: pass
                self._sock = None

    def send_absolute(self, abs_x_cm: int, abs_y_cm: int) -> tuple:
        """
        Send an absolute world coordinate command.
        Clamps to ±999 cm per axis.
        Returns (status_str, elapsed_ms).
        """
        if not self.connected:
            return "NOT_CONNECTED", 0

        abs_x_cm = max(-999, min(999, int(abs_x_cm)))
        abs_y_cm = max(-999, min(999, int(abs_y_cm)))

        xs = '+' if abs_x_cm >= 0 else '-'
        ys = '+' if abs_y_cm >= 0 else '-'
        cmd = f"{xs}{abs(abs_x_cm):03d}{ys}{abs(abs_y_cm):03d}\n"

        with self._lock:
            try:
                t = time.time()
                self._sock.sendall(cmd.encode())
                ms = (time.time() - t) * 1000
                self.last_cmd = cmd.strip()
                # Mirror position
                self.world_x = abs_x_cm
                self.world_y = abs_y_cm
                return "OK", ms
            except Exception as e:
                self.connected = False
                return f"ERROR:{e}", 0

    def send_absolute_async(self, abs_x_cm: int, abs_y_cm: int):
        threading.Thread(
            target=self.send_absolute,
            args=(abs_x_cm, abs_y_cm),
            daemon=True
        ).start()

    def reset_origin(self):
        """Reset tracked world position (call when STM32 is power-cycled)."""
        self.world_x = 0
        self.world_y = 0


# ══════════════════════════════════════════════════════════════
#  YOLO ASYNC WORKER
# ══════════════════════════════════════════════════════════════

class YOLOWorker:
    def __init__(self, model):
        self.model      = model
        self._frame_q   = deque(maxlen=1)
        self._result_q  = deque(maxlen=1)
        self._conf      = YOLO_CONF
        self._running   = True
        self._thread    = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def post_frame(self, frame, conf):
        self._conf = conf
        self._frame_q.append(frame.copy())

    def get_result(self):
        if self._result_q:
            return self._result_q.pop()
        return None

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            if not self._frame_q:
                time.sleep(0.005)
                continue
            frame = self._frame_q.pop()
            try:
                conf    = max(float(self._conf), YOLO_MIN_CONF)
                results = self.model(frame, conf=conf, iou=YOLO_IOU, verbose=False)
                boxes   = []
                for r in results:
                    for box in r.boxes:
                        x1,y1,x2,y2 = box.xyxy[0].tolist()
                        cid  = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        boxes.append((int(x1),int(y1),int(x2),int(y2),cid,conf))
                self._result_q.append(boxes)
            except Exception as e:
                if DEBUG_GRID:
                    print(f"[YOLO thread] {e}")


# ══════════════════════════════════════════════════════════════
#  MAIN GUI
# ══════════════════════════════════════════════════════════════

class Swarm1NavigationGUI:

    def __init__(self, root):
        self.root = root
        self.root.title(
            "AUTOBOT Navigation  –  Ackermann Bot  |  ESP32 TCP Bridge  |  AprilTag ID 0")
        self.root.geometry("1860x1060")
        self.root.state("zoomed")

        # ── AprilTag detector ─────────────────────────────────
        if APRILTAG_OK:
            self.at_detector = pupil_apriltags.Detector(
                families="tag36h11", nthreads=4,
                quad_decimate=1.0, quad_sigma=0.0,
                refine_edges=1, decode_sharpening=0.25,
            )
        else:
            self.at_detector = None
            messagebox.showerror(
                "Missing library",
                "pupil_apriltags not installed!\npip install pupil-apriltags")

        # ── YOLO ──────────────────────────────────────────────
        self.yolo_model  = None
        self.yolo_worker = None
        if YOLO_OK:
            try:
                self.yolo_model  = YOLO(YOLO_MODEL_PATH)
                self.yolo_worker = YOLOWorker(self.yolo_model)
                print(f"✓ YOLO loaded: {YOLO_MODEL_PATH}")
            except Exception as e:
                print(f"Custom model failed ({e}), trying yolo11n.pt …")
                try:
                    self.yolo_model  = YOLO("yolo11n.pt")
                    self.yolo_worker = YOLOWorker(self.yolo_model)
                    print("✓ YOLO loaded: yolo11n.pt (pretrained fallback)")
                except Exception as e2:
                    print(f"✗ YOLO load failed: {e2}")

        self._class_ignore_vars = {}
        self._yolo_latest       = []
        self._yolo_lock         = threading.Lock()

        # ── Video / image state ───────────────────────────────
        self.cap          = None
        self.original_img = None
        self.display_img  = None
        self.is_capturing = False
        self.is_video     = False
        self.is_image     = False
        self._frame_delay = int(1000 / TARGET_FPS)
        self._recorder    = None
        self._recording   = False

        # ── Robot / detection state ───────────────────────────
        self.swarm1_pos   = None    # (px, py) in display coords
        self.swarm1_theta = 0.0     # degrees
        self.swarm1_bbox  = None    # (x1,y1,x2,y2) of tag bounding box
        self.all_tags     = {}      # tid -> (cx,cy,theta,corners)
        self.robot_boxes  = {}      # tid -> (x1,y1,x2,y2) for OTHER robots
        self.obstacles    = []
        self._class_conf_accum = defaultdict(list)

        # ── Calibration ───────────────────────────────────────
        self.cm_per_pixel   = None
        self.grid_size_px   = 20
        self.stop_thresh_px = 30

        # ── Path / grid ───────────────────────────────────────
        self.target_pos  = None
        self.path        = None
        self.grid        = None
        self._heat_grid      = None
        self._heat_grid_size = None

        # ── World coordinate tracking (mirrors STM32) ─────────
        # Origin is wherever AUTOBOT was when the GUI started.
        # Clicking a target in CM or on canvas updates target_world_cm.
        self.origin_px         = None   # pixel position when origin was set
        self.target_world_cm   = None   # (x_cm, y_cm) absolute world target

        # ── Communication ─────────────────────────────────────
        self.robot = Swarm1Client(DEFAULT_ESP32_IP, DEFAULT_ESP32_PORT)

        # ── Tkinter variables ─────────────────────────────────
        self.auto_replan          = tk.BooleanVar(value=True)
        self.auto_follow          = tk.BooleanVar(value=False)
        self.show_detections      = tk.BooleanVar(value=True)
        self.show_grid            = tk.BooleanVar(value=False)
        self.show_path            = tk.BooleanVar(value=True)
        self.auto_detect          = tk.BooleanVar(value=False)
        self.show_heat            = tk.BooleanVar(value=True)
        self.show_keepout         = tk.BooleanVar(value=SHOW_KEEPOUT_BOX)
        self.use_rdp              = tk.BooleanVar(value=True)
        self.use_inflation        = tk.BooleanVar(value=True)
        self.manual_calib_enabled = tk.BooleanVar(value=False)
        self.detect_n_var         = tk.IntVar(value=DETECT_EVERY_N)
        self.heat_thresh_var      = tk.IntVar(value=HEAT_THRESHOLD)
        self.heat_decay_var       = tk.IntVar(value=HEAT_DECAY)
        self.inflate_var          = tk.IntVar(value=OBSTACLE_INFLATE)
        self.manual_cmpx_var      = tk.DoubleVar(value=0.04)
        self.wheel_margin_var     = tk.DoubleVar(value=WHEEL_MARGIN_CM)
        self.coord_x_var          = tk.StringVar(value="0")
        self.coord_y_var          = tk.StringVar(value="0")
        self.coord_mode_var       = tk.StringVar(value="absolute")  # absolute | relative

        # ── Perf tracking ─────────────────────────────────────
        self._frame_count = 0
        self._fps_display = 0.0
        self._fps_counter = 0
        self._fps_timer   = time.time()

        self.coord_x_var.trace_add("write", lambda *_: self._update_coord_preview())
        self.coord_y_var.trace_add("write", lambda *_: self._update_coord_preview())
        self.coord_mode_var.trace_add("write", lambda *_: self._on_coord_mode_change())

        self._setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────
    #  KEEP-OUT BOX HELPERS  (rectangle, not circle)
    # ─────────────────────────────────────────────────────────

    def _robot_keepout_half_size_px(self):
        """
        Returns (half_w_px, half_h_px) of the rectangular keep-out zone
        centred on the robot tag.  Oriented with the robot heading.
        """
        margin = float(self.wheel_margin_var.get())
        half_w_cm = ROBOT_WIDTH_CM  / 2.0 + margin
        half_h_cm = ROBOT_LENGTH_CM / 2.0 + margin
        if self.cm_per_pixel and self.cm_per_pixel > 0:
            hw = max(8, int(half_w_cm / self.cm_per_pixel))
            hh = max(8, int(half_h_cm / self.cm_per_pixel))
        else:
            hw = max(8, int(self.grid_size_px * 0.75))
            hh = max(8, int(self.grid_size_px * 1.0))
        return hw, hh

    def _keepout_corners_px(self):
        """
        Returns the 4 pixel corners of the rotated keep-out rectangle
        (for drawing and hit-testing).  Returns None if robot not detected.
        """
        if self.swarm1_pos is None:
            return None
        cx, cy   = self.swarm1_pos
        hw, hh   = self._robot_keepout_half_size_px()
        theta    = math.radians(self.swarm1_theta)
        # Local corners: (±hw, ±hh)
        local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        world = []
        for lx, ly in local:
            wx = int(cx + lx * cos_t - ly * sin_t)
            wy = int(cy + lx * sin_t + ly * cos_t)
            world.append((wx, wy))
        return world

    def _axis_aligned_keepout_bbox(self):
        """AABB of the rotated keep-out box — used for masking and grid clear."""
        corners = self._keepout_corners_px()
        if corners is None:
            return None
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _point_in_rotated_rect(px, py, cx, cy, hw, hh, theta_rad):
        """True if point (px,py) is inside the axis-aligned rectangle
        defined in the robot's local frame."""
        cos_t, sin_t = math.cos(-theta_rad), math.sin(-theta_rad)
        dx, dy = px - cx, py - cy
        lx =  dx * cos_t - dy * sin_t
        ly =  dx * sin_t + dy * cos_t
        return abs(lx) <= hw and abs(ly) <= hh

    def _box_intersects_keepout(self, x1, y1, x2, y2):
        """
        True if bounding box [x1,y1,x2,y2] overlaps the robot's
        keep-out rotated rectangle.  Checks all 4 box corners + centre.
        """
        if self.swarm1_pos is None:
            return False
        cx, cy = self.swarm1_pos
        hw, hh = self._robot_keepout_half_size_px()
        theta  = math.radians(self.swarm1_theta)
        points = [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2),
            ((x1+x2)//2, (y1+y2)//2)
        ]
        for px, py in points:
            if self._point_in_rotated_rect(px, py, cx, cy, hw, hh, theta):
                return True
        return False

    # ─────────────────────────────────────────────────────────
    #  COORDINATE CONVERSION HELPERS
    # ─────────────────────────────────────────────────────────

    def _px_to_world_cm(self, px, py):
        """
        Convert display-pixel position to world cm relative to origin.
        +X = right, +Y = upward (screen-Y is flipped).
        Returns None if calibration or origin not available.
        """
        if self.cm_per_pixel is None or self.origin_px is None:
            return None
        ox, oy = self.origin_px
        dx_cm =  (px - ox) * self.cm_per_pixel
        dy_cm = -(py - oy) * self.cm_per_pixel   # flip Y
        return round(dx_cm, 1), round(dy_cm, 1)

    def _world_cm_to_px(self, x_cm, y_cm):
        """Inverse: world cm → display pixel."""
        if self.cm_per_pixel is None or self.origin_px is None:
            return None
        ox, oy = self.origin_px
        px = int(ox + x_cm / self.cm_per_pixel)
        py = int(oy - y_cm / self.cm_per_pixel)
        return px, py

    def _set_origin_at_robot(self):
        """Pin the world-coordinate origin to the robot's current pixel position."""
        if self.swarm1_pos is not None:
            self.origin_px = self.swarm1_pos
            self.robot.reset_origin()
            self.status.config(
                text=f"✓ Origin set at AUTOBOT position {self.swarm1_pos}")

    # ─────────────────────────────────────────────────────────
    #  UI BUILD
    # ─────────────────────────────────────────────────────────

    def _setup_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding="6")
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, minsize=LEFT_PANEL_W, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # ── Left scrollable panel ──────────────────────────────
        lc = ttk.Frame(main, width=LEFT_PANEL_W)
        lc.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        lc.grid_propagate(False)

        lcan = tk.Canvas(lc, bg="white", highlightthickness=0,
                         width=LEFT_PANEL_W - 20)
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

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        # ═══ SOURCE ═══════════════════════════════════════════
        sf = ttk.LabelFrame(left, text="📷  Source", padding="8")
        sf.pack(fill=tk.X, pady=4)

        ttk.Button(sf, text="📁  Load Image",
                   command=self._load_image).pack(fill=tk.X, pady=2)
        ttk.Button(sf, text="🎬  Load Video",
                   command=self._load_video).pack(fill=tk.X, pady=2)
        ttk.Separator(sf, orient="horizontal").pack(fill=tk.X, pady=4)

        ttk.Label(sf, text="📹  Webcam", font=("Arial",9,"bold")).pack(anchor=tk.W)
        wf = ttk.Frame(sf); wf.pack(fill=tk.X, pady=2)
        ttk.Label(wf, text="Index:").pack(side=tk.LEFT, padx=(0,4))
        self.webcam_idx = ttk.Spinbox(wf, from_=0, to=10, width=3)
        self.webcam_idx.set(0); self.webcam_idx.pack(side=tk.LEFT)
        self.webcam_btn = ttk.Button(sf, text="Connect Webcam",
                                     command=self._connect_webcam)
        self.webcam_btn.pack(fill=tk.X, pady=3)
        self.webcam_dis = ttk.Button(sf, text="Disconnect Webcam",
                                     command=self._disconnect, state=tk.DISABLED)
        self.webcam_dis.pack(fill=tk.X, pady=2)

        ttk.Separator(sf, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Label(sf, text="🌐  IP Camera / DroidCam",
                  font=("Arial",9,"bold")).pack(anchor=tk.W)
        ttk.Label(sf, text="IP:").pack(anchor=tk.W)
        self.droid_ip = ttk.Entry(sf); self.droid_ip.insert(0,"192.168.1.100")
        self.droid_ip.pack(fill=tk.X, pady=2)
        ttk.Label(sf, text="Port:").pack(anchor=tk.W)
        self.droid_port = ttk.Entry(sf); self.droid_port.insert(0,"4747")
        self.droid_port.pack(fill=tk.X, pady=2)
        self.droid_btn = ttk.Button(sf, text="Connect IP Camera",
                                    command=self._connect_droidcam)
        self.droid_btn.pack(fill=tk.X, pady=3)
        self.droid_dis = ttk.Button(sf, text="Disconnect",
                                    command=self._disconnect, state=tk.DISABLED)
        self.droid_dis.pack(fill=tk.X)

        ttk.Separator(sf, orient="horizontal").pack(fill=tk.X, pady=4)
        self.rec_btn = ttk.Button(sf, text="⏺  Start Recording",
                                  command=self._toggle_recording)
        self.rec_btn.pack(fill=tk.X, pady=2)
        self.rec_lbl = ttk.Label(sf, text="Not recording",
                                 foreground="#888888", font=("Arial",8))
        self.rec_lbl.pack()

        # ═══ DETECTION ════════════════════════════════════════
        df = ttk.LabelFrame(left, text="🔍  Detection", padding="8")
        df.pack(fill=tk.X, pady=4)

        self.at_status = ttk.Label(df,
            text=f"AprilTag: {'✓ ready' if APRILTAG_OK else '❌ missing'}",
            font=("Arial",9))
        self.at_status.pack(anchor=tk.W)
        self.yolo_status = ttk.Label(df,
            text=f"YOLO: {'✓ async worker' if self.yolo_worker else '⚠ not loaded'}",
            font=("Arial",9))
        self.yolo_status.pack(anchor=tk.W, pady=(2,4))

        ttk.Button(df, text="🔍  Detect Once",
                   command=self._detect_once).pack(fill=tk.X, pady=3)
        ttk.Checkbutton(df, text="🔄  Continuous Detection",
                        variable=self.auto_detect,
                        command=self._on_auto_detect_toggle).pack(anchor=tk.W, pady=2)

        ttk.Label(df, text="Detect every N frames:").pack(anchor=tk.W)
        ttk.Spinbox(df, from_=1, to=10,
                    textvariable=self.detect_n_var, width=5).pack(anchor=tk.W)

        self.det_info = ttk.Label(df, text="Tags: –   Obstacles: –",
                                  font=("Arial",9,"bold"))
        self.det_info.pack(pady=4)
        self.fps_lbl = ttk.Label(df, text="FPS: –", font=("Arial",8))
        self.fps_lbl.pack()

        ttk.Checkbutton(df, text="Show Detections",
                        variable=self.show_detections,
                        command=self._redraw).pack(anchor=tk.W)

        ttk.Label(df, text="YOLO Confidence (≥0.40):").pack(anchor=tk.W, pady=(4,0))
        self.conf_scale = ttk.Scale(df, from_=0.40, to=0.90, orient=tk.HORIZONTAL)
        self.conf_scale.set(YOLO_CONF)
        self.conf_scale.pack(fill=tk.X)
        self.conf_lbl = ttk.Label(df, text=f"{YOLO_CONF:.2f}")
        self.conf_lbl.pack()
        self.conf_scale.config(
            command=lambda v: self.conf_lbl.config(text=f"{float(v):.2f}"))

        self.cls_conf_lbl = ttk.Label(df, text="",
                                      font=("Courier",7), foreground="#555555")
        self.cls_conf_lbl.pack(anchor=tk.W, pady=2)

        ttk.Separator(df, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Label(df, text="🚫  Ignore classes:",
                  font=("Arial",9,"bold")).pack(anchor=tk.W)
        for cid, cname in YOLO_CLASSES.items():
            var = tk.BooleanVar(value=(cid in DEFAULT_IGNORE_CLASSES))
            self._class_ignore_vars[cid] = var
            r, g, b = CLASS_COLORS[cid][2], CLASS_COLORS[cid][1], CLASS_COLORS[cid][0]
            hex_col = f"#{r:02x}{g:02x}{b:02x}"
            cb = tk.Checkbutton(df, text=f"  {cname}",
                                variable=var, anchor=tk.W,
                                fg=hex_col, font=("Arial",8,"bold"),
                                command=self._on_class_ignore_change)
            cb.pack(anchor=tk.W)

        # ═══ CALIBRATION & KEEP-OUT ════════════════════════════
        gf = ttk.LabelFrame(left,
            text="📐  Calibration, Grid & Keep-Out", padding="8")
        gf.pack(fill=tk.X, pady=4)

        ttk.Label(gf, text=f"Tag size   : {TAG_SIZE_CM} cm").pack(anchor=tk.W)
        ttk.Label(gf, text=f"Bot length : {ROBOT_LENGTH_CM} cm  (Y axis)").pack(anchor=tk.W)
        ttk.Label(gf, text=f"Bot width  : {ROBOT_WIDTH_CM} cm  (X axis)").pack(anchor=tk.W)
        ttk.Label(gf, text=f"Stop dist  : {STOP_THRESHOLD_CM} cm").pack(anchor=tk.W)

        self.calib_lbl = ttk.Label(gf, text="cm/pixel: calibrating…",
                                   font=("Arial",9,"bold"), foreground="#888888")
        self.calib_lbl.pack(pady=3)
        self.grid_lbl  = ttk.Label(gf, text="Grid cell: – px", font=("Arial",9))
        self.grid_lbl.pack()
        self.keepout_lbl = ttk.Label(gf, text="Keep-out box: – × – px",
                                     font=("Arial",9,"bold"), foreground="#cc4400")
        self.keepout_lbl.pack(pady=2)

        ttk.Separator(gf, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Label(gf, text="🔧  Wheel margin (cm beyond body edge):",
                  font=("Arial",9,"bold")).pack(anchor=tk.W)
        wm_frame = ttk.Frame(gf); wm_frame.pack(fill=tk.X, pady=2)
        ttk.Spinbox(wm_frame, from_=0.0, to=15.0, increment=0.5,
                    textvariable=self.wheel_margin_var, width=6,
                    command=self._on_wheel_margin_change).pack(side=tk.LEFT)
        ttk.Label(wm_frame, text=" cm").pack(side=tk.LEFT)
        ttk.Button(wm_frame, text="Apply",
                   command=self._on_wheel_margin_change).pack(side=tk.LEFT, padx=6)

        ttk.Checkbutton(gf, text="Show Keep-Out Box on canvas",
                        variable=self.show_keepout,
                        command=self._redraw).pack(anchor=tk.W, pady=2)

        ttk.Separator(gf, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Checkbutton(gf, text="Manual calibration override",
                        variable=self.manual_calib_enabled,
                        command=self._on_manual_calib_toggle).pack(anchor=tk.W)
        self.manual_calib_frame = ttk.Frame(gf)
        self.manual_calib_frame.pack(fill=tk.X)
        ttk.Label(self.manual_calib_frame, text="cm/px:").pack(
            side=tk.LEFT, padx=(0,4))
        self.manual_cmpx_entry = ttk.Entry(
            self.manual_calib_frame, width=8, textvariable=self.manual_cmpx_var)
        self.manual_cmpx_entry.pack(side=tk.LEFT)
        self.manual_cmpx_entry.config(state=tk.DISABLED)
        ttk.Button(self.manual_calib_frame, text="Apply",
                   command=self._apply_manual_calib).pack(side=tk.LEFT, padx=4)

        ttk.Separator(gf, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Label(gf, text="Heat threshold (frames):").pack(anchor=tk.W)
        ttk.Spinbox(gf, from_=1, to=20,
                    textvariable=self.heat_thresh_var, width=5).pack(anchor=tk.W)
        ttk.Label(gf, text="Heat decay (per frame):").pack(anchor=tk.W)
        ttk.Spinbox(gf, from_=1, to=10,
                    textvariable=self.heat_decay_var, width=5).pack(anchor=tk.W)
        ttk.Label(gf, text="Obstacle inflate (cells):").pack(anchor=tk.W)
        ttk.Spinbox(gf, from_=0, to=5,
                    textvariable=self.inflate_var, width=5).pack(anchor=tk.W)

        # ═══ POSITIONS ════════════════════════════════════════
        pf = ttk.LabelFrame(left, text="📍  Positions & World Origin", padding="8")
        pf.pack(fill=tk.X, pady=4)

        self.swarm1_lbl = ttk.Label(pf,
            text="🚗 AUTOBOT: Not detected", font=("Arial",9))
        self.swarm1_lbl.pack(anchor=tk.W, pady=2)
        self.origin_lbl = ttk.Label(pf,
            text="🌐 Origin: Not set", font=("Arial",9), foreground="#888888")
        self.origin_lbl.pack(anchor=tk.W, pady=2)
        self.target_lbl = ttk.Label(pf,
            text="🎯 Target: Not set", font=("Arial",9))
        self.target_lbl.pack(anchor=tk.W, pady=2)

        ttk.Button(pf, text="📌  Set World Origin at Robot Now",
                   command=self._set_origin_at_robot).pack(fill=tk.X, pady=3)
        ttk.Button(pf, text="🔄  Reset STM32 World Position (0,0)",
                   command=self._reset_stm32_position).pack(fill=tk.X, pady=2)
        ttk.Button(pf, text="Clear Target",
                   command=self._clear_target).pack(fill=tk.X, pady=2)

        # ═══ COORDINATE ENTRY ═══════════════════════════════════
        ce = ttk.LabelFrame(left,
            text="🎯  Go-To Coordinate  (world cm, +X=right +Y=forward)",
            padding="8")
        ce.pack(fill=tk.X, pady=4)

        ttk.Label(ce,
            text="Choose absolute or relative mode, then enter X/Y in cm.",
            font=("Arial",8,"italic"), foreground="#555555",
            justify=tk.LEFT).pack(anchor=tk.W, pady=(0,4))

        mode_row = ttk.Frame(ce)
        mode_row.pack(fill=tk.X, pady=(0,4))
        ttk.Label(mode_row, text="Mode:").pack(side=tk.LEFT, padx=(0,6))
        ttk.Radiobutton(mode_row, text="Absolute", value="absolute",
                        variable=self.coord_mode_var,
                        command=self._update_coord_preview).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Relative", value="relative",
                        variable=self.coord_mode_var,
                        command=self._update_coord_preview).pack(side=tk.LEFT, padx=(8,0))

        xy_frame = ttk.Frame(ce); xy_frame.pack(fill=tk.X, pady=2)
        ttk.Label(xy_frame, text="X (cm):", width=8).pack(side=tk.LEFT)
        self.coord_x_entry = ttk.Entry(
            xy_frame, textvariable=self.coord_x_var, width=8)
        self.coord_x_entry.pack(side=tk.LEFT, padx=(0,8))
        ttk.Label(xy_frame, text="Y (cm):", width=8).pack(side=tk.LEFT)
        self.coord_y_entry = ttk.Entry(
            xy_frame, textvariable=self.coord_y_var, width=8)
        self.coord_y_entry.pack(side=tk.LEFT)

        self.coord_preview_lbl = ttk.Label(
            ce, text="", font=("Arial",8), foreground="#006699")
        self.coord_preview_lbl.pack(anchor=tk.W, pady=(2,0))

        ttk.Button(ce, text="📍  Set as Target (show on map)",
                   command=self._goto_from_entry_set_target).pack(fill=tk.X, pady=3)
        self.send_coord_btn = ttk.Button(
            ce, text="🚀  Send to Robot NOW",
            command=self._goto_from_entry_send)
        self.send_coord_btn.pack(fill=tk.X, pady=2)

        self.coord_status_lbl = ttk.Label(
            ce, text="", font=("Arial",8,"bold"), foreground="#007700")
        self.coord_status_lbl.pack(pady=2)

        ttk.Separator(ce, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Label(ce, text="Quick presets:", font=("Arial",8,"bold")).pack(anchor=tk.W)
        presets_frame = ttk.Frame(ce); presets_frame.pack(fill=tk.X, pady=2)
        for label, x, y in [
            ("Fwd 30",  0,  30),
            ("Fwd 50",  0,  50),
            ("R 30",   30,   0),
            ("L 30",  -30,   0),
            ("Home",    0,   0),
        ]:
            ttk.Button(
                presets_frame, text=label, width=6,
                command=lambda _x=x, _y=y: self._preset_goto(_x, _y)
            ).pack(side=tk.LEFT, padx=1)

        self._update_coord_preview()

        # ═══ PATH PLANNING ════════════════════════════════════
        plf = ttk.LabelFrame(left, text="🗺️  Path Planning", padding="8")
        plf.pack(fill=tk.X, pady=4)

        ttk.Label(plf,
            text="Click canvas to set pixel target,\nor use Coordinate Entry above.",
            font=("Arial",8,"italic")).pack(anchor=tk.W, pady=2)
        ttk.Button(plf, text="🧭  Calculate A* Path",
                   command=self._calculate_path).pack(fill=tk.X, pady=4)
        ttk.Checkbutton(plf, text="♻️  Auto Re-plan",
                        variable=self.auto_replan).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(plf, text="📐  RDP Simplification",
                        variable=self.use_rdp).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(plf, text="🛡️  Obstacle Inflation",
                        variable=self.use_inflation).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(plf, text="Show Path",
                        variable=self.show_path,
                        command=self._redraw).pack(anchor=tk.W)
        ttk.Checkbutton(plf, text="Show Grid",
                        variable=self.show_grid,
                        command=self._redraw).pack(anchor=tk.W)
        ttk.Checkbutton(plf, text="Show Heat Map",
                        variable=self.show_heat,
                        command=self._redraw).pack(anchor=tk.W)

        self.path_info = ttk.Label(plf, text="No path", font=("Arial",9))
        self.path_info.pack(pady=4)

        # ═══ ROBOT CONNECTION ════════════════════════════════
        rf = ttk.LabelFrame(left,
            text="🤖  AUTOBOT Connection  (ESP32 TCP Bridge)", padding="8")
        rf.pack(fill=tk.X, pady=4)

        ttk.Label(rf, text="ESP32 IP:", font=("Arial",9,"bold")).pack(anchor=tk.W)
        rif = ttk.Frame(rf); rif.pack(fill=tk.X, pady=2)
        ttk.Label(rif, text="IP:").pack(side=tk.LEFT, padx=(0,4))
        self.robot_ip = ttk.Entry(rif, width=16)
        self.robot_ip.insert(0, DEFAULT_ESP32_IP)
        self.robot_ip.pack(side=tk.LEFT, padx=(0,4))
        ttk.Label(rif, text="Port:").pack(side=tk.LEFT, padx=(0,2))
        self.robot_port_e = ttk.Entry(rif, width=6)
        self.robot_port_e.insert(0, str(DEFAULT_ESP32_PORT))
        self.robot_port_e.pack(side=tk.LEFT)

        self.rob_conn_btn = ttk.Button(rf, text="Connect to ESP32",
                                       command=self._connect_robot)
        self.rob_conn_btn.pack(fill=tk.X, pady=2)
        self.rob_dis_btn  = ttk.Button(rf, text="Disconnect",
                                       command=self._disconnect_robot,
                                       state=tk.DISABLED)
        self.rob_dis_btn.pack(fill=tk.X, pady=2)
        self.rob_status = ttk.Label(rf, text="Not connected", foreground="#aa0000")
        self.rob_status.pack(pady=2)

        ttk.Separator(rf, orient="horizontal").pack(fill=tk.X, pady=4)
        self.world_pos_lbl = ttk.Label(
            rf, text="STM32 world pos: (0, 0) cm",
            font=("Arial",8,"bold"), foreground="#005599")
        self.world_pos_lbl.pack()
        self.last_cmd_lbl = ttk.Label(rf, text="Last cmd: –", font=("Arial",8))
        self.last_cmd_lbl.pack()

        ttk.Checkbutton(rf, text="🚀  Auto Path Follow",
                        variable=self.auto_follow).pack(anchor=tk.W, pady=4)
        self.follow_info = ttk.Label(rf, text="", font=("Arial",8))
        self.follow_info.pack()

        # ═══ EXPORT ═══════════════════════════════════════════
        ef = ttk.LabelFrame(left, text="💾  Export", padding="8")
        ef.pack(fill=tk.X, pady=4)
        ttk.Button(ef, text="Save Screenshot",
                   command=self._save_screenshot).pack(fill=tk.X, pady=2)
        ttk.Button(ef, text="Export Path CSV",
                   command=self._export_path).pack(fill=tk.X, pady=2)

        # ═══ STATS ════════════════════════════════════════════
        stf = ttk.LabelFrame(left, text="📊  Stats", padding="8")
        stf.pack(fill=tk.X, pady=4, expand=True)
        self.stats_text = tk.Text(stf, height=16, width=36,
                                  font=("Courier",7), state=tk.DISABLED)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # ═══ CANVAS (right panel) ═════════════════════════════
        cf = ttk.LabelFrame(right, text="Arena View", padding="4")
        cf.grid(row=0, column=0, sticky="nsew")
        cf.columnconfigure(0, weight=1)
        cf.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(cf, width=DISPLAY_W, height=DISPLAY_H,
                                bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self.status = ttk.Label(
            self.root,
            text="AUTOBOT Navigation Ready  "
                 "[AprilTag ID 0 · Ackermann 27.5×20.5 cm · "
                 "ESP32 TCP 192.168.0.51:5000]",
            relief=tk.SUNKEN, anchor=tk.W)
        self.status.grid(row=1, column=0, sticky="ew")

    # ─────────────────────────────────────────────────────────
    #  WHEEL MARGIN LIVE UPDATE
    # ─────────────────────────────────────────────────────────

    def _on_wheel_margin_change(self):
        hw, hh = self._robot_keepout_half_size_px()
        margin = float(self.wheel_margin_var.get())
        self.keepout_lbl.config(
            text=f"Keep-out box: {hw*2}×{hh*2} px  "
                 f"({ROBOT_WIDTH_CM+2*margin:.1f}×"
                 f"{ROBOT_LENGTH_CM+2*margin:.1f} cm)")
        self._heat_grid = None
        self._redraw()

    # ─────────────────────────────────────────────────────────
    #  SOURCE MANAGEMENT
    # ─────────────────────────────────────────────────────────

    def _stop_all(self):
        self.is_capturing = False
        self.is_video     = False
        self.is_image     = False
        self.auto_detect.set(False)
        self._stop_recording()
        if self.cap:
            self.cap.release()
            self.cap = None
        for btn, st in [(self.webcam_btn, tk.NORMAL),
                        (self.webcam_dis, tk.DISABLED),
                        (self.droid_btn,  tk.NORMAL),
                        (self.droid_dis,  tk.DISABLED)]:
            btn.config(state=st)

    def _load_image(self):
        self._stop_all()
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp"),("All","*.*")])
        if not path: return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error","Cannot read image"); return
        self.original_img = img
        self.display_img  = cv2.resize(img, (DISPLAY_W, DISPLAY_H))
        self.is_image     = True
        self._redraw()
        self.status.config(text=f"Loaded: {os.path.basename(path)}")

    def _load_video(self):
        self._stop_all()
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("All","*.*")])
        if not path: return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error","Cannot open video"); return
        self.cap = cap; self.is_capturing = True; self.is_video = True
        fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
        self._frame_delay = max(33, int(1000/min(fps, TARGET_FPS)))
        self.status.config(text=f"Playing: {os.path.basename(path)}")
        self._video_loop()

    def _connect_webcam(self):
        self._stop_all()
        try: idx = int(self.webcam_idx.get())
        except:
            messagebox.showerror("Error","Invalid webcam index"); return
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            messagebox.showerror("Error",f"Cannot open webcam {idx}"); return
        ret, frame = cap.read()
        if not ret:
            cap.release()
            messagebox.showerror("Error","Cannot read from webcam"); return
        self.cap = cap; self.is_capturing = True; self.is_video = False
        self.webcam_btn.config(state=tk.DISABLED)
        self.webcam_dis.config(state=tk.NORMAL)
        self.status.config(text=f"Webcam {idx} connected")
        self._live_loop()

    def _connect_droidcam(self):
        self._stop_all()
        ip   = self.droid_ip.get().strip()
        port = self.droid_port.get().strip()
        for url in [f"http://{ip}:{port}/video",
                    f"http://{ip}:{port}/mjpegfeed"]:
            cap = cv2.VideoCapture(url)
            ret, frame = cap.read()
            if ret and frame is not None:
                self.cap = cap; self.is_capturing = True; self.is_video = False
                self.droid_btn.config(state=tk.DISABLED)
                self.droid_dis.config(state=tk.NORMAL)
                self.status.config(text=f"IP Camera: {url}")
                self._live_loop(); return
            cap.release()
        messagebox.showerror("Error",f"Cannot connect to {ip}:{port}")

    def _disconnect(self):
        self._stop_all()
        self.status.config(text="Disconnected")

    def _live_loop(self):
        if not self.is_capturing or self.is_video or not self.cap: return
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.original_img = frame.copy()
            self.display_img  = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            self._process_frame()
        self.root.after(self._frame_delay, self._live_loop)

    def _video_loop(self):
        if not self.is_capturing or not self.is_video or not self.cap: return
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.original_img = frame.copy()
            self.display_img  = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            self._process_frame()
            self.root.after(self._frame_delay, self._video_loop)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.root.after(self._frame_delay, self._video_loop)

    # ─────────────────────────────────────────────────────────
    #  RECORDING
    # ─────────────────────────────────────────────────────────

    def _toggle_recording(self):
        if self._recording: self._stop_recording()
        else: self._start_recording()

    def _start_recording(self):
        if self.display_img is None:
            messagebox.showwarning("No feed","Connect a camera first"); return
        path = filedialog.asksaveasfilename(
            defaultextension=".avi",
            filetypes=[("AVI","*.avi"),("MP4","*.mp4")])
        if not path: return
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._recorder  = cv2.VideoWriter(path, fourcc, TARGET_FPS,
                                          (DISPLAY_W, DISPLAY_H))
        self._recording = True
        self.rec_btn.config(text="⏹  Stop Recording")
        self.rec_lbl.config(
            text=f"● REC: {os.path.basename(path)}", foreground="#cc0000")

    def _stop_recording(self):
        if self._recorder:
            self._recorder.release()
            self._recorder = None
        self._recording = False
        self.rec_btn.config(text="⏺  Start Recording")
        self.rec_lbl.config(text="Not recording", foreground="#888888")

    # ─────────────────────────────────────────────────────────
    #  FRAME PROCESSING
    # ─────────────────────────────────────────────────────────

    def _process_frame(self):
        self._frame_count += 1
        self._fps_counter += 1
        now = time.time()
        if now - self._fps_timer >= 1.0:
            self._fps_display = self._fps_counter / (now - self._fps_timer)
            self._fps_counter = 0
            self._fps_timer   = now
            self.fps_lbl.config(text=f"FPS: {self._fps_display:.1f}")

        self._decay_heat()

        if self.auto_detect.get():
            n = max(1, self.detect_n_var.get())
            if self._frame_count % n == 0:
                self._run_apriltag()
                self._submit_yolo_async()
                self._poll_yolo_result()
                self._merge_detections()
                self._post_detect_update()

        self._redraw()

    def _detect_once(self):
        if self.display_img is None:
            messagebox.showwarning("No source",
                "Connect a camera or load an image first"); return
        self._run_apriltag()
        if self.yolo_worker:
            conf = max(float(self.conf_scale.get()), YOLO_MIN_CONF)
            self.yolo_worker.post_frame(
                self._mask_robot_regions(self.display_img), conf)
            time.sleep(0.15)
            self._poll_yolo_result()
        self._merge_detections()
        self._post_detect_update()
        self._redraw()

    def _on_auto_detect_toggle(self):
        if self.auto_detect.get() and not (self.is_capturing or self.is_image):
            messagebox.showwarning("No feed",
                "Connect a camera or load an image first")
            self.auto_detect.set(False)

    def _on_class_ignore_change(self):
        self._heat_grid = None

    # ─────────────────────────────────────────────────────────
    #  DETECTION – AprilTag
    # ─────────────────────────────────────────────────────────

    def _run_apriltag(self):
        if self.display_img is None: return
        self.all_tags     = {}
        self.robot_boxes  = {}
        self.swarm1_pos   = None
        self.swarm1_theta = 0.0
        self.swarm1_bbox  = None
        if not self.at_detector: return

        gray = cv2.cvtColor(self.display_img, cv2.COLOR_BGR2GRAY)
        dets = self.at_detector.detect(gray)

        for d in dets:
            tid     = int(d.tag_id)
            corners = d.corners.astype(np.float32).copy()
            cx, cy  = float(d.center[0]), float(d.center[1])
            c0, c1  = corners[0], corners[1]
            theta   = float(np.degrees(
                np.arctan2(c1[1]-c0[1], c1[0]-c0[0]))) % 360.0
            self.all_tags[tid] = (cx, cy, theta, corners)

            edge_px = float(np.linalg.norm(c1 - c0))
            if edge_px > 5 and not self.manual_calib_enabled.get():
                cmp = TAG_SIZE_CM / edge_px
                self.cm_per_pixel   = cmp
                # Grid cell = half the robot's shorter dimension
                self.grid_size_px   = max(10,
                    int((min(ROBOT_WIDTH_CM, ROBOT_LENGTH_CM) / 2.0)
                        / self.cm_per_pixel))
                self.stop_thresh_px = STOP_THRESHOLD_CM / self.cm_per_pixel
                hw, hh = self._robot_keepout_half_size_px()
                margin = float(self.wheel_margin_var.get())
                self.calib_lbl.config(
                    text=f"cm/pixel: {self.cm_per_pixel:.4f}",
                    foreground="#007700")
                self.grid_lbl.config(
                    text=f"Grid cell: {self.grid_size_px} px")
                self.keepout_lbl.config(
                    text=f"Keep-out box: {hw*2}×{hh*2} px  "
                         f"({ROBOT_WIDTH_CM+2*margin:.1f}×"
                         f"{ROBOT_LENGTH_CM+2*margin:.1f} cm)")

            # AUTOBOT = ID 0
            if tid == 0:
                self.swarm1_pos   = (int(cx), int(cy))
                self.swarm1_theta = theta
                # Record bbox for masking / grid clear
                pad = max(20, int(0.60 * edge_px)) if edge_px > 0 else 20
                xs, ys = corners[:, 0], corners[:, 1]
                bx1 = int(max(0, min(DISPLAY_W-1, xs.min()-pad)))
                by1 = int(max(0, min(DISPLAY_H-1, ys.min()-pad)))
                bx2 = int(max(0, min(DISPLAY_W-1, xs.max()+pad)))
                by2 = int(max(0, min(DISPLAY_H-1, ys.max()+pad)))
                self.swarm1_bbox = (bx1, by1, bx2, by2)
                # Auto-set origin on first detection if not set
                if self.origin_px is None:
                    self.origin_px = (int(cx), int(cy))
                    self.origin_lbl.config(
                        text=f"🌐 Origin: {self.origin_px} (auto)",
                        foreground="#007700")
            else:
                # Other robots → rectangular obstacle exclusion zones
                pad = max(20, int(0.40 * edge_px)) if edge_px > 0 else 20
                xs, ys = corners[:, 0], corners[:, 1]
                bx1 = int(max(0, min(DISPLAY_W-1, xs.min()-pad)))
                by1 = int(max(0, min(DISPLAY_H-1, ys.min()-pad)))
                bx2 = int(max(0, min(DISPLAY_W-1, xs.max()+pad)))
                by2 = int(max(0, min(DISPLAY_H-1, ys.max()+pad)))
                self.robot_boxes[tid] = (bx1, by1, bx2, by2)

        # Update labels
        if self.swarm1_pos:
            self.swarm1_lbl.config(
                text=(f"🚗 AUTOBOT: {self.swarm1_pos}  θ={int(self.swarm1_theta)}°"),
                foreground="#007700")
            wcm = self._px_to_world_cm(*self.swarm1_pos)
            if wcm:
                self.swarm1_lbl.config(
                    text=(f"🚗 AUTOBOT: {self.swarm1_pos}  "
                          f"({wcm[0]:.1f},{wcm[1]:.1f})cm  "
                          f"θ={int(self.swarm1_theta)}°"))
        else:
            self.swarm1_lbl.config(
                text="🚗 AUTOBOT: Not detected", foreground="#ff0000")

        if self.origin_px:
            self.origin_lbl.config(
                text=f"🌐 Origin: {self.origin_px}", foreground="#007700")

    # ─────────────────────────────────────────────────────────
    #  DETECTION – YOLO (async)
    # ─────────────────────────────────────────────────────────

    def _submit_yolo_async(self):
        if not self.yolo_worker or self.display_img is None: return
        conf   = max(float(self.conf_scale.get()), YOLO_MIN_CONF)
        masked = self._mask_robot_regions(self.display_img)
        self.yolo_worker.post_frame(masked, conf)

    def _poll_yolo_result(self):
        if not self.yolo_worker: return
        result = self.yolo_worker.get_result()
        if result is not None:
            with self._yolo_lock:
                self._yolo_latest = result

    # ─────────────────────────────────────────────────────────
    #  MASKING – RECTANGULAR KEEP-OUT for AUTOBOT
    # ─────────────────────────────────────────────────────────

    def _mask_robot_regions(self, frame):
        if frame is None: return frame
        masked = frame.copy()

        # AUTOBOT: use its AABB keep-out zone
        aabb = self._axis_aligned_keepout_bbox()
        if aabb is not None:
            x1, y1, x2, y2 = aabb
            x1 = max(0, min(DISPLAY_W-1, x1))
            y1 = max(0, min(DISPLAY_H-1, y1))
            x2 = max(0, min(DISPLAY_W-1, x2))
            y2 = max(0, min(DISPLAY_H-1, y2))
            cv2.rectangle(masked, (x1, y1), (x2, y2), (0,0,0), -1)

        # Other robots: rectangular tag bbox
        for bx1, by1, bx2, by2 in self.robot_boxes.values():
            cv2.rectangle(masked, (bx1, by1), (bx2, by2), (0,0,0), -1)

        return masked

    # ─────────────────────────────────────────────────────────
    #  DETECTION – MERGE (rotated-rect overlap rejection)
    # ─────────────────────────────────────────────────────────

    def _merge_detections(self):
        with self._yolo_lock:
            raw = list(self._yolo_latest)

        self.obstacles = []
        ignore_set = {cid for cid, var in self._class_ignore_vars.items()
                      if var.get()}
        self._class_conf_accum = defaultdict(list)

        hw_ko, hh_ko = self._robot_keepout_half_size_px()

        for (bx1, by1, bx2, by2, cid, conf) in raw:
            if conf < YOLO_MIN_CONF: continue
            if cid in ignore_set: continue

            self._class_conf_accum[cid].append(conf)

            # Edge margin
            if (bx1 < EDGE_MARGIN_PX or by1 < EDGE_MARGIN_PX
                    or bx2 > DISPLAY_W - EDGE_MARGIN_PX
                    or by2 > DISPLAY_H - EDGE_MARGIN_PX):
                continue

            # Box shrink
            bx1s = bx1 + BOX_SHRINK_PX
            by1s = by1 + BOX_SHRINK_PX
            bx2s = bx2 - BOX_SHRINK_PX
            by2s = by2 - BOX_SHRINK_PX
            if bx2s <= bx1s or by2s <= by1s: continue

            # AUTOBOT rotated-rect keep-out check
            if self._box_intersects_keepout(bx1s, by1s, bx2s, by2s):
                if DEBUG_GRID:
                    print(f"[KO DROP] class={YOLO_CLASSES.get(cid,'?')} "
                          f"box=({bx1s},{by1s},{bx2s},{by2s})")
                continue

            # Other robot bbox overlap
            overlaps = False
            for rb in self.robot_boxes.values():
                rx1,ry1,rx2,ry2 = rb
                if not (bx2s < rx1 or bx1s > rx2 or by2s < ry1 or by1s > ry2):
                    overlaps = True; break
            if overlaps: continue

            self.obstacles.append((bx1s, by1s, bx2s, by2s, cid, conf))

        # Per-class confidence display
        lines = []
        for cid in YOLO_CLASSES:
            cfvals = self._class_conf_accum.get(cid, [])
            avg    = f"{np.mean(cfvals):.2f}" if cfvals else " ——"
            marker = "●" if cfvals else "○"
            lines.append(f"{marker} {YOLO_CLASSES[cid]:<12} {avg}")
        self.cls_conf_lbl.config(text="\n".join(lines))

    def _post_detect_update(self):
        self.det_info.config(
            text=f"Tags: {len(self.all_tags)}   Obstacles: {len(self.obstacles)}",
            foreground="#007700" if len(self.all_tags) > 0 else "#ff6600")

        if (self.swarm1_pos and self.target_pos
                and self.auto_replan.get() and self.path):
            self._calculate_path_silent()

        if self.auto_follow.get():
            self._follow_step()

        self._update_stats()

    # ─────────────────────────────────────────────────────────
    #  HEAT GRID
    # ─────────────────────────────────────────────────────────

    def _ensure_heat_grid(self):
        gs = self.grid_size_px
        rows = DISPLAY_H // gs
        cols = DISPLAY_W  // gs
        if self._heat_grid is None or self._heat_grid_size != gs:
            self._heat_grid      = np.zeros((rows, cols), dtype=np.int32)
            self._heat_grid_size = gs

    def _decay_heat(self):
        if self._heat_grid is not None:
            decay = max(1, self.heat_decay_var.get())
            np.maximum(self._heat_grid - decay, 0, out=self._heat_grid)

    # ─────────────────────────────────────────────────────────
    #  GRID BUILD
    # ─────────────────────────────────────────────────────────

    def _build_grid(self):
        gs   = self.grid_size_px
        rows = DISPLAY_H // gs
        cols = DISPLAY_W  // gs
        hthr = max(1, self.heat_thresh_var.get())
        infl = max(0, self.inflate_var.get()) if self.use_inflation.get() else 0

        self._ensure_heat_grid()
        heat = self._heat_grid

        for x1, y1, x2, y2, *_ in self.obstacles:
            if (y2-y1) > DISPLAY_H*0.60 or (x2-x1) > DISPLAY_W*0.60: continue
            r1 = max(0, min(rows-1,  int(y1) // gs))
            r2 = max(0, min(rows-1,  int(y2) // gs))
            c1 = max(0, min(cols-1,  int(x1) // gs))
            c2 = max(0, min(cols-1,  int(x2) // gs))
            heat[r1:r2+1, c1:c2+1] += 1

        np.minimum(heat, hthr + 10, out=heat)

        grid = np.zeros((rows, cols), dtype=np.uint8)
        grid[heat >= hthr] = 1

        # Direct stamp current frame obstacles
        for x1, y1, x2, y2, *_ in self.obstacles:
            if (y2-y1) > DISPLAY_H*0.60 or (x2-x1) > DISPLAY_W*0.60: continue
            r1 = max(0, min(rows-1, int(y1) // gs))
            r2 = max(0, min(rows-1, int(y2) // gs))
            c1 = max(0, min(cols-1, int(x1) // gs))
            c2 = max(0, min(cols-1, int(x2) // gs))
            grid[r1:r2+1, c1:c2+1] = 1

        # Inflation
        if infl > 0:
            try:
                from scipy.ndimage import binary_dilation
                struct = np.ones((2*infl+1, 2*infl+1), dtype=bool)
                grid   = binary_dilation(grid, structure=struct).astype(np.uint8)
            except ImportError:
                dilated = grid.copy()
                for dr in range(-infl, infl+1):
                    for dc in range(-infl, infl+1):
                        dilated |= np.roll(np.roll(grid, dr, 0), dc, 1)
                grid = dilated

        # Clear AUTOBOT keep-out AABB from grid
        aabb = self._axis_aligned_keepout_bbox()
        if aabb is not None:
            ax1, ay1, ax2, ay2 = aabb
            r1 = max(0, min(rows-1, int(ay1) // gs))
            r2 = max(0, min(rows-1, int(ay2) // gs))
            c1 = max(0, min(cols-1, int(ax1) // gs))
            c2 = max(0, min(cols-1, int(ax2) // gs))
            grid[r1:r2+1, c1:c2+1] = 0
            heat[r1:r2+1, c1:c2+1] = 0

        # Clear other robot boxes
        for x1, y1, x2, y2 in self.robot_boxes.values():
            r1 = max(0, min(rows-1, int(y1) // gs))
            r2 = max(0, min(rows-1, int(y2) // gs))
            c1 = max(0, min(cols-1, int(x1) // gs))
            c2 = max(0, min(cols-1, int(x2) // gs))
            grid[r1:r2+1, c1:c2+1] = 0
            heat[r1:r2+1, c1:c2+1] = 0

        return grid

    # ─────────────────────────────────────────────────────────
    #  COORDINATE HELPERS
    # ─────────────────────────────────────────────────────────

    def _px_to_grid(self, px, py):
        gs   = self.grid_size_px
        rows = DISPLAY_H // gs
        cols = DISPLAY_W  // gs
        return (max(0, min(rows-1, py // gs)),
                max(0, min(cols-1, px // gs)))

    def _grid_to_px(self, r, c):
        gs = self.grid_size_px
        return c*gs + gs//2, r*gs + gs//2

    # ─────────────────────────────────────────────────────────
    #  PATH PLANNING
    # ─────────────────────────────────────────────────────────

    def _calculate_path_silent(self):
        if self.swarm1_pos is None or self.target_pos is None: return
        self.grid = self._build_grid()
        sg = self._px_to_grid(*self.swarm1_pos)
        eg = self._px_to_grid(*self.target_pos)
        if self.grid[eg[0], eg[1]] == 1: return
        self.grid[sg[0], sg[1]] = 0
        self.grid[eg[0], eg[1]] = 0
        path_g = AStarPlanner(self.grid).plan(sg, eg)
        if path_g:
            pts = [self._grid_to_px(r, c) for r, c in path_g]
            if self.use_rdp.get():
                pts = rdp_simplify(pts, RDP_EPSILON)
            self.path = pts

    def _calculate_path(self):
        if self.swarm1_pos is None:
            messagebox.showwarning("No robot",
                "AUTOBOT not detected – point camera at AprilTag ID 0"); return
        if self.target_pos is None:
            messagebox.showwarning("No target",
                "Click on the arena or enter coordinates to set a target"); return

        self.grid = self._build_grid()
        sg = self._px_to_grid(*self.swarm1_pos)
        eg = self._px_to_grid(*self.target_pos)

        if self.grid[eg[0], eg[1]] == 1:
            messagebox.showerror("Invalid","Target is inside an obstacle!"); return

        self.grid[sg[0], sg[1]] = 0
        self.grid[eg[0], eg[1]] = 0
        path_g = AStarPlanner(self.grid).plan(sg, eg)

        if path_g is None:
            messagebox.showerror("No path","A* could not find a path!"); return

        pts = [self._grid_to_px(r, c) for r, c in path_g]
        if self.use_rdp.get():
            pts = rdp_simplify(pts, RDP_EPSILON)
        self.path = pts

        length = sum(
            math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
            for i in range(len(pts)-1))
        label = (f"✓ {len(pts)} wpts  {length:.0f}px"
                 f"  ≈{length*self.cm_per_pixel:.1f}cm"
                 if self.cm_per_pixel else
                 f"✓ {len(pts)} waypoints  {length:.0f}px")
        self.path_info.config(text=label, foreground="#007700")
        self._redraw()
        self._update_stats()

    # ─────────────────────────────────────────────────────────
    #  AUTO FOLLOW  (path waypoints → send cm commands)
    # ─────────────────────────────────────────────────────────

    def _follow_step(self):
        if not self.robot.connected: return
        if not self.path or len(self.path) < 2: return
        if self.swarm1_pos is None: return

        wp   = self.path[1]
        dx   = wp[0] - self.swarm1_pos[0]
        dy   = wp[1] - self.swarm1_pos[1]
        dist = math.hypot(dx, dy)

        stop_px = (self.stop_thresh_px
                   if self.cm_per_pixel else self.grid_size_px * 0.8)

        if dist < self.grid_size_px * 0.7:
            self.path.pop(0)
            if len(self.path) < 2:
                self.auto_follow.set(False)
                self.path_info.config(text="✓ Target reached!", foreground="#007700")
            return

        if dist < stop_px:
            self.auto_follow.set(False)
            self.path_info.config(
                text="✓ Within stop threshold", foreground="#007700")
            return

        # Convert waypoint to world cm and send
        wcm = self._px_to_world_cm(*wp)
        if wcm:
            x_cm, y_cm = int(round(wcm[0])), int(round(wcm[1]))
            self.robot.send_absolute_async(x_cm, y_cm)
            self.follow_info.config(
                text=f"→ ({x_cm},{y_cm})cm  dist={dist:.0f}px",
                foreground="#007700")
            self._update_world_pos_label()

    # ─────────────────────────────────────────────────────────
    #  COORDINATE ENTRY FEATURE
    # ─────────────────────────────────────────────────────────

    def _parse_coord_entry(self, show_error=True):
        """Parse X/Y entries, return (x_cm, y_cm) ints or None on error."""
        try:
            x_cm = int(float(self.coord_x_var.get()))
            y_cm = int(float(self.coord_y_var.get()))
        except (ValueError, tk.TclError):
            if show_error:
                messagebox.showerror("Invalid Input",
                    "X and Y must be numeric values in cm.\n"
                    "Examples: 0, 30, -50, 120")
            return None
        if abs(x_cm) > 999 or abs(y_cm) > 999:
            messagebox.showerror("Out of range",
                "Coordinates must be within ±999 cm per axis.")
            return None
        return x_cm, y_cm

    def _resolve_coord_entry(self):
        """Return (entered_x, entered_y, abs_x, abs_y, mode) or None."""
        coords = self._parse_coord_entry()
        if coords is None:
            return None
        x_cm, y_cm = coords
        mode = self.coord_mode_var.get()
        if mode == "relative":
            abs_x = int(self.robot.world_x + x_cm)
            abs_y = int(self.robot.world_y + y_cm)
        else:
            abs_x, abs_y = x_cm, y_cm
        abs_x = max(-999, min(999, abs_x))
        abs_y = max(-999, min(999, abs_y))
        return x_cm, y_cm, abs_x, abs_y, mode

    def _update_coord_preview(self):
        if not hasattr(self, "coord_preview_lbl"):
            return
        coords = self._parse_coord_entry(show_error=False)
        if coords is None:
            self.coord_preview_lbl.config(text="")
            return

        x_cm, y_cm = coords
        mode = self.coord_mode_var.get()
        if mode == "relative":
            abs_x = int(self.robot.world_x + x_cm)
            abs_y = int(self.robot.world_y + y_cm)
            preview = (f"Relative input: Δ({x_cm}, {y_cm}) cm  →  "
                       f"absolute ({abs_x}, {abs_y}) cm")
        else:
            abs_x, abs_y = x_cm, y_cm
            preview = f"Absolute input: ({abs_x}, {abs_y}) cm"

        if self.cm_per_pixel is not None and self.origin_px is not None:
            px = self._world_cm_to_px(abs_x, abs_y)
            if px is not None:
                preview += f"  →  {px[0]} px, {px[1]} px"
        self.coord_preview_lbl.config(text=preview)

    def _on_coord_mode_change(self):
        self._update_coord_preview()

    def _goto_from_entry_set_target(self):
        """Convert entered cm coordinates to a canvas target only."""
        resolved = self._resolve_coord_entry()
        if resolved is None:
            return
        x_cm, y_cm, abs_x, abs_y, mode = resolved

        if self.cm_per_pixel is None:
            messagebox.showwarning("No calibration",
                "AprilTag not detected yet — calibration needed to convert cm→px.\n"
                "Point camera at the arena with AprilTag ID 0 visible,\n"
                "or use Manual Calibration override.")
            return

        if self.origin_px is None:
            messagebox.showwarning("No origin",
                "World origin not set.\n"
                "Detect the robot first or press 'Set World Origin at Robot Now'.")
            return

        px = self._world_cm_to_px(abs_x, abs_y)
        if px is None:
            return

        self.target_pos       = px
        self.target_world_cm  = (abs_x, abs_y)
        self.target_lbl.config(
            text=f"🎯 Target: ({abs_x},{abs_y}) cm  →  {px} px",
            foreground="#007700")
        self.path = None
        self.path_info.config(text="Target set from coordinates – calculate path")
        if mode == "relative":
            self.coord_status_lbl.config(
                text=f"Target set: Δ({x_cm}, {y_cm}) cm → abs ({abs_x}, {abs_y}) cm",
                foreground="#007700")
        else:
            self.coord_status_lbl.config(
                text=f"Target set: ({abs_x}, {abs_y}) cm", foreground="#007700")
        self._redraw()
        self._update_stats()

    def _goto_from_entry_send(self):
        """Parse entries and send coordinate command to AUTOBOT immediately."""
        resolved = self._resolve_coord_entry()
        if resolved is None:
            return
        x_cm, y_cm, abs_x, abs_y, mode = resolved

        if not self.robot.connected:
            messagebox.showwarning("Not connected",
                "Connect to ESP32 first via the Robot Connection section.")
            return

        status, ms = self.robot.send_absolute(abs_x, abs_y)
        if status == "OK":
            if mode == "relative":
                msg = f"✓ Sent Δ({x_cm},{y_cm})cm → ({abs_x},{abs_y})cm  [{ms:.1f}ms]"
            else:
                msg = f"✓ Sent ({abs_x},{abs_y})cm  [{ms:.1f}ms]"
            self.coord_status_lbl.config(text=msg, foreground="#007700")
            self.status.config(text=f"✓ Command sent to AUTOBOT: ({abs_x},{abs_y}) cm")
            self._update_world_pos_label()
            self._goto_from_entry_set_target()
        else:
            self.coord_status_lbl.config(
                text=f"✗ {status}", foreground="#cc0000")
            messagebox.showerror("Send Error",
                f"Failed to send command:\n{status}\n\n"
                "Check ESP32 connection.")

    def _preset_goto(self, x_cm, y_cm):
        """Quick preset buttons – fill entries and send or preview."""
        self.coord_x_var.set(str(x_cm))
        self.coord_y_var.set(str(y_cm))
        if self.robot.connected:
            self._goto_from_entry_send()
        else:
            self._goto_from_entry_set_target()

    def _update_world_pos_label(self):
        wx, wy = self.robot.world_x, self.robot.world_y
        self.world_pos_lbl.config(
            text=f"STM32 world pos: ({wx}, {wy}) cm")
        if self.robot.last_cmd:
            self.last_cmd_lbl.config(
                text=f"Last cmd: {self.robot.last_cmd}")
        self._update_coord_preview()

    # ─────────────────────────────────────────────────────────
    #  RESET STM32 POSITION
    # ─────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────
    #  RESET STM32 POSITION
    # ─────────────────────────────────────────────────────────

    def _reset_stm32_position(self):
        """
        Sends (0,0) to the STM32, which resets its internal worldX/worldY.
        Also resets the GUI mirror.
        """
        self.robot.reset_origin()
        if self.robot.connected:
            status, ms = self.robot.send_absolute(0, 0)
            msg = f"✓ Reset sent [{ms:.1f}ms]" if status == "OK" else f"Reset: {status}"
            self.status.config(text=msg)
        self._update_world_pos_label()
        # Re-pin origin to current robot position
        self._set_origin_at_robot()

    # ─────────────────────────────────────────────────────────
    #  DRAWING
    # ─────────────────────────────────────────────────────────

    def _redraw(self):
        if self.display_img is None: return
        img = self.display_img.copy()

        # Grid / heat overlay
        if self.show_grid.get():
            gs  = self.grid_size_px
            h, w = img.shape[:2]
            ov  = img.copy()
            for x in range(0, w, gs): cv2.line(ov,(x,0),(x,h),(60,60,60),1)
            for y in range(0, h, gs): cv2.line(ov,(0,y),(w,y),(60,60,60),1)
            if (self.show_heat.get()
                    and self._heat_grid is not None
                    and self._heat_grid_size == gs):
                hthr = max(1, self.heat_thresh_var.get())
                rows, cols = self._heat_grid.shape
                for r in range(rows):
                    for c in range(cols):
                        v = self._heat_grid[r, c]
                        if v > 0:
                            x1, y1 = c*gs, r*gs
                            alpha  = min(v / hthr, 1.0)
                            color  = (0, 60, 255) if v < hthr else (0, 0, 220)
                            sub    = ov[y1:y1+gs, x1:x1+gs]
                            sub[:] = (sub*(1-alpha*0.55)
                                      + np.array(color)*(alpha*0.55)).astype(np.uint8)
            img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)

        # YOLO detections
        if self.show_detections.get():
            for x1,y1,x2,y2,cid,conf in self.obstacles:
                col   = CLASS_COLORS.get(cid,(0,0,220))
                name  = YOLO_CLASSES.get(cid,f"cls{cid}")
                label = f"{name} {conf:.2f}"
                cv2.rectangle(img,(x1,y1),(x2,y2),col,2)
                (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
                lx,ly = x1, max(0,y1-4)
                cv2.rectangle(img,(lx,ly-th-4),(lx+tw+4,ly),col,-1)
                cv2.putText(img,label,(lx+2,ly-2),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

            # AprilTag overlays – all detected tags
            for tid,(cx,cy,theta,corners) in self.all_tags.items():
                col  = ROBOT_COLORS.get(tid,(200,200,200))
                name = ROBOT_NAMES.get(tid,f"ID{tid}")
                icx,icy = int(cx),int(cy)
                cv2.polylines(img,[corners.astype(int)],True,col,2)
                cv2.circle(img,(icx,icy),22,col,2)
                cv2.drawMarker(img,(icx,icy),col,cv2.MARKER_DIAMOND,16,2)
                r  = math.radians(theta)
                ex = int(icx+35*math.cos(r))
                ey = int(icy+35*math.sin(r))
                cv2.arrowedLine(img,(icx,icy),(ex,ey),col,2,tipLength=0.4)
                cv2.putText(img,name,(icx+28,icy-12),
                            cv2.FONT_HERSHEY_SIMPLEX,0.65,col,2)
                cv2.putText(img,f"({icx},{icy}) θ={int(theta)}°",
                            (icx+28,icy+8),cv2.FONT_HERSHEY_SIMPLEX,0.40,col,1)

            # Other robot bboxes
            for tid,(bx1,by1,bx2,by2) in self.robot_boxes.items():
                col = ROBOT_COLORS.get(tid,(200,200,200))
                cv2.rectangle(img,(bx1,by1),(bx2,by2),col,1)

        # ── AUTOBOT keep-out ROTATED BOX visualisation ─────────
        if self.show_keepout.get() and self.swarm1_pos is not None:
            corners_ko = self._keepout_corners_px()
            if corners_ko:
                pts = np.array(corners_ko, dtype=np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 100))
                img = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
                cv2.polylines(img, [pts], True, (0, 255, 100), 2)
                hw, hh = self._robot_keepout_half_size_px()
                cx_ko, cy_ko = self.swarm1_pos
                cv2.putText(img,
                    f"keep-out {hw*2}×{hh*2}px",
                    (cx_ko + hw + 4, cy_ko),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 255, 100), 1)
                # Body rectangle (slightly smaller)
                if self.cm_per_pixel:
                    hw_body = max(4, int((ROBOT_WIDTH_CM/2.0)/self.cm_per_pixel))
                    hh_body = max(4, int((ROBOT_LENGTH_CM/2.0)/self.cm_per_pixel))
                    margin  = float(self.wheel_margin_var.get())
                    body_local = [(-hw_body,-hh_body),(hw_body,-hh_body),
                                  (hw_body,hh_body),(-hw_body,hh_body)]
                    theta_r = math.radians(self.swarm1_theta)
                    cos_t, sin_t = math.cos(theta_r), math.sin(theta_r)
                    body_world = []
                    for lx,ly in body_local:
                        wx = int(cx_ko + lx*cos_t - ly*sin_t)
                        wy = int(cy_ko + lx*sin_t + ly*cos_t)
                        body_world.append((wx,wy))
                    cv2.polylines(img,
                        [np.array(body_world,dtype=np.int32)],
                        True,(0,180,80),1)

        # World origin marker
        if self.origin_px is not None:
            ox, oy = self.origin_px
            cv2.drawMarker(img,(ox,oy),(255,255,0),
                           cv2.MARKER_STAR,20,2)
            cv2.putText(img,"ORIGIN",(ox+10,oy-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,0),1)

        # Target marker
        if self.target_pos:
            tx,ty = self.target_pos
            cv2.drawMarker(img,(tx,ty),(0,255,0),cv2.MARKER_CROSS,32,3)
            cv2.circle(img,(tx,ty),22,(0,255,255),3)
            if self.target_world_cm:
                label = f"TARGET ({self.target_world_cm[0]},{self.target_world_cm[1]})cm"
            else:
                label = "TARGET"
            cv2.putText(img,label,(tx+26,ty-26),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),2)

        # Path overlay
        if self.show_path.get() and self.path and len(self.path) > 1:
            for i in range(len(self.path)-1):
                cv2.line(img,self.path[i],self.path[i+1],(0,220,255),2)
            for i,pt in enumerate(self.path):
                if   i == 0:               cv2.circle(img,pt,8,(0,255,0),-1)
                elif i == len(self.path)-1: cv2.circle(img,pt,8,(0,0,255),-1)
                else:                       cv2.circle(img,pt,5,(255,220,0),-1)
            # Draw cm labels on first few waypoints
            if self.cm_per_pixel and self.origin_px:
                for i, pt in enumerate(self.path[:6]):
                    wcm = self._px_to_world_cm(*pt)
                    if wcm:
                        cv2.putText(img,f"({wcm[0]:.0f},{wcm[1]:.0f})",
                            (pt[0]+6,pt[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.32,(200,255,200),1)

        # HUD
        if self.is_capturing and self._fps_display > 0:
            cv2.putText(img,f"{self._fps_display:.1f} FPS",
                (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,220,0),2)
        if self._recording:
            cv2.circle(img,(DISPLAY_W-20,18),8,(0,0,220),-1)
            cv2.putText(img,"REC",(DISPLAY_W-50,24),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,220),1)
        if self.cm_per_pixel:
            hw,hh = self._robot_keepout_half_size_px()
            cv2.putText(img,
                f"Scale:{self.cm_per_pixel:.4f}cm/px  "
                f"Grid:{self.grid_size_px}px  "
                f"KeepOut:{hw*2}×{hh*2}px  "
                f"Heat≥{self.heat_thresh_var.get()}",
                (10,DISPLAY_H-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(160,160,0),1)
        # STM32 world position HUD
        cv2.putText(img,
            f"STM32 pos: ({self.robot.world_x},{self.robot.world_y})cm",
            (10,DISPLAY_H-28),
            cv2.FONT_HERSHEY_SIMPLEX,0.38,(100,200,255),1)

        if self._recording and self._recorder:
            self._recorder.write(img)

        # Scale to actual canvas size
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw > 10 and ch > 10 and (cw != DISPLAY_W or ch != DISPLAY_H):
            img = cv2.resize(img,(cw,ch),interpolation=cv2.INTER_LINEAR)

        rgb   = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.create_image(0,0,anchor=tk.NW,image=imgtk)
        self.canvas.image = imgtk

    # ─────────────────────────────────────────────────────────
    #  TARGET (canvas click)
    # ─────────────────────────────────────────────────────────

    def _on_canvas_click(self, event):
        if self.display_img is None and not self.is_image: return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        cx = int(event.x * DISPLAY_W / cw) if cw > 10 else event.x
        cy = int(event.y * DISPLAY_H / ch) if ch > 10 else event.y
        self.target_pos = (cx, cy)

        # Also compute world cm if possible
        wcm = self._px_to_world_cm(cx, cy)
        self.target_world_cm = wcm
        if wcm:
            self.target_lbl.config(
                text=f"🎯 Target: ({cx},{cy})px  ({wcm[0]:.1f},{wcm[1]:.1f})cm",
                foreground="#007700")
            if self.coord_mode_var.get() == "relative":
                self.coord_x_var.set(str(int(round(wcm[0] - self.robot.world_x))))
                self.coord_y_var.set(str(int(round(wcm[1] - self.robot.world_y))))
            else:
                self.coord_x_var.set(str(int(round(wcm[0]))))
                self.coord_y_var.set(str(int(round(wcm[1]))))
        else:
            self.target_lbl.config(
                text=f"🎯 Target: ({cx},{cy})px", foreground="#007700")

        self.path = None
        self.path_info.config(text="Target set – calculate path")
        self._redraw()
        self._update_stats()

    def _clear_target(self):
        self.target_pos      = None
        self.target_world_cm = None
        self.path            = None
        self.target_lbl.config(text="🎯 Target: Not set", foreground="#888888")
        self.path_info.config(text="No path")
        self.coord_status_lbl.config(text="")
        self._redraw()
        self._update_stats()

    # ─────────────────────────────────────────────────────────
    #  CALIBRATION OVERRIDE
    # ─────────────────────────────────────────────────────────

    def _on_manual_calib_toggle(self):
        if self.manual_calib_enabled.get():
            self.manual_cmpx_entry.config(state=tk.NORMAL)
            self._apply_manual_calib()
        else:
            self.manual_cmpx_entry.config(state=tk.DISABLED)

    def _apply_manual_calib(self):
        try:
            val = float(self.manual_cmpx_var.get())
            if val <= 0: raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Error","Enter a positive cm/pixel value"); return
        self.cm_per_pixel   = val
        self.grid_size_px   = max(10,
            int((min(ROBOT_WIDTH_CM, ROBOT_LENGTH_CM) / 2.0) / val))
        self.stop_thresh_px = STOP_THRESHOLD_CM / val
        self._heat_grid     = None
        hw, hh = self._robot_keepout_half_size_px()
        margin = float(self.wheel_margin_var.get())
        self.calib_lbl.config(
            text=f"cm/pixel: {val:.4f} (manual)", foreground="#e06000")
        self.grid_lbl.config(text=f"Grid cell: {self.grid_size_px} px")
        self.keepout_lbl.config(
            text=f"Keep-out box: {hw*2}×{hh*2} px  "
                 f"({ROBOT_WIDTH_CM+2*margin:.1f}×"
                 f"{ROBOT_LENGTH_CM+2*margin:.1f} cm)")

    # ─────────────────────────────────────────────────────────
    #  ROBOT CONNECTION
    # ─────────────────────────────────────────────────────────

    def _connect_robot(self):
        ip = self.robot_ip.get().strip()
        try: port = int(self.robot_port_e.get().strip())
        except:
            messagebox.showerror("Error","Invalid port"); return
        self.robot.ip   = ip
        self.robot.port = port
        self.status.config(text=f"Connecting to ESP32 {ip}:{port}…")
        self.root.update()
        ok, err = self.robot.connect()
        if ok:
            self.rob_conn_btn.config(state=tk.DISABLED)
            self.rob_dis_btn.config(state=tk.NORMAL)
            self.rob_status.config(
                text=f"✓ {ip}:{port}  (ESP32 TCP)", foreground="#007700")
            self.status.config(
                text=f"✓ Connected to AUTOBOT ESP32 at {ip}:{port}")
        else:
            messagebox.showerror(
                "Connect Error",
                f"Cannot connect to ESP32 at {ip}:{port}\n\n{err}\n\n"
                "Check:\n1. IP correct (192.168.0.51)\n"
                "2. Connected to same WiFi network\n"
                "3. ESP32 powered on and running")
            self.rob_status.config(text="Not connected", foreground="#aa0000")

    def _disconnect_robot(self):
        self.robot.disconnect()
        self.rob_conn_btn.config(state=tk.NORMAL)
        self.rob_dis_btn.config(state=tk.DISABLED)
        self.rob_status.config(text="Not connected", foreground="#aa0000")
        self.status.config(text="Robot disconnected")

    # ─────────────────────────────────────────────────────────
    #  STATS
    # ─────────────────────────────────────────────────────────

    def _update_stats(self):
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)

        lines = ["=== APRILTAGS ===",
                 f"Detected: {len(self.all_tags)}"]
        for tid,(cx,cy,theta,_) in self.all_tags.items():
            lines.append(f"  ID{tid} {ROBOT_NAMES.get(tid,'?')}: "
                         f"({int(cx)},{int(cy)}) θ={int(theta)}°")

        hw,hh = self._robot_keepout_half_size_px()
        margin = float(self.wheel_margin_var.get())
        lines += ["","=== AUTOBOT KEEP-OUT BOX ===",
                  f"Body W×L     : {ROBOT_WIDTH_CM}×{ROBOT_LENGTH_CM} cm",
                  f"Wheel margin : {margin:.1f} cm",
                  f"Total W×L    : {ROBOT_WIDTH_CM+2*margin:.1f}×{ROBOT_LENGTH_CM+2*margin:.1f} cm",
                  f"In pixels    : {hw*2}×{hh*2} px",
                  f"Robot pos    : {self.swarm1_pos or 'not detected'}",
                  f"Robot θ      : {int(self.swarm1_theta)}°"]

        lines += ["","=== CALIBRATION ===",
                  f"cm/pixel : {f'{self.cm_per_pixel:.4f}' if self.cm_per_pixel else 'pending'}",
                  f"Manual   : {'yes' if self.manual_calib_enabled.get() else 'no'}",
                  f"Grid cell: {self.grid_size_px} px",
                  f"Stop thr : {self.stop_thresh_px:.1f} px"]

        lines += ["","=== WORLD COORDS ===",
                  f"Origin px  : {self.origin_px}"]
        if self.swarm1_pos and self.origin_px:
            wcm = self._px_to_world_cm(*self.swarm1_pos)
            lines.append(f"Robot (cm) : {wcm}")
        lines += [f"Target cm  : {self.target_world_cm}",
                  f"STM32 pos  : ({self.robot.world_x},{self.robot.world_y}) cm"]

        lines += ["","=== OBSTACLES ===",
                  f"Total: {len(self.obstacles)}"]
        class_counts = defaultdict(int)
        for *_,cid,_ in self.obstacles:
            class_counts[YOLO_CLASSES.get(cid,'?')] += 1
        for cname,cnt in sorted(class_counts.items()):
            lines.append(f"  {cname}: {cnt}")

        lines += ["","=== PATH ==="]
        if self.path:
            length = sum(
                math.hypot(self.path[i+1][0]-self.path[i][0],
                           self.path[i+1][1]-self.path[i][1])
                for i in range(len(self.path)-1))
            lines += [f"Waypoints: {len(self.path)}",
                      f"Length: {length:.0f}px"]
            if self.cm_per_pixel:
                lines.append(f"  = {length*self.cm_per_pixel:.1f} cm")
        else:
            lines.append("No path")

        lines += ["","=== ROBOT ===",
                  f"Connected: {self.robot.connected}",
                  f"ESP32 IP : {self.robot.ip}:{self.robot.port}",
                  f"Last cmd : {self.robot.last_cmd or '–'}"]

        self.stats_text.insert(1.0,"\n".join(lines))
        self.stats_text.config(state=tk.DISABLED)

    # ─────────────────────────────────────────────────────────
    #  EXPORT
    # ─────────────────────────────────────────────────────────

    def _save_screenshot(self):
        if self.display_img is None:
            messagebox.showwarning("No image","Nothing to save"); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("All","*.*")])
        if not path: return
        try:
            cv2.imwrite(path, self.display_img)
            self.status.config(text="✓ Screenshot saved")
        except Exception as e:
            messagebox.showerror("Error",str(e))

    def _export_path(self):
        if not self.path:
            messagebox.showwarning("No path","Calculate path first"); return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("Text","*.txt"),("All","*.*")])
        if not path: return
        try:
            with open(path,"w") as f:
                f.write("# AUTOBOT path export\n")
                f.write(f"# Robot pos     : {self.swarm1_pos}\n")
                f.write(f"# Robot heading : {self.swarm1_theta:.1f}°\n")
                f.write(f"# Target        : {self.target_pos}\n")
                f.write(f"# Target cm     : {self.target_world_cm}\n")
                f.write(f"# Origin px     : {self.origin_px}\n")
                f.write(f"# cm/pixel      : {self.cm_per_pixel}\n")
                f.write(f"# Keep-out      : {self._robot_keepout_half_size_px()}\n")
                f.write("index,x_px,y_px,x_cm,y_cm\n")
                for i,(x,y) in enumerate(self.path):
                    wcm = self._px_to_world_cm(x,y)
                    xcm = f"{wcm[0]:.2f}" if wcm else ""
                    ycm = f"{wcm[1]:.2f}" if wcm else ""
                    f.write(f"{i},{x},{y},{xcm},{ycm}\n")
            self.status.config(text="✓ Path exported")
        except Exception as e:
            messagebox.showerror("Error",str(e))

    # ─────────────────────────────────────────────────────────
    #  CLEANUP
    # ─────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_recording()
        if self.yolo_worker:
            self.yolo_worker.stop()
        if self.cap:
            self.cap.release()
        self.robot.disconnect()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = Swarm1NavigationGUI(root)
    root.mainloop()
