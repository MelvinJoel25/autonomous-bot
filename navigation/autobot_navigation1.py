"""
Autobot_navigation_v3.py  –  Production Release
═══════════════════════════════════════════════════════════════════════════════
Navigation GUI for the Omniwheel Autobot (Arduino R4 WiFi)

New in v3  (over v2-fixed)
───────────────────────────
  • CLASS-AWARE detection  – each of your 6 trained classes shown in a distinct
    colour with its name on the bounding box.
  • IGNORE-CLASS filter    – "table" excluded from obstacle grid by default
    (robot sits on the table).  Toggleable per-class in the UI.
  • ASYNC YOLO             – YOLO runs in a background thread; the GUI never
    blocks waiting for inference.
  • OBSTACLE INFLATION     – configurable safety-margin dilation of grid cells
    so A* routes stay clear of walls.
  • DIAGONAL COST A*       – √2 weight on diagonal moves; path is geometrically
    more accurate.
  • HEADING-AWARE follow   – robot heading (from AprilTag) used to send the
    correct strafe command (not just dx/dy).
  • SPEED TIERS            – Slow / Normal / Fast teleop speed sent to Arduino.
  • VIDEO RECORDING        – one-click record of the arena feed to .avi.
  • MANUAL CALIBRATION     – override cm/pixel with a slider if no AprilTag
    is visible.
  • HEAT-MAP CONTROLS      – threshold and decay tunable live from the UI.
  • SMOOTH PATH (RDP)      – Ramer-Douglas-Peucker path simplification so the
    robot follows fewer, longer waypoints.
  • PER-CLASS CONF DISPLAY – live mini-table of class → average confidence.
  • FULL KEYBOARD MAP      – all keys shown; SPACE emergency-stop.
  • All FIX-1 … FIX-9 from v2 retained unchanged.

AprilTag IDs
─────────────
  0 = AUTOBOT   (magenta)   ← main robot
  1 = SWARM-1   (green)
  2 = SWARM-2   (orange)
  3 = SWARM-3   (cyan)

Arduino HTTP commands  GET /cmd?v=X
────────────────────────────────────
  F  B  L  R  A(FL)  D(FR)  Q(BL)  E(BR)  S(stop)
  Speed prefix: 1=slow  2=normal  3=fast   e.g. "2F"

YOLO classes (your trained model)
───────────────────────────────────
  0=box  1=sticky_note  2=bottle  3=plastic  4=table  5=tape
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
import requests
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
#  CONFIGURATION  (edit these; nothing else needs touching)
# ══════════════════════════════════════════════════════════════

TAG_SIZE_CM        = 12.0          # physical side length of AprilTag
ROBOT_SIZE_CM      = 40.0          # robot footprint (sets grid cell size)
STOP_THRESHOLD_CM  = 8.0           # auto-follow: stop within this distance

YOLO_MODEL_PATH    = r"C:\Users\meljo\krab-3\runs\obstacle_v1\weights\best.pt"
YOLO_CONF          = 0.45          # default confidence threshold
YOLO_MIN_CONF      = 0.40          # hard floor – boxes below always ignored
YOLO_IOU           = 0.45

# ── Your 6 trained classes ──────────────────────────────────
YOLO_CLASSES = {
    0: "box",
    1: "sticky_note",
    2: "bottle",
    3: "plastic",
    4: "table",
    5: "tape",
}
# Classes that are NOT treated as obstacles (robot is on the table, etc.)
DEFAULT_IGNORE_CLASSES = {4}   # table

# Per-class BGR colours for bounding-box drawing
CLASS_COLORS = {
    0: (0,  100, 255),   # box         – orange-red
    1: (0,  220, 220),   # sticky_note – yellow
    2: (255, 60,  60),   # bottle      – blue
    3: (0,  200,  60),   # plastic     – green
    4: (130,130, 130),   # table       – grey
    5: (200, 0,  200),   # tape        – magenta
}

DISPLAY_W          = 1280
DISPLAY_H          = 860

DEFAULT_ROBOT_IP   = "192.168.4.1"
ROBOT_PORT         = 80

ROBOT_COLORS = {0:(255,0,255), 1:(0,255,0), 2:(0,165,255), 3:(255,255,0)}
ROBOT_NAMES  = {0:"AUTOBOT",  1:"SWARM-1", 2:"SWARM-2",   3:"SWARM-3"}

TARGET_FPS         = 30
DETECT_EVERY_N     = 2

LEFT_PANEL_W       = 350

# ── Grid / detection stability ──────────────────────────────
EDGE_MARGIN_PX     = 30
HEAT_THRESHOLD     = 3
HEAT_DECAY         = 1
BOX_SHRINK_PX      = 4
OBSTACLE_INFLATE   = 1    # grid cells to inflate obstacles for safety margin

# Path simplification tolerance (pixels); 0 = disabled
RDP_EPSILON        = 8.0

DEBUG_GRID         = False


# ══════════════════════════════════════════════════════════════
#  A*  PLANNER  (diagonal cost = √2)
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
        f     = {start: self._heuristic(start, goal)}
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
                    f[nb]    = tg + self._heuristic(nb, goal)
                    heapq.heappush(heap, (f[nb], nb))
        return None


# ══════════════════════════════════════════════════════════════
#  PATH UTILITIES
# ══════════════════════════════════════════════════════════════

def rdp_simplify(points, epsilon):
    """Ramer-Douglas-Peucker path simplification."""
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
#  ARDUINO HTTP CLIENT
# ══════════════════════════════════════════════════════════════

class OmniRobotClient:
    SPEED_PREFIX = {1: "1", 2: "2", 3: "3"}

    def __init__(self, ip=DEFAULT_ROBOT_IP):
        self.ip        = ip
        self.port      = ROBOT_PORT
        self.connected = False
        self._lock     = threading.Lock()
        self.last_cmd  = None
        self.speed     = 2      # 1=slow 2=normal 3=fast

    def set_ip(self, ip):
        self.ip = ip
        if self.connected:
            self.disconnect()

    def connect(self):
        try:
            r = requests.get(f"http://{self.ip}:{self.port}/cmd?v=S", timeout=2)
            if r.status_code == 200:
                self.connected = True
                return True, ""
            return False, f"Status {r.status_code}"
        except Exception as e:
            return False, str(e)

    def disconnect(self):
        with self._lock:
            self.connected = False

    def send_cmd(self, cmd: str):
        if not self.connected:
            return "NOT_CONNECTED", 0
        full = f"{self.SPEED_PREFIX.get(self.speed,'2')}{cmd}" if cmd != "S" else "S"
        with self._lock:
            try:
                t = time.time()
                r = requests.get(
                    f"http://{self.ip}:{self.port}/cmd?v={full}", timeout=2)
                ms = (time.time()-t)*1000
                self.last_cmd = full
                return ("OK" if r.status_code == 200
                        else f"ERR_{r.status_code}"), ms
            except requests.exceptions.Timeout:
                return "TIMEOUT", (time.time()-t)*1000
            except Exception as e:
                return f"ERROR:{e}", 0

    def send_cmd_no_wait(self, cmd: str):
        if not self.connected:
            return
        full = f"{self.SPEED_PREFIX.get(self.speed,'2')}{cmd}" if cmd != "S" else "S"
        def _go():
            try:
                requests.get(
                    f"http://{self.ip}:{self.port}/cmd?v={full}", timeout=1)
                self.last_cmd = full
            except:
                pass
        threading.Thread(target=_go, daemon=True).start()


# ══════════════════════════════════════════════════════════════
#  YOLO  ASYNC  WORKER
# ══════════════════════════════════════════════════════════════

class YOLOWorker:
    """
    Runs YOLO inference in a dedicated daemon thread.
    The GUI thread posts frames; the worker posts results back.
    """
    def __init__(self, model):
        self.model      = model
        self._frame_q   = deque(maxlen=1)   # latest frame only
        self._result_q  = deque(maxlen=1)   # latest result only
        self._conf      = YOLO_CONF
        self._running   = True
        self._thread    = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def post_frame(self, frame, conf):
        self._conf = conf
        self._frame_q.append(frame.copy())

    def get_result(self):
        """Returns list of (x1,y1,x2,y2,class_id,confidence) or None."""
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
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cid  = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        boxes.append((int(x1), int(y1), int(x2), int(y2), cid, conf))
                self._result_q.append(boxes)
            except Exception as e:
                if DEBUG_GRID:
                    print(f"[YOLO thread] {e}")


# ══════════════════════════════════════════════════════════════
#  MAIN  GUI
# ══════════════════════════════════════════════════════════════

class AutobotNavigationGUI:

    def __init__(self, root):
        self.root  = root
        self.root.title(
            "Autobot Navigation  v3  –  AprilTag + YOLO (6-class) + Dynamic Grid")
        self.root.geometry("1860x1060")
        self.root.state("zoomed")

        # ── AprilTag detector ──────────────────────────────────
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

        # ── YOLO model + async worker ──────────────────────────
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

        # ── Class ignore set ────────────────────────────────────
        # tk.BooleanVar per class – updated live
        self._class_ignore_vars = {}

        # latest YOLO results  (x1,y1,x2,y2,class_id,conf)
        self._yolo_latest = []
        self._yolo_lock   = threading.Lock()

        # ── Capture state ──────────────────────────────────────
        self.cap          = None
        self.original_img = None
        self.display_img  = None
        self.is_capturing = False
        self.is_video     = False
        self.is_image     = False
        self._frame_delay = int(1000 / TARGET_FPS)

        # ── Video recording ────────────────────────────────────
        self._recorder     = None
        self._recording    = False

        # ── Detection state ────────────────────────────────────
        self.robot_pos      = None
        self.robot_theta    = 0.0
        self.all_tags       = {}
        self.obstacles      = []     # [(x1,y1,x2,y2,class_id,conf), …]
        self.robot_boxes    = {}

        # per-class confidence tracking
        self._class_conf_accum = defaultdict(list)

        # ── Calibration ────────────────────────────────────────
        self.cm_per_pixel   = None
        self.grid_size_px   = 20
        self.stop_thresh_px = 30

        # ── Planning ───────────────────────────────────────────
        self.target_pos = None
        self.path       = None
        self.grid       = None

        # ── Heat grid (FIX-4) ─────────────────────────────────
        self._heat_grid      = None
        self._heat_grid_size = None

        # ── Robot HTTP client ─────────────────────────────────
        self.robot = OmniRobotClient(DEFAULT_ROBOT_IP)

        # ── tk variables ──────────────────────────────────────
        self.auto_replan          = tk.BooleanVar(value=True)
        self.auto_follow          = tk.BooleanVar(value=False)
        self.show_detections      = tk.BooleanVar(value=True)
        self.show_grid            = tk.BooleanVar(value=False)
        self.show_path            = tk.BooleanVar(value=True)
        self.auto_detect          = tk.BooleanVar(value=False)
        self.show_heat            = tk.BooleanVar(value=True)
        self.use_rdp              = tk.BooleanVar(value=True)
        self.use_inflation        = tk.BooleanVar(value=True)
        self.use_heading_follow   = tk.BooleanVar(value=True)
        self.manual_calib_enabled = tk.BooleanVar(value=False)
        self.detect_n_var         = tk.IntVar(value=DETECT_EVERY_N)
        self.heat_thresh_var      = tk.IntVar(value=HEAT_THRESHOLD)
        self.heat_decay_var       = tk.IntVar(value=HEAT_DECAY)
        self.inflate_var          = tk.IntVar(value=OBSTACLE_INFLATE)
        self.speed_var            = tk.IntVar(value=2)
        self.manual_cmpx_var      = tk.DoubleVar(value=0.04)

        # ── FPS counter ────────────────────────────────────────
        self._frame_count = 0
        self._fps_display = 0.0
        self._fps_counter = 0
        self._fps_timer   = time.time()

        # ── Keyboard teleop ────────────────────────────────────
        self.pressed_keys = set()
        self.key_lock     = threading.Lock()

        self._setup_ui()
        self.root.bind("<KeyPress>",   self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────
    #  UI  BUILD
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

        # ── Right panel ────────────────────────────────────────
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        # ═══════════════════════════════
        #  SOURCE
        # ═══════════════════════════════
        sf = ttk.LabelFrame(left, text="📷  Source", padding="8")
        sf.pack(fill=tk.X, pady=4)

        ttk.Button(sf, text="Load Image",
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
        ttk.Label(sf, text="🌐  DroidCam / IP Camera",
                  font=("Arial",9,"bold")).pack(anchor=tk.W)
        ttk.Label(sf, text="IP:").pack(anchor=tk.W)
        self.droid_ip = ttk.Entry(sf); self.droid_ip.insert(0,"192.168.1.100")
        self.droid_ip.pack(fill=tk.X, pady=2)
        ttk.Label(sf, text="Port:").pack(anchor=tk.W)
        self.droid_port = ttk.Entry(sf); self.droid_port.insert(0,"4747")
        self.droid_port.pack(fill=tk.X, pady=2)
        self.droid_btn = ttk.Button(sf, text="Connect DroidCam",
                                    command=self._connect_droidcam)
        self.droid_btn.pack(fill=tk.X, pady=3)
        self.droid_dis = ttk.Button(sf, text="Disconnect DroidCam",
                                    command=self._disconnect, state=tk.DISABLED)
        self.droid_dis.pack(fill=tk.X)

        # ── Recording ──────────────────────────────────────────
        ttk.Separator(sf, orient="horizontal").pack(fill=tk.X, pady=4)
        self.rec_btn = ttk.Button(sf, text="⏺  Start Recording",
                                  command=self._toggle_recording)
        self.rec_btn.pack(fill=tk.X, pady=2)
        self.rec_lbl = ttk.Label(sf, text="Not recording", foreground="#888888",
                                 font=("Arial",8))
        self.rec_lbl.pack()

        # ═══════════════════════════════
        #  DETECTION
        # ═══════════════════════════════
        df = ttk.LabelFrame(left, text="🔍  Detection", padding="8")
        df.pack(fill=tk.X, pady=4)

        self.at_status   = ttk.Label(df,
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

        # Per-class confidence display
        self.cls_conf_lbl = ttk.Label(df, text="",
                                      font=("Courier",7), foreground="#555555")
        self.cls_conf_lbl.pack(anchor=tk.W, pady=2)

        # ── Class ignore checkboxes ────────────────────────────
        ttk.Separator(df, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Label(df, text="🚫  Ignore classes (no obstacle):",
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

        # ═══════════════════════════════
        #  CALIBRATION
        # ═══════════════════════════════
        gf = ttk.LabelFrame(left, text="📐  Calibration & Grid", padding="8")
        gf.pack(fill=tk.X, pady=4)

        ttk.Label(gf, text=f"Tag size: {TAG_SIZE_CM} cm").pack(anchor=tk.W)
        ttk.Label(gf, text=f"Robot size: {ROBOT_SIZE_CM} cm").pack(anchor=tk.W)
        ttk.Label(gf, text=f"Stop threshold: {STOP_THRESHOLD_CM} cm").pack(anchor=tk.W)

        self.calib_lbl = ttk.Label(gf, text="cm/pixel: calibrating…",
                                   font=("Arial",9,"bold"), foreground="#888888")
        self.calib_lbl.pack(pady=3)
        self.grid_lbl  = ttk.Label(gf, text="Grid cell: – px", font=("Arial",9))
        self.grid_lbl.pack()

        ttk.Checkbutton(gf, text="Manual calibration override",
                        variable=self.manual_calib_enabled,
                        command=self._on_manual_calib_toggle).pack(anchor=tk.W, pady=(4,0))
        self.manual_calib_frame = ttk.Frame(gf)
        self.manual_calib_frame.pack(fill=tk.X)
        ttk.Label(self.manual_calib_frame, text="cm/px:", font=("Arial",8)).pack(
            side=tk.LEFT, padx=(0,4))
        self.manual_cmpx_entry = ttk.Entry(self.manual_calib_frame, width=8,
                                           textvariable=self.manual_cmpx_var)
        self.manual_cmpx_entry.pack(side=tk.LEFT)
        self.manual_cmpx_entry.config(state=tk.DISABLED)
        ttk.Button(self.manual_calib_frame, text="Apply",
                   command=self._apply_manual_calib).pack(side=tk.LEFT, padx=4)

        ttk.Separator(gf, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Label(gf, text="Heat threshold (frames):").pack(anchor=tk.W)
        self.heat_thresh_spin = ttk.Spinbox(gf, from_=1, to=20,
                                            textvariable=self.heat_thresh_var,
                                            width=5)
        self.heat_thresh_spin.pack(anchor=tk.W)
        ttk.Label(gf, text="Heat decay (per frame):").pack(anchor=tk.W)
        ttk.Spinbox(gf, from_=1, to=10,
                    textvariable=self.heat_decay_var, width=5).pack(anchor=tk.W)
        ttk.Label(gf, text=f"Edge margin: {EDGE_MARGIN_PX} px",
                  font=("Arial",8,"italic")).pack(anchor=tk.W)
        ttk.Label(gf, text=f"Box shrink: {BOX_SHRINK_PX} px inset",
                  font=("Arial",8,"italic")).pack(anchor=tk.W)

        ttk.Separator(gf, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Label(gf, text="Obstacle inflate (cells):").pack(anchor=tk.W)
        ttk.Spinbox(gf, from_=0, to=5,
                    textvariable=self.inflate_var, width=5).pack(anchor=tk.W)

        # ═══════════════════════════════
        #  POSITIONS
        # ═══════════════════════════════
        pf = ttk.LabelFrame(left, text="📍  Positions", padding="8")
        pf.pack(fill=tk.X, pady=4)

        ttk.Label(pf, text="Click image to set target",
                  font=("Arial",9,"italic")).pack(anchor=tk.W, pady=2)
        self.robot_lbl = ttk.Label(pf,
            text="🤖 AUTOBOT: Not detected", font=("Arial",9))
        self.robot_lbl.pack(anchor=tk.W, pady=2)
        self.target_lbl = ttk.Label(pf,
            text="🎯 Target: Not set", font=("Arial",9))
        self.target_lbl.pack(anchor=tk.W, pady=2)
        ttk.Button(pf, text="Clear Target",
                   command=self._clear_target).pack(fill=tk.X, pady=4)

        # ═══════════════════════════════
        #  PATH PLANNING
        # ═══════════════════════════════
        plf = ttk.LabelFrame(left, text="🗺️  Path Planning", padding="8")
        plf.pack(fill=tk.X, pady=4)

        ttk.Button(plf, text="🧭  Calculate A* Path",
                   command=self._calculate_path).pack(fill=tk.X, pady=4)
        ttk.Checkbutton(plf, text="♻️  Auto Re-plan",
                        variable=self.auto_replan).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(plf, text="📐  RDP Path Simplification",
                        variable=self.use_rdp).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(plf, text="🛡️  Obstacle Inflation",
                        variable=self.use_inflation).pack(anchor=tk.W, pady=2)

        self.path_info = ttk.Label(plf, text="No path", font=("Arial",9))
        self.path_info.pack(pady=4)

        ttk.Checkbutton(plf, text="Show Path",
                        variable=self.show_path,
                        command=self._redraw).pack(anchor=tk.W)
        ttk.Checkbutton(plf, text="Show Grid",
                        variable=self.show_grid,
                        command=self._redraw).pack(anchor=tk.W)
        ttk.Checkbutton(plf, text="Show Heat Map",
                        variable=self.show_heat,
                        command=self._redraw).pack(anchor=tk.W)

        # ═══════════════════════════════
        #  ROBOT CONTROL
        # ═══════════════════════════════
        rf = ttk.LabelFrame(left, text="🤖  Robot Control (Arduino R4 WiFi)",
                            padding="8")
        rf.pack(fill=tk.X, pady=4)

        ttk.Label(rf, text="Robot IP:", font=("Arial",9,"bold")).pack(anchor=tk.W)
        rif = ttk.Frame(rf); rif.pack(fill=tk.X, pady=2)
        ttk.Label(rif, text="IP:").pack(side=tk.LEFT, padx=(0,4))
        self.robot_ip = ttk.Entry(rif, width=16)
        self.robot_ip.insert(0, DEFAULT_ROBOT_IP)
        self.robot_ip.pack(side=tk.LEFT, padx=(0,4))
        ttk.Label(rif, text="Port:").pack(side=tk.LEFT, padx=(0,2))
        self.robot_port_e = ttk.Entry(rif, width=5)
        self.robot_port_e.insert(0, str(ROBOT_PORT))
        self.robot_port_e.pack(side=tk.LEFT)

        self.rob_conn_btn = ttk.Button(rf, text="Connect Robot",
                                       command=self._connect_robot)
        self.rob_conn_btn.pack(fill=tk.X, pady=2)
        self.rob_dis_btn = ttk.Button(rf, text="Disconnect Robot",
                                      command=self._disconnect_robot,
                                      state=tk.DISABLED)
        self.rob_dis_btn.pack(fill=tk.X, pady=2)
        self.rob_status = ttk.Label(rf, text="Not connected",
                                    foreground="#aa0000")
        self.rob_status.pack(pady=2)

        ttk.Separator(rf, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Label(rf, text="🏎️  Speed:", font=("Arial",9,"bold")).pack(anchor=tk.W)
        spf = ttk.Frame(rf); spf.pack(fill=tk.X)
        for val, lbl in [(1,"Slow"),(2,"Normal"),(3,"Fast")]:
            ttk.Radiobutton(spf, text=lbl, variable=self.speed_var, value=val,
                            command=lambda: setattr(self.robot, "speed",
                                                    self.speed_var.get())
                            ).pack(side=tk.LEFT, padx=4)

        ttk.Separator(rf, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Label(rf, text="⌨️  Keyboard Teleop:",
                  font=("Arial",8,"bold")).pack(anchor=tk.W)
        for line in ["W = Forward      S = Backward",
                     "A = Strafe Left  D = Strafe Right",
                     "Q = Fwd-Left     E = Fwd-Right",
                     "Z = Bwd-Left     C = Bwd-Right",
                     "SPACE = ⛔  Emergency Stop"]:
            ttk.Label(rf, text=line, font=("Courier",7)).pack(anchor=tk.W)

        ttk.Separator(rf, orient="horizontal").pack(fill=tk.X, pady=4)
        ttk.Checkbutton(rf, text="🚀  Auto Path Follow",
                        variable=self.auto_follow).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(rf, text="🧭  Heading-Aware Follow",
                        variable=self.use_heading_follow).pack(anchor=tk.W, pady=2)

        self.follow_info = ttk.Label(rf, text="", font=("Arial",8))
        self.follow_info.pack()

        # ═══════════════════════════════
        #  EXPORT
        # ═══════════════════════════════
        ef = ttk.LabelFrame(left, text="💾  Export", padding="8")
        ef.pack(fill=tk.X, pady=4)
        ttk.Button(ef, text="Save Screenshot",
                   command=self._save_screenshot).pack(fill=tk.X, pady=2)
        ttk.Button(ef, text="Export Path CSV",
                   command=self._export_path).pack(fill=tk.X, pady=2)

        # ═══════════════════════════════
        #  STATS
        # ═══════════════════════════════
        stf = ttk.LabelFrame(left, text="📊  Stats", padding="8")
        stf.pack(fill=tk.X, pady=4, expand=True)
        self.stats_text = tk.Text(stf, height=12, width=34,
                                  font=("Courier",7), state=tk.DISABLED)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # ═══════════════════════════════
        #  CANVAS  (right panel)
        # ═══════════════════════════════
        cf = ttk.LabelFrame(right, text="Arena View", padding="4")
        cf.grid(row=0, column=0, sticky="nsew")
        cf.columnconfigure(0, weight=1)
        cf.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(cf, width=DISPLAY_W, height=DISPLAY_H,
                                bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Status bar
        self.status = ttk.Label(self.root,
            text="Ready  [v3 – 6-class async YOLO + heading-aware A* follow]",
            relief=tk.SUNKEN, anchor=tk.W)
        self.status.grid(row=1, column=0, sticky="ew")

    # ─────────────────────────────────────────────────────────
    #  SOURCE  MANAGEMENT
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
                self.status.config(text=f"DroidCam: {url}")
                self._live_loop(); return
            cap.release()
        messagebox.showerror("Error",f"Cannot connect to DroidCam {ip}:{port}")

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
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

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
        self.rec_lbl.config(text=f"● REC: {os.path.basename(path)}",
                            foreground="#cc0000")
        self.status.config(text=f"Recording to {os.path.basename(path)}")

    def _stop_recording(self):
        if self._recorder:
            self._recorder.release()
            self._recorder = None
        self._recording = False
        self.rec_btn.config(text="⏺  Start Recording")
        self.rec_lbl.config(text="Not recording", foreground="#888888")

    def _write_recording(self, frame):
        if self._recording and self._recorder:
            self._recorder.write(frame)

    # ─────────────────────────────────────────────────────────
    #  FRAME  PROCESSING
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
        # For single-shot: run synchronously so result is available immediately
        if self.yolo_worker:
            conf = max(float(self.conf_scale.get()), YOLO_MIN_CONF)
            self.yolo_worker.post_frame(
                self._mask_robot_regions(self.display_img), conf)
            time.sleep(0.15)   # brief wait for async result
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
        """Rebuild ignore set and clear heat grid so stale cells vanish."""
        self._heat_grid = None

    # ─────────────────────────────────────────────────────────
    #  DETECTION  –  AprilTag
    # ─────────────────────────────────────────────────────────

    def _run_apriltag(self):
        if self.display_img is None: return
        self.all_tags    = {}
        self.robot_boxes = {}
        self.robot_pos   = None
        self.robot_theta = 0.0
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
            if edge_px > 5:
                cmp = TAG_SIZE_CM / edge_px
                if not self.manual_calib_enabled.get():
                    self.cm_per_pixel   = cmp
                    self.grid_size_px   = max(10,
                        int(ROBOT_SIZE_CM / self.cm_per_pixel))
                    self.stop_thresh_px = STOP_THRESHOLD_CM / self.cm_per_pixel
                    self.calib_lbl.config(
                        text=f"cm/pixel: {self.cm_per_pixel:.4f}",
                        foreground="#007700")
                    self.grid_lbl.config(
                        text=f"Grid cell: {self.grid_size_px} px")

            pad = max(20, int(0.40 * edge_px)) if edge_px > 0 else 20
            xs, ys = corners[:, 0], corners[:, 1]
            bx1 = int(max(0, min(DISPLAY_W-1, xs.min()-pad)))
            by1 = int(max(0, min(DISPLAY_H-1, ys.min()-pad)))
            bx2 = int(max(0, min(DISPLAY_W-1, xs.max()+pad)))
            by2 = int(max(0, min(DISPLAY_H-1, ys.max()+pad)))
            self.robot_boxes[tid] = (bx1, by1, bx2, by2)

            if tid == 0:
                self.robot_pos   = (int(cx), int(cy))
                self.robot_theta = theta

    # ─────────────────────────────────────────────────────────
    #  DETECTION  –  YOLO  (async)
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
    #  DETECTION  –  MERGE  (FIX-3, FIX-5, FIX-6)
    # ─────────────────────────────────────────────────────────

    def _merge_detections(self):
        with self._yolo_lock:
            raw = list(self._yolo_latest)

        self.obstacles = []
        ignore_set = {cid for cid, var in self._class_ignore_vars.items()
                      if var.get()}

        # reset per-class confidence accumulators each merge cycle
        self._class_conf_accum = defaultdict(list)

        for (bx1, by1, bx2, by2, cid, conf) in raw:
            # Hard conf floor (FIX-5)
            if conf < YOLO_MIN_CONF:
                continue

            # Ignore filtered classes (e.g. table)
            if cid in ignore_set:
                continue

            # Track confidence
            self._class_conf_accum[cid].append(conf)

            # Edge margin (FIX-3)
            if (bx1 < EDGE_MARGIN_PX or by1 < EDGE_MARGIN_PX
                    or bx2 > DISPLAY_W - EDGE_MARGIN_PX
                    or by2 > DISPLAY_H - EDGE_MARGIN_PX):
                if DEBUG_GRID:
                    print(f"[EDGE DROP] class={YOLO_CLASSES.get(cid,'?')} "
                          f"box=({bx1},{by1},{bx2},{by2})")
                continue

            # Box shrink (FIX-6)
            bx1s = bx1 + BOX_SHRINK_PX
            by1s = by1 + BOX_SHRINK_PX
            bx2s = bx2 - BOX_SHRINK_PX
            by2s = by2 - BOX_SHRINK_PX
            if bx2s <= bx1s or by2s <= by1s:
                continue

            # Skip if overlapping robot bounding box
            overlaps_robot = False
            for rb in self.robot_boxes.values():
                rx1, ry1, rx2, ry2 = rb
                if not (bx2s < rx1 or bx1s > rx2 or by2s < ry1 or by1s > ry2):
                    overlaps_robot = True; break
            if not overlaps_robot:
                self.obstacles.append((bx1s, by1s, bx2s, by2s, cid, conf))

        # Update per-class conf display
        lines = []
        for cid in YOLO_CLASSES:
            cfvals = self._class_conf_accum.get(cid, [])
            avg = f"{np.mean(cfvals):.2f}" if cfvals else " ——"
            marker = "●" if cfvals else "○"
            lines.append(f"{marker} {YOLO_CLASSES[cid]:<12} {avg}")
        self.cls_conf_lbl.config(text="\n".join(lines))

    def _mask_robot_regions(self, frame):
        if frame is None: return frame
        masked = frame.copy()
        for bx1, by1, bx2, by2 in self.robot_boxes.values():
            cv2.rectangle(masked, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        return masked

    def _post_detect_update(self):
        n_tags = len(self.all_tags)
        n_obs  = len(self.obstacles)
        self.det_info.config(
            text=f"Tags: {n_tags}   Obstacles: {n_obs}",
            foreground="#007700" if n_tags > 0 else "#ff6600")

        if self.robot_pos:
            self.robot_lbl.config(
                text=(f"🤖 AUTOBOT: {self.robot_pos}  "
                      f"θ={int(self.robot_theta)}°"),
                foreground="#007700")
        else:
            self.robot_lbl.config(
                text="🤖 AUTOBOT: Not detected", foreground="#ff0000")

        if (self.robot_pos and self.target_pos
                and self.auto_replan.get() and self.path):
            self._calculate_path_silent()

        if self.auto_follow.get():
            self._follow_step()

        self._update_stats()

    # ─────────────────────────────────────────────────────────
    #  HEAT  GRID  (FIX-4)
    # ─────────────────────────────────────────────────────────

    def _ensure_heat_grid(self):
        gs   = self.grid_size_px
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
    #  GRID  BUILD  (FIX-1, FIX-2, FIX-4, FIX-9)
    # ─────────────────────────────────────────────────────────

    def _build_grid(self):
        gs    = self.grid_size_px
        rows  = DISPLAY_H // gs
        cols  = DISPLAY_W  // gs
        hthr  = max(1, self.heat_thresh_var.get())
        infl  = max(0, self.inflate_var.get()) if self.use_inflation.get() else 0

        self._ensure_heat_grid()
        heat = self._heat_grid

        # Accumulate heat for current obstacles (FIX-2: hard clamp)
        for x1, y1, x2, y2, *_ in self.obstacles:
            if (y2 - y1) > DISPLAY_H * 0.60 or (x2 - x1) > DISPLAY_W * 0.60:
                continue
            r1 = max(0, min(rows-1,  int(y1) // gs))
            r2 = max(0, min(rows-1, (int(y2)) // gs))
            c1 = max(0, min(cols-1,  int(x1) // gs))
            c2 = max(0, min(cols-1, (int(x2)) // gs))
            if DEBUG_GRID:
                print(f"[GRID] obs=({x1},{y1},{x2},{y2}) → "
                      f"r[{r1}:{r2}] c[{c1}:{c2}]")
            heat[r1:r2+1, c1:c2+1] += 1

        np.minimum(heat, hthr + 10, out=heat)

        # Binary grid from heat threshold (FIX-4)
        grid = np.zeros((rows, cols), dtype=np.uint8)
        grid[heat >= hthr] = 1

        # Obstacle inflation (safety margin)
        if infl > 0:
            from scipy.ndimage import binary_dilation
            try:
                struct = np.ones((2*infl+1, 2*infl+1), dtype=bool)
                grid   = binary_dilation(grid, structure=struct).astype(np.uint8)
            except ImportError:
                # fallback: manual dilation
                dilated = grid.copy()
                for dr in range(-infl, infl+1):
                    for dc in range(-infl, infl+1):
                        shifted = np.roll(np.roll(grid, dr, axis=0), dc, axis=1)
                        dilated |= shifted
                grid = dilated

        # Clear robot body cells
        for x1, y1, x2, y2 in self.robot_boxes.values():
            r1 = max(0, min(rows-1,  int(y1) // gs))
            r2 = max(0, min(rows-1, (int(y2)) // gs))
            c1 = max(0, min(cols-1,  int(x1) // gs))
            c2 = max(0, min(cols-1, (int(x2)) // gs))
            grid[r1:r2+1, c1:c2+1] = 0
            heat[r1:r2+1, c1:c2+1] = 0

        return grid

    # ─────────────────────────────────────────────────────────
    #  COORDINATE  HELPERS
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
    #  PATH  PLANNING
    # ─────────────────────────────────────────────────────────

    def _calculate_path_silent(self):
        if self.robot_pos is None or self.target_pos is None: return
        self.grid = self._build_grid()
        sg = self._px_to_grid(*self.robot_pos)
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
        if self.robot_pos is None:
            messagebox.showwarning("No robot",
                "AUTOBOT not detected – point camera at AprilTag ID 0"); return
        if self.target_pos is None:
            messagebox.showwarning("No target",
                "Click on the arena to set a target"); return

        self.grid = self._build_grid()
        sg = self._px_to_grid(*self.robot_pos)
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
    #  AUTO  FOLLOW  (heading-aware)
    # ─────────────────────────────────────────────────────────

    def _follow_step(self):
        with self.key_lock:
            if any(k in self.pressed_keys for k in ("W","S","A","D")): return
        if not self.robot.connected: return
        if not self.path or len(self.path) < 2: return
        if self.robot_pos is None: return

        wp   = self.path[1]
        dx   = wp[0] - self.robot_pos[0]
        dy   = wp[1] - self.robot_pos[1]
        dist = math.hypot(dx, dy)

        stop_px = (self.stop_thresh_px
                   if self.cm_per_pixel
                   else self.grid_size_px * 0.8)

        if dist < self.grid_size_px * 0.7:
            self.path.pop(0)
            if len(self.path) < 2:
                self.robot.send_cmd_no_wait("S")
                self.auto_follow.set(False)
                self.path_info.config(
                    text="✓ Target reached!", foreground="#007700")
                self.status.config(
                    text="✓ Target reached! Robot stopped.")
            return

        if dist < stop_px:
            self.robot.send_cmd_no_wait("S")
            self.auto_follow.set(False)
            self.path_info.config(
                text="✓ Within stop threshold", foreground="#007700")
            return

        cmd = (self._vec_to_cmd_heading(dx, dy, self.robot_theta)
               if self.use_heading_follow.get()
               else self._vec_to_cmd(dx, dy))

        self.robot.send_cmd_no_wait(cmd)
        self.follow_info.config(
            text=f"→ {cmd}  dist={dist:.0f}px  "
                 f"θ={int(self.robot_theta)}°",
            foreground="#007700")

    def _vec_to_cmd(self, dx, dy) -> str:
        """Simple dx/dy → command (no heading)."""
        ax, ay = abs(dx), abs(dy)
        if   ax > ay * 2:  return "R" if dx > 0 else "L"
        elif ay > ax * 2:  return "B" if dy > 0 else "F"
        elif dx > 0 and dy < 0: return "D"
        elif dx < 0 and dy < 0: return "A"
        elif dx > 0 and dy > 0: return "E"
        else:                   return "Q"

    def _vec_to_cmd_heading(self, dx, dy, theta_deg) -> str:
        """
        Heading-aware command: compute target bearing in world frame,
        then subtract robot heading to get the relative bearing,
        and map to one of the 8 omniwheel directions.
        """
        world_angle = math.degrees(math.atan2(-dy, dx)) % 360.0
        # AprilTag theta is the tag's forward direction; robot forward = -theta
        robot_fwd   = (- theta_deg) % 360.0
        rel         = (world_angle - robot_fwd) % 360.0

        # 8-sector mapping (45° per sector, centred)
        if   rel <  22.5 or rel >= 337.5: return "F"
        elif rel <  67.5:                 return "D"    # fwd-right
        elif rel < 112.5:                 return "R"
        elif rel < 157.5:                 return "E"    # bwd-right
        elif rel < 202.5:                 return "B"
        elif rel < 247.5:                 return "Q"    # bwd-left
        elif rel < 292.5:                 return "L"
        else:                             return "A"    # fwd-left

    # ─────────────────────────────────────────────────────────
    #  DRAWING
    # ─────────────────────────────────────────────────────────

    def _redraw(self):
        if self.display_img is None: return
        img = self.display_img.copy()

        # Grid overlay
        if self.show_grid.get():
            gs  = self.grid_size_px
            h, w = img.shape[:2]
            ov  = img.copy()
            for x in range(0, w, gs): cv2.line(ov,(x,0),(x,h),(60,60,60),1)
            for y in range(0, h, gs): cv2.line(ov,(0,y),(w,y),(60,60,60),1)

            # Heat map tint (FIX-4)
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
                            sub[:] = (sub * (1-alpha*0.55)
                                      + np.array(color) * (alpha*0.55)
                                      ).astype(np.uint8)
            img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)

        # YOLO detections – class-coloured
        if self.show_detections.get():
            for x1, y1, x2, y2, cid, conf in self.obstacles:
                col   = CLASS_COLORS.get(cid, (0, 0, 220))
                name  = YOLO_CLASSES.get(cid, f"cls{cid}")
                label = f"{name} {conf:.2f}"
                cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
                # label background
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                lx, ly = x1, max(0, y1-4)
                cv2.rectangle(img,
                              (lx, ly - th - 4), (lx + tw + 4, ly),
                              col, -1)
                cv2.putText(img, label, (lx+2, ly-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # AprilTag detections
            for tid, (cx, cy, theta, corners) in self.all_tags.items():
                col  = ROBOT_COLORS.get(tid, (200,200,200))
                name = ROBOT_NAMES.get(tid, f"ID{tid}")
                icx, icy = int(cx), int(cy)
                cv2.polylines(img, [corners.astype(int)], True, col, 2)
                cv2.circle(img, (icx,icy), 22, col, 2)
                cv2.drawMarker(img, (icx,icy), col, cv2.MARKER_DIAMOND, 16, 2)
                r  = math.radians(theta)
                ex = int(icx + 35*math.cos(r))
                ey = int(icy + 35*math.sin(r))
                cv2.arrowedLine(img, (icx,icy), (ex,ey), col, 2, tipLength=0.4)
                cv2.putText(img, name,
                            (icx+28, icy-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
                cv2.putText(img, f"({icx},{icy}) θ={int(theta)}°",
                            (icx+28, icy+8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

            # Robot bounding boxes
            for tid, (bx1,by1,bx2,by2) in self.robot_boxes.items():
                col  = ROBOT_COLORS.get(tid, (200,200,200))
                name = ROBOT_NAMES.get(tid, f"ID{tid}")
                cv2.rectangle(img, (bx1,by1), (bx2,by2), col, 1)

        # Target marker
        if self.target_pos:
            tx, ty = self.target_pos
            cv2.drawMarker(img, (tx,ty), (0,255,0), cv2.MARKER_CROSS, 32, 3)
            cv2.circle(img, (tx,ty), 22, (0,255,255), 3)
            cv2.putText(img, "TARGET", (tx+26, ty-26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        # Path overlay
        if self.show_path.get() and self.path and len(self.path) > 1:
            for i in range(len(self.path)-1):
                cv2.line(img, self.path[i], self.path[i+1], (0,220,255), 2)
            for i, pt in enumerate(self.path):
                if   i == 0:               cv2.circle(img, pt, 8, (0,255,0), -1)
                elif i == len(self.path)-1: cv2.circle(img, pt, 8, (0,0,255), -1)
                else:                       cv2.circle(img, pt, 5, (255,220,0), -1)

        # HUD overlays
        if self.is_capturing and self._fps_display > 0:
            cv2.putText(img, f"{self._fps_display:.1f} FPS",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,0), 2)

        if self._recording:
            cv2.circle(img, (DISPLAY_W-20, 18), 8, (0,0,220), -1)
            cv2.putText(img, "REC",
                        (DISPLAY_W-50, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,220), 1)

        if self.cm_per_pixel:
            cv2.putText(img,
                f"Scale: {self.cm_per_pixel:.4f} cm/px  "
                f"Grid: {self.grid_size_px}px  "
                f"Heat≥{self.heat_thresh_var.get()}",
                (10, DISPLAY_H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,0), 1)

        # Write to recorder (before resize)
        if self._recording:
            self._write_recording(img)

        # FIX-8: scale to actual canvas size before displaying
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw > 10 and ch > 10 and (cw != DISPLAY_W or ch != DISPLAY_H):
            img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

        rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk   # keep reference

    # ─────────────────────────────────────────────────────────
    #  TARGET
    # ─────────────────────────────────────────────────────────

    def _on_canvas_click(self, event):
        if self.display_img is None and not self.is_image: return
        # FIX-8: remap canvas coords back to display-pixel space
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        cx = int(event.x * DISPLAY_W / cw) if cw > 10 else event.x
        cy = int(event.y * DISPLAY_H / ch) if ch > 10 else event.y
        self.target_pos = (cx, cy)
        self.target_lbl.config(
            text=f"🎯 Target: ({cx}, {cy})", foreground="#007700")
        self.path = None
        self.path_info.config(text="Target set – calculate path")
        self._redraw()
        self._update_stats()

    def _clear_target(self):
        self.target_pos = None
        self.path       = None
        self.target_lbl.config(
            text="🎯 Target: Not set", foreground="#888888")
        self.path_info.config(text="No path")
        self._redraw()
        self._update_stats()

    # ─────────────────────────────────────────────────────────
    #  CALIBRATION  OVERRIDE
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
        self.grid_size_px   = max(10, int(ROBOT_SIZE_CM / val))
        self.stop_thresh_px = STOP_THRESHOLD_CM / val
        self._heat_grid     = None   # reset heat grid on scale change
        self.calib_lbl.config(
            text=f"cm/pixel: {val:.4f} (manual)",
            foreground="#e06000")
        self.grid_lbl.config(text=f"Grid cell: {self.grid_size_px} px")

    # ─────────────────────────────────────────────────────────
    #  ROBOT  CONNECTION
    # ─────────────────────────────────────────────────────────

    def _connect_robot(self):
        ip = self.robot_ip.get().strip()
        try: port = int(self.robot_port_e.get().strip())
        except:
            messagebox.showerror("Error","Invalid port"); return
        self.robot.set_ip(ip)
        self.robot.port  = port
        self.robot.speed = self.speed_var.get()
        self.status.config(text=f"Connecting to {ip}:{port}…")
        self.root.update()
        ok, err = self.robot.connect()
        if ok:
            self.rob_conn_btn.config(state=tk.DISABLED)
            self.rob_dis_btn.config(state=tk.NORMAL)
            self.rob_status.config(
                text=f"✓ {ip}:{port}", foreground="#007700")
            self.status.config(
                text=f"✓ Connected to Autobot {ip}:{port}")
        else:
            messagebox.showerror(
                "Connect Error",
                f"Cannot connect to {ip}:{port}\n\n{err}\n\n"
                "Check:\n1. IP correct (from Serial Monitor)\n"
                "2. Connected to OMNI_ROBOT WiFi\n3. Arduino powered on")
            self.rob_status.config(
                text="Not connected", foreground="#aa0000")

    def _disconnect_robot(self):
        self.robot.disconnect()
        self.rob_conn_btn.config(state=tk.NORMAL)
        self.rob_dis_btn.config(state=tk.DISABLED)
        self.rob_status.config(
            text="Not connected", foreground="#aa0000")
        self.status.config(text="Robot disconnected")

    # ─────────────────────────────────────────────────────────
    #  KEYBOARD  TELEOP
    # ─────────────────────────────────────────────────────────

    def _on_key_press(self, event):
        key = event.keysym.upper()
        with self.key_lock:
            if key in ("W","A","S","D","Q","E","Z","C"):
                if key not in self.pressed_keys:
                    self.pressed_keys.add(key)
                    self._send_movement()
            elif key == "SPACE":
                self._emergency_stop()

    def _on_key_release(self, event):
        key = event.keysym.upper()
        with self.key_lock:
            if key in ("W","A","S","D","Q","E","Z","C"):
                self.pressed_keys.discard(key)
                self._send_movement()

    def _send_movement(self):
        if not self.robot.connected: return
        keys = self.pressed_keys
        if   "W" in keys and not any(k in keys for k in ("A","D","S")): self._send("F")
        elif "S" in keys and not any(k in keys for k in ("A","D","W")): self._send("B")
        elif "A" in keys and not any(k in keys for k in ("W","S","D")): self._send("L")
        elif "D" in keys and not any(k in keys for k in ("W","S","A")): self._send("R")
        elif "Q" in keys: self._send("A")
        elif "E" in keys: self._send("D")
        elif "Z" in keys: self._send("Q")
        elif "C" in keys: self._send("E")
        elif "W" in keys and "A" in keys: self._send("A")
        elif "W" in keys and "D" in keys: self._send("D")
        elif "S" in keys and "A" in keys: self._send("Q")
        elif "S" in keys and "D" in keys: self._send("E")
        else: self._send("S")

    def _send(self, cmd):
        try:
            reply, ms = self.robot.send_cmd(cmd)
            self.status.config(
                text=(f"→ {cmd} [{ms:.1f}ms]"
                      if reply == "OK"
                      else f"✗ {cmd}: {reply}"))
        except Exception as e:
            self.status.config(text=f"Error: {e}")

    def _emergency_stop(self):
        with self.key_lock:
            self.pressed_keys.clear()
        self.auto_follow.set(False)
        self.robot.send_cmd_no_wait("S")
        self.path_info.config(
            text="⛔ EMERGENCY STOP", foreground="#ff0000")
        self.status.config(
            text="⛔ EMERGENCY STOP – Robot stopped!")
        self.follow_info.config(text="")

    # ─────────────────────────────────────────────────────────
    #  STATS
    # ─────────────────────────────────────────────────────────

    def _update_stats(self):
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)

        lines = ["=== APRILTAGS ===", f"Detected: {len(self.all_tags)}"]
        for tid, (cx,cy,theta,_) in self.all_tags.items():
            lines.append(f"  ID{tid} {ROBOT_NAMES.get(tid,'?')}: "
                         f"({int(cx)},{int(cy)}) θ={int(theta)}°")

        lines += ["", "=== CALIBRATION ===",
                  f"cm/pixel : {f'{self.cm_per_pixel:.4f}' if self.cm_per_pixel else 'pending'}",
                  f"Manual   : {'yes' if self.manual_calib_enabled.get() else 'no'}",
                  f"Grid cell: {self.grid_size_px} px",
                  f"Stop thr : {self.stop_thresh_px:.1f} px",
                  f"Inflate  : {self.inflate_var.get()} cells",
                  "", "=== OBSTACLES ===",
                  f"Total: {len(self.obstacles)}"]

        class_counts = defaultdict(int)
        for *_, cid, _ in self.obstacles:
            class_counts[YOLO_CLASSES.get(cid,'?')] += 1
        for cname, cnt in sorted(class_counts.items()):
            lines.append(f"  {cname}: {cnt}")

        lines += ["", "=== PATH ==="]
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

        if self.robot_pos and self.target_pos:
            d = math.hypot(
                self.target_pos[0]-self.robot_pos[0],
                self.target_pos[1]-self.robot_pos[1])
            lines.append(f"Direct dist: {d:.0f}px")
            if self.cm_per_pixel:
                lines.append(f"  = {d*self.cm_per_pixel:.1f} cm")

        lines += ["", "=== ROBOT ===",
                  f"Connected: {self.robot.connected}",
                  f"Speed    : {['Slow','Normal','Fast'][self.robot.speed-1]}",
                  f"Last cmd : {self.robot.last_cmd or '–'}",
                  f"Heading  : {int(self.robot_theta)}°"]

        self.stats_text.insert(1.0, "\n".join(lines))
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
            messagebox.showerror("Error", str(e))

    def _export_path(self):
        if not self.path:
            messagebox.showwarning("No path","Calculate path first"); return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("Text","*.txt"),("All","*.*")])
        if not path: return
        try:
            with open(path, "w") as f:
                f.write("# Autobot v3 path export\n")
                f.write(f"# Robot pos: {self.robot_pos}\n")
                f.write(f"# Target   : {self.target_pos}\n")
                f.write(f"# Heading  : {self.robot_theta:.1f}°\n")
                f.write(f"# cm/pixel : {self.cm_per_pixel}\n")
                f.write("index,x_px,y_px,x_cm,y_cm\n")
                for i, (x,y) in enumerate(self.path):
                    xcm = f"{x*self.cm_per_pixel:.2f}" if self.cm_per_pixel else ""
                    ycm = f"{y*self.cm_per_pixel:.2f}" if self.cm_per_pixel else ""
                    f.write(f"{i},{x},{y},{xcm},{ycm}\n")
            self.status.config(text="✓ Path exported")
        except Exception as e:
            messagebox.showerror("Error", str(e))

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
#  ENTRY  POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = AutobotNavigationGUI(root)
    root.mainloop()