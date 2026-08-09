"""
camera_calibration.py  –  Single-File Camera Calibration Tool
═══════════════════════════════════════════════════════════════════════════════
Robot Banner-Grid Navigation System  ·  Calibration Module

Layout
──────
  Left panel (scrollable)  ←→  Right canvas (live preview)

Left panel sections (top → bottom)
────────────────────────────────────
  1. CAMERA SOURCE     – webcam scan / DroidCam / IP camera / connect
  2. LENS CALIBRATION  – capture frames → run calibrateCamera → get RMS
  3. BANNER MAPPING    – auto-detect banner boundary (contour + Hough)
                         manual 4-corner fallback → compute homography
  4. SAVE / LOAD       – name the file yourself, pick a folder, save
                         load existing .npz, view/delete saved profiles

What gets saved (.npz)
────────────────────────
  camera_matrix, dist_coeffs, resolution, rms_error,
  camera_name, profile_name,
  height_cm, arena_w_cm, arena_h_cm, cell_size_cm, grid_cols, grid_rows,
  tilt_deg, facing_dir, notes,
  cm_per_pixel, anchor_px, anchor_cell,   ← mapping (if banner detected)
  homography, banner_corners_px            ← perspective data

Banner assumptions
───────────────────
  • Plain uniform 20 × 20 square grid printed on banner.
  • NOT a chessboard — no alternating black/white squares.
  • Each cell = 15 cm × 15 cm  →  full banner = 300 × 300 cm.
  • Camera overhead (orthogonal or near-orthogonal).
  • One AprilTag (ID 0) on the robot provides live cm/pixel at runtime;
    this calibration provides the homography for undistortion + anchor.

Compatibility
─────────────
  Saved .npz files are consumed by Autobot_navigation_v4.py via
  the load_calibration_profile() helper at the bottom of this file.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import os
import platform
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════

BANNER_COLS    = 20
BANNER_ROWS    = 20
BANNER_CELL_CM = 15.0
BANNER_W_CM    = BANNER_COLS * BANNER_CELL_CM   # 300
BANNER_H_CM    = BANNER_ROWS * BANNER_CELL_CM   # 300

INNER_COLS     = BANNER_COLS - 1   # 19  (inner intersection columns)
INNER_ROWS     = BANNER_ROWS - 1   # 19  (inner intersection rows)

MIN_CAL_FRAMES = 3
REC_CAL_FRAMES = 10

TARGET_FPS     = 25
FRAME_MS       = int(1000 / TARGET_FPS)

LEFT_W         = 370

# ── Palette (dark industrial) ─────────────────────────────────
BG      = "#1a1a1e"
BG2     = "#25252b"
BG3     = "#2e2e36"
ACCENT  = "#00d4aa"
ACCENT2 = "#ff6b35"
FG      = "#e8e8e8"
FG2     = "#aaaaaa"
GREEN   = "#00cc77"
RED     = "#ff4444"
MONO    = ("Courier New", 9)
LABELF  = ("Segoe UI", 9)
HEADF   = ("Segoe UI", 10, "bold")

DEFAULT_SAVE_DIR = Path("calibration_profiles")


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — CAMERA DETECTION
# ══════════════════════════════════════════════════════════════

@dataclass
class CameraSource:
    label:       str
    device:      Any          # int index or URL string
    source_type: str          # "webcam" | "droidcam" | "ipcam"

    def __str__(self) -> str:
        return self.label


def _is_windows() -> bool:
    return platform.system() == "Windows"


def scan_webcams(max_index: int = 8) -> list[CameraSource]:
    """Probe indices 0…max_index-1; return those that yield a frame."""
    found: list[CameraSource] = []
    backend = cv2.CAP_DSHOW if _is_windows() else cv2.CAP_ANY
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ret, _ = cap.read()
        cap.release()
        if not ret:
            continue
        label = f"Webcam  {idx}" + ("  (built-in)" if idx == 0 else "")
        found.append(CameraSource(label=label, device=idx, source_type="webcam"))
    return found


def probe_url(url: str, timeout: float = 4.0) -> bool:
    """Return True if OpenCV can open url and read a frame within timeout."""
    ok = [False]
    def _try():
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok[0] = cap.read()[0]
        cap.release()
    t = threading.Thread(target=_try, daemon=True)
    t.start(); t.join(timeout=timeout)
    return ok[0]


def open_camera(source: CameraSource) -> Optional[cv2.VideoCapture]:
    """Open CameraSource → VideoCapture or None."""
    backend = cv2.CAP_DSHOW if (isinstance(source.device, int) and _is_windows()) else cv2.CAP_ANY
    cap = cv2.VideoCapture(source.device, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.read()[0]:
        cap.release()
        return None
    return cap


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — LENS CALIBRATION ENGINE
# ══════════════════════════════════════════════════════════════

def _make_grid_objpoints() -> np.ndarray:
    """
    3-D world coords for all 19×19 inner intersections of the 20×20 banner.
    Origin = top-left inner corner; z = 0; spacing = BANNER_CELL_CM.
    """
    objp = np.zeros((INNER_ROWS * INNER_COLS, 3), np.float32)
    objp[:, :2] = (np.mgrid[0:INNER_COLS, 0:INNER_ROWS]
                   .T.reshape(-1, 2) * BANNER_CELL_CM)
    return objp


OBJ_PTS_TMPL = _make_grid_objpoints()


@dataclass
class LensCalResult:
    rms_error:     float
    camera_matrix: np.ndarray
    dist_coeffs:   np.ndarray
    resolution:    tuple[int, int]
    n_frames:      int


class LensCalibrationEngine:
    """
    Accumulates inner-corner detections from the plain banner grid
    (NOT chessboard) and runs cv2.calibrateCamera.

    Detection pipeline per frame:
      1. Shi-Tomasi goodFeaturesToTrack  (fast, crisp lines)
      2. Hough-line intersection fallback (robust, lower contrast)
    """

    def __init__(self):
        self._objpts:    list[np.ndarray] = []
        self._imgpts:    list[np.ndarray] = []
        self._frame_sz:  Optional[tuple[int, int]] = None

    @property
    def frame_count(self) -> int:
        return len(self._objpts)

    @property
    def ready(self) -> bool:
        return self.frame_count >= MIN_CAL_FRAMES

    def reset(self):
        self._objpts.clear(); self._imgpts.clear(); self._frame_sz = None

    # ── Frame collection ──────────────────────────────────────

    def collect(self, frame: np.ndarray) -> tuple[bool, np.ndarray, str]:
        """
        Try to detect inner corners in frame.
        Returns (found, annotated_frame, status_message).
        """
        h, w = frame.shape[:2]
        self._frame_sz = (w, h)

        # Method 1: Shi-Tomasi
        found, corners, msg = self._collect_shi_tomasi(frame, w, h)
        if not found:
            # Method 2: Hough line intersections
            found, corners, msg = self._collect_hough(frame, w, h)

        annotated = frame.copy()
        if found and corners is not None:
            n = len(corners)
            obj = OBJ_PTS_TMPL[:n].copy()
            self._objpts.append(obj)
            self._imgpts.append(corners)
            for c in corners:
                cv2.circle(annotated, (int(c[0][0]), int(c[0][1])),
                           4, (0, 255, 100), -1)
            cv2.putText(annotated,
                        f"✓ {n} corners  frame {self.frame_count}/{REC_CAL_FRAMES}",
                        (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 100), 2)
            # Progress bar
            bar_w = int(w * 0.45)
            filled = int(bar_w * min(self.frame_count, REC_CAL_FRAMES) / REC_CAL_FRAMES)
            cv2.rectangle(annotated, (20, 52), (20 + bar_w, 68), (50, 50, 50), -1)
            cv2.rectangle(annotated, (20, 52), (20 + filled, 68), (0, 200, 80), -1)
        else:
            cv2.putText(annotated, f"Not detected: {msg}",
                        (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2)
        return found, annotated, msg

    def _collect_shi_tomasi(self, frame, w, h):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        corners = cv2.goodFeaturesToTrack(
            blurred, maxCorners=450, qualityLevel=0.01,
            minDistance=max(8, min(w, h) // 30), blockSize=7)
        if corners is None or len(corners) < INNER_COLS * INNER_ROWS * 0.4:
            n = len(corners) if corners is not None else 0
            return False, None, f"Shi-Tomasi: only {n} corners"
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), crit)
        arranged = self._arrange_grid(corners, w, h)
        if arranged is None:
            return False, None, "Shi-Tomasi: could not arrange into grid"
        return True, arranged, f"Shi-Tomasi: {len(arranged)} corners"

    def _collect_hough(self, frame, w, h):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 20, 80)
        min_line = int(min(w, h) * 0.10)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=40, minLineLength=min_line, maxLineGap=15)
        if lines is None or len(lines) < 4:
            return False, None, "Hough: not enough lines"

        h_segs, v_segs = [], []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if ang < 25 or ang > 155: h_segs.append((x1, y1, x2, y2))
            elif 65 < ang < 115:      v_segs.append((x1, y1, x2, y2))

        h_segs = self._cluster_lines(h_segs, "h", w, h)
        v_segs = self._cluster_lines(v_segs, "v", w, h)
        if len(h_segs) < 4 or len(v_segs) < 4:
            return False, None, f"Hough: H={len(h_segs)} V={len(v_segs)}"

        pts = []
        for hs in h_segs:
            for vs in v_segs:
                pt = self._seg_intersect(hs, vs)
                if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                    pts.append([[pt[0], pt[1]]])
        if len(pts) < INNER_COLS * INNER_ROWS * 0.35:
            return False, None, f"Hough: {len(pts)} intersections in frame"

        corners = np.array(pts, dtype=np.float32)
        arranged = self._arrange_grid(corners, w, h)
        if arranged is None:
            return False, None, "Hough: cannot arrange intersections"
        return True, arranged, f"Hough: {len(arranged)} corners"

    @staticmethod
    def _cluster_lines(segs, axis, w, h):
        if not segs: return []
        span = h if axis == "h" else w
        tol  = span * 0.03
        key  = (lambda s: (s[1]+s[3])/2) if axis=="h" else (lambda s: (s[0]+s[2])/2)
        segs_s = sorted(segs, key=key)
        clusters = [[segs_s[0]]]
        for seg in segs_s[1:]:
            if abs(key(seg) - key(clusters[-1][-1])) < tol:
                clusters[-1].append(seg)
            else:
                clusters.append([seg])
        return [max(c, key=lambda s: math.hypot(s[2]-s[0], s[3]-s[1]))
                for c in clusters]

    @staticmethod
    def _seg_intersect(s1, s2) -> Optional[tuple[float, float]]:
        x1,y1,x2,y2 = s1; x3,y3,x4,y4 = s2
        d = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
        if abs(d) < 1e-6: return None
        t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/d
        return x1+t*(x2-x1), y1+t*(y2-y1)

    @staticmethod
    def _arrange_grid(corners, w, h) -> Optional[np.ndarray]:
        pts = corners.reshape(-1, 2)
        if len(pts) < 4: return None
        ys = sorted(pts[:, 1].tolist())
        diffs = [ys[i+1]-ys[i] for i in range(len(ys)-1) if ys[i+1]-ys[i] > 2]
        if not diffs: return None
        row_h  = float(np.median(diffs)) * 0.8
        sorted_pts = sorted(pts.tolist(), key=lambda p: p[1])
        rows: list[list] = [[sorted_pts[0]]]
        for pt in sorted_pts[1:]:
            if abs(pt[1] - rows[-1][0][1]) < row_h:
                rows[-1].append(pt)
            else:
                rows.append([pt])
        rows = [sorted(r, key=lambda p: p[0]) for r in rows]
        flat = [p for row in rows for p in row]
        return np.array(flat, dtype=np.float32).reshape(-1, 1, 2)

    def finish(self) -> Optional[LensCalResult]:
        if self.frame_count < MIN_CAL_FRAMES or not self._frame_sz:
            return None
        rms, mtx, dist, _, _ = cv2.calibrateCamera(
            self._objpts, self._imgpts, self._frame_sz, None, None)
        return LensCalResult(
            rms_error=float(rms), camera_matrix=mtx,
            dist_coeffs=dist, resolution=self._frame_sz,
            n_frames=self.frame_count)


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — BANNER BOUNDARY DETECTION + MAPPING
# ══════════════════════════════════════════════════════════════

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return (4,2) float32 in [TL, TR, BR, BL] order."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s   = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def auto_detect_banner(
    frame: np.ndarray,
    min_frac: float = 0.10,
) -> tuple[bool, Optional[np.ndarray], str, np.ndarray]:
    """
    Try contour detection then Hough-line detection to find the 4 banner
    corners in *frame*.

    Returns (success, corners_px (4,2) float32, method_label, annotated_frame).
    corners_px order: [TL, TR, BR, BL].
    """
    h, w = frame.shape[:2]
    min_area = w * h * min_frac

    # ── Stage 1: Contour ──────────────────────────────────────
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh  = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 4)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    conts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None; best_area = 0.0
    for cnt in conts:
        area = cv2.contourArea(cnt)
        if area < min_area: continue
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_quad = approx.reshape(4, 2).astype(np.float32)
            best_area = area

    if best_quad is not None:
        ordered   = _order_corners(best_quad)
        annotated = _draw_quad(frame.copy(), ordered, "Auto (contour)")
        return True, ordered, "contour", annotated

    # ── Stage 2: Hough lines ──────────────────────────────────
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    min_line = int(min(w, h) * 0.20)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=80, minLineLength=min_line, maxLineGap=20)

    if lines is not None and len(lines) >= 4:
        def _rho_theta(ln):
            x1,y1,x2,y2 = ln[0]
            theta = math.atan2(y2-y1, x2-x1)
            return math.cos(theta)*x1 + math.sin(theta)*y1, theta

        rt_all = [_rho_theta(l) for l in lines]
        h_rt   = [rt for rt in rt_all if abs(rt[1]) < np.pi/4 or abs(rt[1]) > 3*np.pi/4]
        v_rt   = [rt for rt in rt_all if np.pi/4 <= abs(rt[1]) <= 3*np.pi/4]

        if len(h_rt) >= 2 and len(v_rt) >= 2:
            h_s = sorted(h_rt, key=lambda r: abs(r[0]))
            v_s = sorted(v_rt, key=lambda r: abs(r[0]))
            top, bot, left, right = h_s[0], h_s[-1], v_s[0], v_s[-1]

            def _intersect_rt(a, b):
                A = np.array([[math.cos(a[1]), math.sin(a[1])],
                              [math.cos(b[1]), math.sin(b[1])]])
                bv = np.array([a[0], b[0]])
                det = A[0,0]*A[1,1]-A[0,1]*A[1,0]
                if abs(det) < 1e-6: return None
                x=(bv[0]*A[1,1]-bv[1]*A[0,1])/det
                y=(A[0,0]*bv[1]-A[1,0]*bv[0])/det
                return float(x), float(y)

            corners_list = [_intersect_rt(top,left), _intersect_rt(top,right),
                            _intersect_rt(bot,right), _intersect_rt(bot,left)]
            if all(c is not None for c in corners_list):
                corners = np.array(corners_list, dtype=np.float32)
                if cv2.contourArea(corners) >= min_area:
                    ordered   = _order_corners(corners)
                    annotated = _draw_quad(frame.copy(), ordered, "Auto (Hough)")
                    return True, ordered, "hough", annotated

    annotated = frame.copy()
    cv2.putText(annotated, "Auto detection failed — use manual mode",
                (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 60, 220), 2)
    return False, None, "failed", annotated


def compute_banner_mapping(
    frame:         np.ndarray,
    corners_px:    np.ndarray,    # (4,2) float32 TL/TR/BR/BL
    camera_matrix: np.ndarray,
    dist_coeffs:   np.ndarray,
) -> dict[str, Any]:
    """
    Given 4 detected corners, compute:
      cm_per_pixel, px_per_cell, anchor_px, anchor_cell, homography.

    Homography H maps world_cm → image_px.
    World origin (0,0) = bottom-left = banner cell (0,0).
    col increases right, row increases up.
    In image space row 0 = bottom (largest y) because image y is flipped.

    Returns a plain dict with all mapping fields.
    """
    # Undistort detected corners
    und = cv2.undistortPoints(
        corners_px.reshape(-1, 1, 2),
        camera_matrix, dist_coeffs, P=camera_matrix
    ).reshape(-1, 2).astype(np.float32)

    # World corners [TL, TR, BR, BL] in cm (y=0 at bottom-left)
    world = np.array([
        [0,          BANNER_H_CM],
        [BANNER_W_CM, BANNER_H_CM],
        [BANNER_W_CM, 0          ],
        [0,          0           ],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(world, und)

    # cm_per_pixel from average side length
    sides = [
        np.linalg.norm(und[1]-und[0]),
        np.linalg.norm(und[2]-und[1]),
        np.linalg.norm(und[3]-und[2]),
        np.linalg.norm(und[0]-und[3]),
    ]
    avg_px       = float(np.mean(sides))
    cm_per_pixel = BANNER_W_CM / avg_px
    px_per_cell  = BANNER_CELL_CM / cm_per_pixel

    # anchor_px = pixel centre of banner cell (0,0) (bottom-left)
    half = BANNER_CELL_CM / 2.0
    src  = np.array([[[half, half]]], dtype=np.float32)
    dst  = cv2.perspectiveTransform(src, H)
    anchor_px = (int(round(float(dst[0,0,0]))),
                 int(round(float(dst[0,0,1]))))

    return {
        "cm_per_pixel":      cm_per_pixel,
        "px_per_cell":       px_per_cell,
        "anchor_px":         anchor_px,
        "anchor_cell":       (0, 0),
        "homography":        H,
        "banner_corners_px": und,
    }


def _draw_quad(frame, corners, label=""):
    pts = corners.astype(np.int32)
    names  = ["TL","TR","BR","BL"]
    colors = [(0,220,255),(100,255,0),(0,100,255),(255,180,0)]
    for i in range(4):
        cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1)%4]), (0,220,0), 2)
    for i,(pt,name,col) in enumerate(zip(pts,names,colors)):
        cv2.circle(frame, tuple(pt), 10, col, -1)
        cv2.putText(frame, name, (pt[0]+12, pt[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
    if label:
        cv2.putText(frame, label, (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,0), 2)
    return frame


def draw_grid_overlay(
    frame: np.ndarray,
    H: np.ndarray,
    anchor_px: tuple[int,int],
    px_per_cell: float,
) -> np.ndarray:
    """Project all 20×20 cell borders through H onto the frame."""
    h, w = frame.shape[:2]

    def _warp(x_cm, y_cm):
        pt = cv2.perspectiveTransform(
            np.array([[[x_cm, y_cm]]], dtype=np.float32), H)
        return int(pt[0,0,0]), int(pt[0,0,1])

    for col in range(BANNER_COLS + 1):
        x = col * BANNER_CELL_CM
        p0 = _warp(x, 0); p1 = _warp(x, BANNER_H_CM)
        if (0<=p0[0]<w or 0<=p1[0]<w):
            cv2.line(frame, p0, p1, (0, 180, 80), 1)
    for row in range(BANNER_ROWS + 1):
        y = row * BANNER_CELL_CM
        p0 = _warp(0, y); p1 = _warp(BANNER_W_CM, y)
        if (0<=p0[1]<h or 0<=p1[1]<h):
            cv2.line(frame, p0, p1, (0, 180, 80), 1)

    # cell (0,0) anchor
    ax, ay = anchor_px
    r = max(6, int(px_per_cell * 0.25))
    cv2.circle(frame, (ax, ay), r, (0, 60, 255), -1)
    cv2.circle(frame, (ax, ay), r+3, (255,255,0), 2)
    cv2.putText(frame, "cell(0,0)", (ax+r+5, ay+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
    return frame


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — SAVE / LOAD
# ══════════════════════════════════════════════════════════════

def save_calibration(path: Path, data: dict[str, Any]) -> None:
    """Save a calibration profile to *path* (.npz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kw: dict[str, Any] = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            kw[key] = val
        elif isinstance(val, (list, tuple)):
            kw[key] = np.array(val)
        else:
            kw[key] = np.array(val)
    np.savez_compressed(str(path), **kw)


def load_calibration(path: Path) -> Optional[dict[str, Any]]:
    """Load a calibration profile from *path*. Returns plain dict or None."""
    if not path.exists():
        return None
    raw = np.load(str(path), allow_pickle=True)
    out: dict[str, Any] = {}
    for key in raw.files:
        val = raw[key]
        # Unwrap 0-d arrays
        if val.ndim == 0:
            item = val.item()
            out[key] = item
        elif key in ("camera_matrix", "dist_coeffs", "homography",
                     "banner_corners_px"):
            out[key] = val
        elif key in ("resolution", "anchor_px", "anchor_cell"):
            out[key] = tuple(int(x) for x in val.tolist())
        elif key in ("cm_per_pixel", "px_per_cell", "rms_error",
                     "height_cm", "arena_w_cm", "arena_h_cm",
                     "cell_size_cm", "tilt_deg"):
            out[key] = float(val)
        elif key in ("grid_cols", "grid_rows", "n_frames"):
            out[key] = int(val)
        else:
            try:
                out[key] = str(val)
            except Exception:
                out[key] = val
    return out


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — RUNTIME HELPER (for navigation code)
# ══════════════════════════════════════════════════════════════

def load_calibration_profile(path: str | Path) -> Optional[dict[str, Any]]:
    """
    Load a saved .npz calibration file and return a ready-to-use dict:

        {
          "camera_matrix":  np.ndarray,
          "dist_coeffs":    np.ndarray,
          "resolution":     (w, h),
          "rms_error":      float,
          "cm_per_pixel":   float,       ← if banner was mapped
          "anchor_px":      (x, y),      ← if banner was mapped
          "anchor_cell":    (col, row),  ← always (0, 0)
          "homography":     np.ndarray,  ← if banner was mapped
          ...all other saved fields...
        }

    Usage in Autobot_navigation_v4.py:
        from camera_calibration import load_calibration_profile
        prof = load_calibration_profile("my_setup.npz")
        self.banner.cm_per_pixel = prof["cm_per_pixel"]
        self.banner.anchor_px    = prof["anchor_px"]
        self.banner.anchor_cell  = prof["anchor_cell"]
        mtx  = prof["camera_matrix"]
        dist = prof["dist_coeffs"]
        frame = cv2.undistort(raw_frame, mtx, dist, None, mtx)
    """
    return load_calibration(Path(path))


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — TKINTER GUI
# ══════════════════════════════════════════════════════════════

def _apply_style(root: tk.Tk) -> None:
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=BG, foreground=FG,
                fieldbackground=BG2, troughcolor=BG2,
                borderwidth=0, relief="flat")
    s.configure("TFrame",       background=BG)
    s.configure("TLabelframe",  background=BG2, foreground=ACCENT,
                borderwidth=1, relief="solid")
    s.configure("TLabelframe.Label", background=BG2, foreground=ACCENT,
                font=HEADF)
    s.configure("TButton",      background=BG3, foreground=FG,
                padding=(8, 5), font=LABELF)
    s.map("TButton",
          background=[("active", ACCENT), ("pressed", BG2)],
          foreground=[("active", BG)])
    s.configure("Accent.TButton", background=ACCENT, foreground=BG,
                font=("Segoe UI", 9, "bold"))
    s.map("Accent.TButton",
          background=[("active", "#00ffcc"), ("pressed", "#009977")])
    s.configure("TLabel",       background=BG, foreground=FG,  font=LABELF)
    s.configure("Dim.TLabel",   background=BG, foreground=FG2, font=LABELF)
    s.configure("Mono.TLabel",  background=BG, foreground=ACCENT, font=MONO)
    s.configure("TEntry",       fieldbackground=BG2, foreground=FG,
                insertcolor=ACCENT, font=LABELF)
    s.configure("TSpinbox",     fieldbackground=BG2, foreground=FG,
                arrowcolor=ACCENT, font=LABELF)
    s.configure("TScrollbar",   background=BG2, arrowcolor=ACCENT)
    s.configure("TProgressbar", troughcolor=BG2, background=ACCENT)
    s.configure("TSeparator",   background=BG3)
    s.configure("Treeview",     background=BG2, foreground=FG,
                fieldbackground=BG2, rowheight=22)
    s.configure("Treeview.Heading", background=BG3, foreground=ACCENT,
                font=("Segoe UI", 9, "bold"))
    s.map("Treeview",
          background=[("selected", ACCENT)],
          foreground=[("selected", BG)])


class CalibrationGUI:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Camera Calibration  ·  Banner-Grid Robot System")
        self.root.geometry("1440x860")
        self.root.configure(bg=BG)
        self.root.state("zoomed")
        _apply_style(root)

        # ── Camera state ──────────────────────────────────────
        self.cap:     Optional[cv2.VideoCapture] = None
        self.source:  Optional[CameraSource]     = None
        self.sources: list[CameraSource]         = []
        self._live:   Optional[np.ndarray]       = None
        self._is_live = False

        # ── Calibration state ─────────────────────────────────
        self.engine    = LensCalibrationEngine()
        self.lens_cal: Optional[LensCalResult] = None

        # ── Banner / mapping state ────────────────────────────
        self._corners:     Optional[np.ndarray] = None  # (4,2) float32
        self._det_method   = ""
        self._mapping:     Optional[dict]       = None
        self._show_overlay = False

        # Manual corner picking
        self._manual_mode    = False
        self._manual_corners: list[tuple[int,int]] = []

        # ── Canvas overlay state ──────────────────────────────
        self._overlay_frame: Optional[np.ndarray] = None

        # ── Tkinter variables ─────────────────────────────────
        self._v_droid_ip   = tk.StringVar(value="192.168.1.100")
        self._v_droid_port = tk.StringVar(value="4747")
        self._v_ipcam_url  = tk.StringVar(value="http://")
        self._v_cam_name   = tk.StringVar(value="")
        self._v_filename   = tk.StringVar(value="my_calibration")
        self._v_save_dir   = tk.StringVar(value=str(DEFAULT_SAVE_DIR))
        self._v_height     = tk.DoubleVar(value=120.0)
        self._v_arena_w    = tk.DoubleVar(value=300.0)
        self._v_arena_h    = tk.DoubleVar(value=300.0)
        self._v_tilt       = tk.DoubleVar(value=0.0)
        self._v_facing     = tk.StringVar(value="overhead")
        self._v_notes      = tk.StringVar(value="")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        threading.Thread(target=self._scan_webcams_bg, daemon=True).start()
        self.root.after(FRAME_MS, self._live_loop)

    # ─────────────────────────────────────────────────────────
    #  UI  BUILD
    # ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Header ────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=BG3, height=46)
        hdr.pack(fill=tk.X, side=tk.TOP)
        tk.Label(hdr, text="⬡  CAMERA CALIBRATION",
                 font=("Segoe UI", 13, "bold"),
                 bg=BG3, fg=ACCENT).pack(side=tk.LEFT, padx=18, pady=10)
        self._status_lbl = tk.Label(hdr, text="Ready — connect a camera",
                                    font=LABELF, bg=BG3, fg=FG2)
        self._status_lbl.pack(side=tk.RIGHT, padx=18)

        # ── Body: left panel + right canvas ───────────────────
        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
        body.columnconfigure(0, weight=0, minsize=LEFT_W)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Left: scrollable panel
        lc = tk.Frame(body, bg=BG, width=LEFT_W)
        lc.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        lc.grid_propagate(False)
        lcan = tk.Canvas(lc, bg=BG, highlightthickness=0, width=LEFT_W-20)
        lscr = ttk.Scrollbar(lc, orient="vertical", command=lcan.yview)
        lf   = ttk.Frame(lcan)
        lf.bind("<Configure>",
                lambda e: lcan.configure(scrollregion=lcan.bbox("all")))
        lcan.create_window((0, 0), window=lf, anchor="nw")
        lcan.configure(yscrollcommand=lscr.set)
        lcan.bind_all("<MouseWheel>",
                      lambda e: lcan.yview_scroll(int(-e.delta/120), "units"))
        lcan.pack(side="left", fill="both", expand=True)
        lscr.pack(side="right", fill="y")

        # Right: canvas
        rc = ttk.LabelFrame(body, text="Live Preview")
        rc.grid(row=0, column=1, sticky="nsew")
        rc.rowconfigure(0, weight=1)
        rc.columnconfigure(0, weight=1)
        self._canvas = tk.Canvas(rc, bg="#080810", cursor="crosshair",
                                 highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._hud = tk.Label(rc, text="", font=MONO,
                             bg=BG3, fg=ACCENT, anchor=tk.W, padx=8, pady=3)
        self._hud.grid(row=1, column=0, sticky="ew")

        left = lf
        self._build_section_camera(left)
        self._build_section_lens(left)
        self._build_section_banner(left)
        self._build_section_save(left)
        self._build_section_profiles(left)

    # ── Section helpers ───────────────────────────────────────

    def _sep(self, parent): ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=5)
    def _btn(self, parent, text, cmd, style="TButton"):
        ttk.Button(parent, text=text, command=cmd, style=style).pack(fill=tk.X, pady=2)
    def _lbl(self, parent, text, style="TLabel"):
        ttk.Label(parent, text=text, style=style, wraplength=330).pack(anchor=tk.W)
    def _entry(self, parent, label, var, w=18):
        r = ttk.Frame(parent); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text=label, width=20, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=var, width=w).pack(side=tk.LEFT, padx=4)
    def _spin(self, parent, label, var, frm, to, inc=1.0):
        r = ttk.Frame(parent); r.pack(fill=tk.X, pady=1)
        ttk.Label(r, text=label, width=20, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(r, textvariable=var, from_=frm, to=to,
                    increment=inc, width=10).pack(side=tk.LEFT, padx=4)
    def _mono(self, parent, text=""):
        l = ttk.Label(parent, text=text, style="Mono.TLabel", wraplength=330)
        l.pack(anchor=tk.W, pady=1)
        return l

    # ═══════════════════════════════════════════════════════════
    #  SECTION: CAMERA SOURCE
    # ═══════════════════════════════════════════════════════════

    def _build_section_camera(self, left):
        sf = ttk.LabelFrame(left, text="📷  Camera Source", padding=8)
        sf.pack(fill=tk.X, pady=4)

        # Webcam listbox
        self._cam_lb = tk.Listbox(sf, height=4,
                                  bg=BG2, fg=FG, selectbackground=ACCENT,
                                  selectforeground=BG, font=LABELF,
                                  borderwidth=0, highlightthickness=0)
        self._cam_lb.pack(fill=tk.X, pady=(0, 4))
        bf = ttk.Frame(sf); bf.pack(fill=tk.X, pady=2)
        ttk.Button(bf, text="🔍 Scan Webcams",
                   command=lambda: threading.Thread(
                       target=self._scan_webcams_bg, daemon=True).start()
                   ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(bf, text="▶ Connect",
                   command=self._connect_selected,
                   style="Accent.TButton").pack(side=tk.LEFT)

        self._sep(sf)
        ttk.Label(sf, text="DroidCam / Phone", font=("Segoe UI",8,"bold"),
                  background=BG2, foreground=FG2).pack(anchor=tk.W)
        r = ttk.Frame(sf); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="IP:").pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=self._v_droid_ip, width=14).pack(side=tk.LEFT, padx=4)
        ttk.Label(r, text="Port:").pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=self._v_droid_port, width=6).pack(side=tk.LEFT, padx=4)
        self._btn(sf, "▶ Connect DroidCam", self._connect_droidcam, "Accent.TButton")

        self._sep(sf)
        ttk.Label(sf, text="IP Camera URL", font=("Segoe UI",8,"bold"),
                  background=BG2, foreground=FG2).pack(anchor=tk.W)
        ttk.Entry(sf, textvariable=self._v_ipcam_url).pack(fill=tk.X, pady=2)
        self._btn(sf, "▶ Connect IP Camera", self._connect_ipcam, "Accent.TButton")

        self._sep(sf)
        self._entry(sf, "Camera label:", self._v_cam_name)
        self._btn(sf, "⏹ Disconnect", self._disconnect)

    # ═══════════════════════════════════════════════════════════
    #  SECTION: LENS CALIBRATION
    # ═══════════════════════════════════════════════════════════

    def _build_section_lens(self, left):
        sf = ttk.LabelFrame(left, text="🔧  Lens Calibration", padding=8)
        sf.pack(fill=tk.X, pady=4)

        for line in [
            "1. Point camera at the banner grid.",
            "2. Capture 10+ frames from different angles.",
            "3. Press Run Calibration.",
        ]:
            self._lbl(sf, line, "Dim.TLabel")

        self._sep(sf)
        self._frames_lbl = self._mono(sf, f"Frames: 0 / {REC_CAL_FRAMES}")
        self._prog = ttk.Progressbar(sf, maximum=REC_CAL_FRAMES, length=300)
        self._prog.pack(fill=tk.X, pady=3)
        self._rms_lbl = self._mono(sf, "RMS: –")

        self._btn(sf, "📸  Capture Frame",      self._capture_frame)
        self._btn(sf, "⚙️  Run Calibration",    self._run_calibration, "Accent.TButton")
        self._btn(sf, "🔄  Reset Frames",        self._reset_lens)

        self._lens_lbl = ttk.Label(sf, text="", font=LABELF,
                                   background=BG2, foreground=FG2,
                                   wraplength=330, justify=tk.LEFT)
        self._lens_lbl.pack(anchor=tk.W, pady=3)

    # ═══════════════════════════════════════════════════════════
    #  SECTION: BANNER MAPPING
    # ═══════════════════════════════════════════════════════════

    def _build_section_banner(self, left):
        sf = ttk.LabelFrame(left, text="📐  Banner Mapping", padding=8)
        sf.pack(fill=tk.X, pady=4)

        self._lbl(sf, "Requires lens calibration above first.", "Dim.TLabel")
        self._sep(sf)

        self._btn(sf, "🔍  Auto Detect Banner",          self._auto_detect)
        self._sep(sf)
        self._lbl(sf, "If auto fails — manual mode:", "Dim.TLabel")
        self._btn(sf, "✏️  Start Manual (click TL→TR→BR→BL)", self._start_manual)
        self._manual_confirm_btn = ttk.Button(
            sf, text="✅  Confirm 4 Corners",
            command=self._confirm_manual, state=tk.DISABLED)
        self._manual_confirm_btn.pack(fill=tk.X, pady=2)
        self._btn(sf, "🗑  Clear Manual Points", self._clear_manual)
        self._sep(sf)
        self._btn(sf, "📐  Compute Mapping",
                  self._compute_mapping, "Accent.TButton")
        self._btn(sf, "🗺️  Toggle Grid Overlay", self._toggle_overlay)

        self._manual_lbl  = self._mono(sf, "Manual points: 0 / 4")

        # Results
        self._map_lbl = ttk.Label(sf, text="–", font=MONO,
                                  background=BG2, foreground=ACCENT,
                                  wraplength=330, justify=tk.LEFT)
        self._map_lbl.pack(anchor=tk.W, pady=3)

    # ═══════════════════════════════════════════════════════════
    #  SECTION: SAVE / LOAD
    # ═══════════════════════════════════════════════════════════

    def _build_section_save(self, left):
        sf = ttk.LabelFrame(left, text="💾  Save / Load Profile", padding=8)
        sf.pack(fill=tk.X, pady=4)

        # File name + folder
        self._entry(sf, "File name (.npz):", self._v_filename)
        r = ttk.Frame(sf); r.pack(fill=tk.X, pady=2)
        ttk.Label(r, text="Save folder:", width=13, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Entry(r, textvariable=self._v_save_dir, width=18).pack(side=tk.LEFT, padx=4)
        ttk.Button(r, text="📁", command=self._browse_folder,
                   width=3).pack(side=tk.LEFT)

        self._sep(sf)
        ttk.Label(sf, text="Physical Setup", font=("Segoe UI",8,"bold"),
                  background=BG2, foreground=FG2).pack(anchor=tk.W, pady=(2,0))
        self._spin(sf, "Camera height (cm):", self._v_height, 10, 500, 5)
        self._spin(sf, "Arena width (cm):",   self._v_arena_w, 10, 1000, 10)
        self._spin(sf, "Arena height (cm):",  self._v_arena_h, 10, 1000, 10)
        self._spin(sf, "Tilt angle (deg):",   self._v_tilt, 0, 90, 0.5)
        self._entry(sf, "Facing direction:",  self._v_facing)
        self._entry(sf, "Notes:",             self._v_notes)

        self._sep(sf)
        self._save_status = ttk.Label(sf, font=MONO,
                                      background=BG2, foreground=FG2,
                                      text="Lens: ✗   Mapping: ✗",
                                      wraplength=330)
        self._save_status.pack(anchor=tk.W, pady=2)
        self._btn(sf, "💾  Save Profile", self._save_profile, "Accent.TButton")
        self._save_lbl = self._mono(sf)

        self._sep(sf)
        self._btn(sf, "📂  Load Profile (.npz)", self._load_profile)
        self._load_lbl = ttk.Label(sf, text="", font=MONO,
                                   background=BG2, foreground=ACCENT,
                                   wraplength=330, justify=tk.LEFT)
        self._load_lbl.pack(anchor=tk.W, pady=2)

    # ═══════════════════════════════════════════════════════════
    #  SECTION: PROFILES BROWSER
    # ═══════════════════════════════════════════════════════════

    def _build_section_profiles(self, left):
        sf = ttk.LabelFrame(left, text="📂  Saved Profiles", padding=8)
        sf.pack(fill=tk.X, pady=4)

        bf = ttk.Frame(sf); bf.pack(fill=tk.X, pady=(0,4))
        ttk.Button(bf, text="🔄 Refresh",
                   command=self._refresh_profiles).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(bf, text="📋 Load Selected",
                   command=self._load_selected).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(bf, text="🗑 Delete",
                   command=self._delete_selected).pack(side=tk.LEFT)

        cols = ("file", "lens", "mapping", "height")
        self._tree = ttk.Treeview(sf, columns=cols, show="headings", height=6)
        for col, w, lbl in [("file",100,"File"),("lens",55,"RMS"),
                              ("mapping",65,"cm/px"),("height",55,"H cm")]:
            self._tree.heading(col, text=lbl)
            self._tree.column(col, width=w, anchor=tk.W)
        self._tree.pack(fill=tk.X)
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        self._detail_lbl = ttk.Label(sf, text="", font=MONO,
                                     background=BG2, foreground=ACCENT,
                                     wraplength=330, justify=tk.LEFT)
        self._detail_lbl.pack(anchor=tk.W, pady=4)

        self._refresh_profiles()

    # ─────────────────────────────────────────────────────────
    #  LIVE  LOOP
    # ─────────────────────────────────────────────────────────

    def _live_loop(self) -> None:
        if self.cap and self._is_live:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._live = frame

        frame = self._build_display_frame()
        if frame is not None:
            self._show(frame)

        self.root.after(FRAME_MS, self._live_loop)

    def _build_display_frame(self) -> Optional[np.ndarray]:
        """Priority: overlay → detection annotated → live."""
        if self._overlay_frame is not None and self._show_overlay:
            base = self._overlay_frame.copy()
        elif self._overlay_frame is not None and not self._show_overlay:
            base = (self._live.copy() if self._live is not None
                    else self._overlay_frame.copy())
        elif self._live is not None:
            base = self._live.copy()
        else:
            return None

        # Draw manual corners on top
        labels = ["TL","TR","BR","BL"]
        colors = [(0,220,255),(100,255,0),(0,100,255),(255,180,0)]
        for i, (px, py) in enumerate(self._manual_corners):
            cv2.circle(base, (px, py), 10, colors[i], -1)
            cv2.putText(base, labels[i], (px+12, py-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors[i], 2)

        return base

    def _show(self, frame: np.ndarray) -> None:
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10: return
        img = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self._canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self._canvas.image = imgtk

    # ─────────────────────────────────────────────────────────
    #  CAMERA  ACTIONS
    # ─────────────────────────────────────────────────────────

    def _scan_webcams_bg(self):
        self._set_status("Scanning webcams…")
        found = scan_webcams(8)
        self.sources = found
        self._cam_lb.delete(0, tk.END)
        for s in found:
            self._cam_lb.insert(tk.END, s.label)
        if found:
            self._cam_lb.select_set(0)
            self._set_status(f"Found {len(found)} webcam(s)")
        else:
            self._set_status("No webcams found")

    def _connect_selected(self):
        sel = self._cam_lb.curselection()
        if not sel:
            messagebox.showwarning("No selection", "Select a webcam first")
            return
        idx = sel[0]
        if idx < len(self.sources):
            self._connect(self.sources[idx])

    def _connect_droidcam(self):
        ip   = self._v_droid_ip.get().strip()
        port = self._v_droid_port.get().strip()
        url  = f"http://{ip}:{port}/video"
        self._set_status(f"Probing DroidCam {ip}:{port}…")
        self.root.update()
        if not probe_url(url):
            messagebox.showerror("Failed",
                f"Cannot reach {url}\nCheck IP, port and DroidCam app.")
            self._set_status("DroidCam failed")
            return
        self._connect(CameraSource(f"DroidCam {ip}:{port}", url, "droidcam"))

    def _connect_ipcam(self):
        url = self._v_ipcam_url.get().strip()
        if not url or url == "http://":
            messagebox.showwarning("No URL", "Enter a valid stream URL")
            return
        self._set_status(f"Probing {url}…")
        self.root.update()
        if not probe_url(url):
            messagebox.showerror("Failed", f"Cannot open: {url}")
            return
        self._connect(CameraSource(f"IP Cam {url}", url, "ipcam"))

    def _connect(self, source: CameraSource):
        self._disconnect()
        cap = open_camera(source)
        if cap is None:
            messagebox.showerror("Error", f"Cannot open: {source.label}")
            return
        self.cap = cap; self.source = source; self._is_live = True
        if not self._v_cam_name.get():
            self._v_cam_name.set(
                re.sub(r"[^\w]", "_", source.label)[:40])
        self._set_status(f"Connected: {source.label}")

    def _disconnect(self):
        self._is_live = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.source = None
        self._set_status("Disconnected")

    # ─────────────────────────────────────────────────────────
    #  LENS  CALIBRATION  ACTIONS
    # ─────────────────────────────────────────────────────────

    def _capture_frame(self):
        if self._live is None:
            messagebox.showwarning("No frame", "Connect a camera first")
            return
        found, annotated, msg = self.engine.collect(self._live.copy())
        # Show annotated briefly then revert to live
        self._overlay_frame = annotated
        self._show_overlay  = True
        self.root.after(1200, self._clear_overlay)
        self._update_lens_ui()
        self._set_status(f"{'✓ Captured' if found else '✗ Failed'}: {msg}")
        if not found:
            messagebox.showinfo(
                "Not detected", f"{msg}\n\nTips:\n"
                "• Ensure even lighting over the banner\n"
                "• Banner should fill ≥40% of frame\n"
                "• Try moving camera slightly closer")

    def _clear_overlay(self):
        self._show_overlay = False

    def _run_calibration(self):
        if self.engine.frame_count < MIN_CAL_FRAMES:
            messagebox.showwarning(
                "Too few frames",
                f"Need at least {MIN_CAL_FRAMES} frames "
                f"(have {self.engine.frame_count}).\n"
                f"Recommended: {REC_CAL_FRAMES}+")
            return
        self._set_status("Running calibrateCamera…"); self.root.update()
        result = self.engine.finish()
        if result is None:
            messagebox.showerror("Error", "Calibration failed")
            return
        self.lens_cal = result
        rms = result.rms_error
        quality = ("Excellent" if rms < 0.5 else
                   "Good"      if rms < 1.0 else
                   "Acceptable" if rms < 2.0 else "Poor – recalibrate")
        col = GREEN if rms < 1.0 else ACCENT2 if rms < 2.0 else RED
        self._rms_lbl.config(
            text=f"RMS: {rms:.4f}  [{quality}]", foreground=col)
        self._lens_lbl.config(
            text=f"✓ Done  {result.n_frames} frames\n"
                 f"  Res: {result.resolution[0]}×{result.resolution[1]}",
            foreground=col)
        self._set_status(f"Calibration done — RMS={rms:.4f}")
        self._refresh_save_status()

    def _reset_lens(self):
        if messagebox.askyesno("Reset", "Discard all captured frames?"):
            self.engine.reset()
            self.lens_cal = None
            self._update_lens_ui()
            self._rms_lbl.config(text="RMS: –", foreground=ACCENT)
            self._lens_lbl.config(text="")
            self._refresh_save_status()

    def _update_lens_ui(self):
        n = self.engine.frame_count
        self._frames_lbl.config(text=f"Frames: {n} / {REC_CAL_FRAMES}")
        self._prog["value"] = min(n, REC_CAL_FRAMES)

    # ─────────────────────────────────────────────────────────
    #  BANNER  DETECTION  ACTIONS
    # ─────────────────────────────────────────────────────────

    def _auto_detect(self):
        if self.lens_cal is None:
            messagebox.showwarning("No calibration",
                                   "Run lens calibration first")
            return
        frame = self._live
        if frame is None:
            messagebox.showwarning("No frame", "Connect a camera first")
            return
        und = cv2.undistort(frame,
                            self.lens_cal.camera_matrix,
                            self.lens_cal.dist_coeffs,
                            None, self.lens_cal.camera_matrix)
        self._set_status("Auto-detecting banner…"); self.root.update()
        success, corners, method, annotated = auto_detect_banner(und)
        self._overlay_frame = annotated
        self._show_overlay  = True
        if success:
            self._corners    = corners
            self._det_method = method
            self._mapping    = None   # invalidate old mapping
            self._set_status(f"Banner detected ({method}) — compute mapping next")
            messagebox.showinfo("Detected",
                f"Method: {method}\nNow press 'Compute Mapping'.")
        else:
            self._set_status("Auto detection failed — try manual mode")
            messagebox.showwarning("Failed",
                "Could not find banner boundary automatically.\n"
                "Use manual mode: click TL→TR→BR→BL on preview.")

    def _start_manual(self):
        self._manual_mode    = True
        self._manual_corners = []
        self._corners        = None
        self._mapping        = None
        self._show_overlay   = False
        self._manual_confirm_btn.config(state=tk.DISABLED)
        self._manual_lbl.config(text="Manual: click TL on preview")
        self._set_status("Manual mode — click TL on the live preview")

    def _on_canvas_click(self, event: tk.Event):
        if not self._manual_mode:
            return
        if len(self._manual_corners) >= 4:
            return
        frame = self._live
        if frame is None:
            return
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        fh, fw = frame.shape[:2]
        px = int(event.x * fw / cw)
        py = int(event.y * fh / ch)
        self._manual_corners.append((px, py))
        n = len(self._manual_corners)
        next_lbl = ["TR","BR","BL","done"][n-1] if n < 4 else "done"
        self._manual_lbl.config(
            text=f"Manual: {n}/4  (next: {next_lbl})")
        if n == 4:
            self._manual_confirm_btn.config(state=tk.NORMAL)
            self._set_status("4 corners set — press Confirm")

    def _confirm_manual(self):
        if len(self._manual_corners) != 4:
            messagebox.showwarning("Need 4 points", "Click exactly 4 corners")
            return
        pts = np.array(self._manual_corners, dtype=np.float32)
        self._corners        = _order_corners(pts)
        self._det_method     = "manual"
        self._manual_mode    = False
        self._mapping        = None
        frame = self._live or np.zeros((480, 640, 3), np.uint8)
        self._overlay_frame  = _draw_quad(frame.copy(), self._corners,
                                          "Manual corners")
        self._show_overlay   = True
        self._manual_confirm_btn.config(state=tk.DISABLED)
        self._manual_lbl.config(text="Manual: confirmed ✓")
        self._set_status("Manual corners confirmed — compute mapping now")

    def _clear_manual(self):
        self._manual_corners = []
        self._manual_mode    = False
        self._manual_lbl.config(text="Manual points: 0 / 4")
        self._show_overlay   = False

    def _compute_mapping(self):
        if self.lens_cal is None:
            messagebox.showwarning("No calibration",
                                   "Run lens calibration first")
            return
        if self._corners is None:
            messagebox.showwarning("No corners",
                                   "Detect banner (auto or manual) first")
            return
        frame = self._live
        if frame is None:
            messagebox.showwarning("No frame", "Connect a camera first")
            return
        self._set_status("Computing banner mapping…"); self.root.update()
        try:
            mapping = compute_banner_mapping(
                frame, self._corners,
                self.lens_cal.camera_matrix,
                self.lens_cal.dist_coeffs,
            )
        except Exception as exc:
            messagebox.showerror("Mapping Error", str(exc))
            return
        self._mapping = mapping

        # Build overlay frame with grid
        und = cv2.undistort(frame,
                            self.lens_cal.camera_matrix,
                            self.lens_cal.dist_coeffs,
                            None, self.lens_cal.camera_matrix)
        annotated = _draw_quad(und.copy(), self._corners, "Banner mapped")
        draw_grid_overlay(annotated,
                          mapping["homography"],
                          mapping["anchor_px"],
                          mapping["px_per_cell"])
        self._overlay_frame = annotated
        self._show_overlay  = True

        cpp = mapping["cm_per_pixel"]
        ppc = mapping["px_per_cell"]
        ap  = mapping["anchor_px"]
        self._map_lbl.config(
            text=f"cm/px    : {cpp:.6f}\n"
                 f"px/cell  : {ppc:.2f}\n"
                 f"anchor_px: {ap}\n"
                 f"anchor_cell: (0, 0)\n"
                 f"method   : {self._det_method}")
        self._hud.config(
            text=f"cm/px={cpp:.5f}  px/cell={ppc:.1f}  "
                 f"anchor={ap}  method={self._det_method}")
        self._refresh_save_status()
        self._set_status(f"Mapping ready — cm/px={cpp:.5f}  px/cell={ppc:.1f}")

    def _toggle_overlay(self):
        self._show_overlay = not self._show_overlay

    # ─────────────────────────────────────────────────────────
    #  SAVE / LOAD  ACTIONS
    # ─────────────────────────────────────────────────────────

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Choose save folder")
        if folder:
            self._v_save_dir.set(folder)

    def _refresh_save_status(self):
        l = f"{'✓' if self.lens_cal else '✗'}"
        m = f"{'✓' if self._mapping  else '✗'}"
        self._save_status.config(
            text=f"Lens cal: {l}   Banner mapping: {m}")

    def _save_profile(self):
        if self.lens_cal is None:
            messagebox.showerror("Missing", "Run lens calibration first")
            return
        fname = self._v_filename.get().strip()
        if not fname:
            messagebox.showerror("Missing", "Enter a file name")
            return
        if not fname.endswith(".npz"):
            fname += ".npz"
        folder = Path(self._v_save_dir.get().strip() or str(DEFAULT_SAVE_DIR))
        path   = folder / fname

        data: dict[str, Any] = dict(
            profile_name  = fname,
            camera_name   = self._v_cam_name.get(),
            camera_matrix = self.lens_cal.camera_matrix,
            dist_coeffs   = self.lens_cal.dist_coeffs,
            resolution    = list(self.lens_cal.resolution),
            rms_error     = self.lens_cal.rms_error,
            n_frames      = self.lens_cal.n_frames,
            height_cm     = self._v_height.get(),
            arena_w_cm    = self._v_arena_w.get(),
            arena_h_cm    = self._v_arena_h.get(),
            cell_size_cm  = BANNER_CELL_CM,
            grid_cols     = BANNER_COLS,
            grid_rows     = BANNER_ROWS,
            tilt_deg      = self._v_tilt.get(),
            facing_dir    = self._v_facing.get(),
            notes         = self._v_notes.get(),
        )
        if self._mapping:
            data.update(
                cm_per_pixel      = self._mapping["cm_per_pixel"],
                anchor_px         = list(self._mapping["anchor_px"]),
                anchor_cell       = [0, 0],
                homography        = self._mapping["homography"],
                banner_corners_px = self._mapping["banner_corners_px"],
            )
        try:
            save_calibration(path, data)
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))
            return
        self._save_lbl.config(text=f"✓ Saved:\n{path}", foreground=GREEN)
        self._set_status(f"Saved: {path}")
        self._refresh_profiles()

    def _load_profile(self):
        path = filedialog.askopenfilename(
            title="Load calibration profile",
            initialdir=self._v_save_dir.get(),
            filetypes=[("Calibration profile","*.npz"),("All","*.*")])
        if not path:
            return
        self._load_from_path(Path(path))

    def _load_from_path(self, path: Path):
        data = load_calibration(path)
        if data is None:
            messagebox.showerror("Error", f"Cannot load: {path}")
            return
        # Restore lens cal
        if "camera_matrix" in data:
            self.lens_cal = LensCalResult(
                rms_error     = data.get("rms_error", 0.0),
                camera_matrix = data["camera_matrix"],
                dist_coeffs   = data["dist_coeffs"],
                resolution    = data.get("resolution", (0,0)),
                n_frames      = data.get("n_frames", 0),
            )
            self._rms_lbl.config(
                text=f"RMS: {self.lens_cal.rms_error:.4f} (loaded)",
                foreground=ACCENT)
        # Restore mapping
        if "cm_per_pixel" in data:
            self._mapping = {
                "cm_per_pixel":      data["cm_per_pixel"],
                "px_per_cell":       data.get("px_per_cell",
                                              BANNER_CELL_CM/data["cm_per_pixel"]),
                "anchor_px":         data.get("anchor_px", (0,0)),
                "anchor_cell":       data.get("anchor_cell", (0,0)),
                "homography":        data.get("homography", np.eye(3)),
                "banner_corners_px": data.get("banner_corners_px",
                                              np.zeros((4,2),np.float32)),
            }
            cpp = self._mapping["cm_per_pixel"]
            ppc = self._mapping["px_per_cell"]
            ap  = self._mapping["anchor_px"]
            self._map_lbl.config(
                text=f"cm/px    : {cpp:.6f}\n"
                     f"px/cell  : {ppc:.2f}\n"
                     f"anchor_px: {ap}\n"
                     f"anchor_cell: (0, 0)\n"
                     f"(loaded from file)")

        # Restore UI fields
        self._v_cam_name.set(str(data.get("camera_name","")))
        self._v_filename.set(path.stem)
        self._v_save_dir.set(str(path.parent))
        self._v_height.set(float(data.get("height_cm", 120)))
        self._v_arena_w.set(float(data.get("arena_w_cm", 300)))
        self._v_arena_h.set(float(data.get("arena_h_cm", 300)))
        self._v_tilt.set(float(data.get("tilt_deg", 0)))
        self._v_facing.set(str(data.get("facing_dir","")))
        self._v_notes.set(str(data.get("notes","")))

        self._refresh_save_status()
        self._set_status(f"Loaded: {path.name}")
        self._show_load_detail(data)

    def _show_load_detail(self, data: dict):
        lines = [
            f"File     : {data.get('profile_name','')}",
            f"Camera   : {data.get('camera_name','')}",
            f"RMS      : {data.get('rms_error',0):.4f}",
            f"Res      : {data.get('resolution','-')}",
            f"Height   : {data.get('height_cm','-')} cm",
        ]
        if "cm_per_pixel" in data:
            lines += [f"cm/px    : {data['cm_per_pixel']:.6f}",
                      f"px/cell  : {BANNER_CELL_CM/data['cm_per_pixel']:.2f}",
                      f"anchor   : {data.get('anchor_px','-')}"]
        self._load_lbl.config(text="\n".join(lines))

    # ─────────────────────────────────────────────────────────
    #  PROFILES  BROWSER
    # ─────────────────────────────────────────────────────────

    def _refresh_profiles(self):
        self._tree.delete(*self._tree.get_children())
        folder = Path(self._v_save_dir.get())
        if not folder.exists():
            return
        for f in sorted(folder.glob("*.npz")):
            data = load_calibration(f)
            if data is None:
                continue
            rms  = f"{data.get('rms_error', 0):.3f}"
            cpp  = (f"{data['cm_per_pixel']:.5f}"
                    if "cm_per_pixel" in data else "–")
            h    = f"{data.get('height_cm', 0):.0f}"
            self._tree.insert("", tk.END,
                              values=(f.name, rms, cpp, h),
                              tags=(str(f),))

    def _load_selected(self):
        sel = self._tree.selection()
        if not sel:
            messagebox.showwarning("No selection", "Select a profile first")
            return
        tags = self._tree.item(sel[0], "tags")
        if tags:
            self._load_from_path(Path(tags[0]))

    def _delete_selected(self):
        sel = self._tree.selection()
        if not sel:
            return
        tags = self._tree.item(sel[0], "tags")
        if not tags:
            return
        path = Path(tags[0])
        if messagebox.askyesno("Delete", f"Delete {path.name}?"):
            path.unlink(missing_ok=True)
            self._refresh_profiles()
            self._set_status(f"Deleted: {path.name}")

    def _on_tree_select(self, _):
        sel = self._tree.selection()
        if not sel:
            return
        tags = self._tree.item(sel[0], "tags")
        if tags:
            data = load_calibration(Path(tags[0]))
            if data:
                self._show_load_detail(data)

    # ─────────────────────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._status_lbl.config(text=msg)

    def _on_close(self):
        self._is_live = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════
#  ENTRY  POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    CalibrationGUI(root)
    root.mainloop()