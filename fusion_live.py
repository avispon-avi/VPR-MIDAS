#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion Live — Dual-mode Indoor Localization System

Combines PoseNet (VPR), Visual Odometry, and MiDaS Depth into a single
real-time GUI. Automatically switches between modes based on PoseNet
confidence.

Architecture:
  Mode 1 (Known Room): PoseNet gives absolute position, MiDaS draws walls
  Mode 2 (Unknown Room): Visual Odometry tracks relative motion, MiDaS draws walls

The confidence check: when PoseNet outputs near the training-set center
(mean), it's confused → switch to VO. When it outputs a distinctive
position, it recognizes the room → use PoseNet.

Examples:
    python fusion_live.py --url http://192.168.137.134/capture
    python fusion_live.py --camera 0
    python fusion_live.py --url http://192.168.137.134/capture --no-depth
"""

import argparse
import os
import sys
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- PoseNet

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE_SHORT  = 256
CROP          = 224


class PoseNet(nn.Module):
    FEAT_DIMS = {"mobilenet_v3_small": 576, "mobilenet_v3_large": 960,
                 "resnet18": 512}

    def __init__(self, backbone="mobilenet_v3_small", dropout=0.5):
        super().__init__()
        import torchvision.models as tvm
        if backbone == "mobilenet_v3_small":
            self.features = tvm.mobilenet_v3_small(weights=None).features
        elif backbone == "mobilenet_v3_large":
            self.features = tvm.mobilenet_v3_large(weights=None).features
        elif backbone == "resnet18":
            self.features = nn.Sequential(
                *list(tvm.resnet18(weights=None).children())[:-2])
        else:
            raise ValueError(f"unknown backbone {backbone}")
        feat = self.FEAT_DIMS[backbone]
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(feat, 2048), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.fc_xyz  = nn.Linear(2048, 3)
        self.fc_quat = nn.Linear(2048, 4)

    def forward(self, x):
        f = self.fc(self.features(x))
        return self.fc_xyz(f), F.normalize(self.fc_quat(f), p=2, dim=1)


def preprocess_posenet(img_rgb):
    """RGB uint8 HxWx3 -> (1,3,224,224) normalized tensor."""
    h, w = img_rgb.shape[:2]
    scale = RESIZE_SHORT / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    top, left = (h - CROP) // 2, (w - CROP) // 2
    img = img[top:top + CROP, left:left + CROP]
    x = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0) / 255.0
    x = (x - torch.from_numpy(IMAGENET_MEAN).view(1, 3, 1, 1)) / \
        torch.from_numpy(IMAGENET_STD).view(1, 3, 1, 1)
    return x


class PoseNetEngine:
    """PoseNet with confidence scoring."""

    def __init__(self, weights_path, confidence_threshold=0.15):
        ck = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.model = PoseNet(ck.get("backbone", "mobilenet_v3_small"))
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()
        self.pos_mean = np.asarray(ck["pos_mean"], dtype=np.float32)
        self.pos_std  = np.asarray(ck["pos_std"], dtype=np.float32)
        self.confidence_threshold = confidence_threshold
        print(f"[PoseNet] Loaded: {weights_path}")
        print(f"[PoseNet] Training center (pos_mean): {self.pos_mean}")
        print(f"[PoseNet] Training spread (pos_std):  {self.pos_std}")

    @torch.no_grad()
    def predict(self, img_rgb):
        """
        Returns (position_xyz, quaternion, confidence).
        confidence: 0.0 = at center (confused), 1.0 = far from center (confident)
        """
        x = preprocess_posenet(img_rgb)
        p, q = self.model(x)
        pos = p.numpy()[0] * self.pos_std + self.pos_mean
        quat = q.numpy()[0]

        # Confidence = normalized distance from the training mean
        # When PoseNet sees an unknown room, it outputs ~pos_mean
        # When it recognizes the room, it outputs a distinctive position
        displacement = (p.numpy()[0])  # This is already in normalized space
        dist_from_center = float(np.linalg.norm(displacement))
        # Sigmoid-like mapping: 0 at center, ~1 when far
        confidence = min(1.0, dist_from_center / 2.0)

        return pos, quat, confidence

    def is_confident(self, confidence):
        return confidence > self.confidence_threshold


# ---------------------------------------------------------------- Visual Odometry

class VisualOdometry:
    """Sparse optical flow VO for relative motion tracking."""

    FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01,
                          minDistance=15, blockSize=7)
    LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               30, 0.01))
    MIN_FEATURES = 15

    def __init__(self, scale=1.0):
        self.scale = scale
        self.prev_gray = None
        self.prev_kp = None
        self.position = np.array([0.0, 0.0, 0.0])
        self.heading = 0.0
        self.trajectory = []
        self.focal = None
        self.pp = None
        self.quality = 0.0

    def reset_to(self, x, y, z):
        """Snap VO position to a known position (from PoseNet)."""
        self.position = np.array([x, y, z], dtype=np.float64)

    def process(self, frame_bgr):
        """Process one frame, return (dx, dz, quality, n_features)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if self.focal is None:
            self.focal = w * 0.8
            self.pp = (w / 2.0, h / 2.0)

        dx, dz = 0.0, 0.0
        n_feat = 0

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = cv2.goodFeaturesToTrack(gray, **self.FEATURE_PARAMS)
            return dx, dz, 0.0, 0

        if self.prev_kp is None or len(self.prev_kp) < self.MIN_FEATURES:
            self.prev_kp = cv2.goodFeaturesToTrack(gray, **self.FEATURE_PARAMS)
            self.prev_gray = gray
            return dx, dz, 0.0, 0

        # Track
        curr_kp, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_kp, None, **self.LK_PARAMS)

        if curr_kp is not None and status is not None:
            good = status.ravel() == 1
            if good.sum() >= self.MIN_FEATURES:
                prev_pts = self.prev_kp[good]
                curr_pts = curr_kp[good]
                n_feat = len(curr_pts)

                K = np.array([[self.focal, 0, self.pp[0]],
                              [0, self.focal, self.pp[1]],
                              [0, 0, 1]], dtype=np.float64)

                E, mask = cv2.findEssentialMat(
                    curr_pts, prev_pts, K,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0)

                if E is not None and mask is not None:
                    self.quality = float(mask.ravel().mean())
                    _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts, K)
                    t = t.ravel()  # (3,1) -> (3,)

                    if self.quality > 0.3:
                        dx = float(t[0]) * self.scale
                        dz = float(t[2]) * self.scale

                        cos_h = np.cos(self.heading)
                        sin_h = np.sin(self.heading)
                        world_dx = cos_h * dx - sin_h * dz
                        world_dz = sin_h * dx + cos_h * dz

                        self.position[0] += world_dx
                        self.position[2] += world_dz

                        yaw = np.arctan2(R[0, 2], R[2, 2])
                        self.heading += yaw

                        dx, dz = world_dx, world_dz

                if n_feat < self.MIN_FEATURES * 2:
                    self.prev_kp = cv2.goodFeaturesToTrack(gray, **self.FEATURE_PARAMS)
                else:
                    self.prev_kp = curr_pts.reshape(-1, 1, 2)

        self.prev_gray = gray
        return dx, dz, self.quality, n_feat


# ---------------------------------------------------------------- MiDaS Depth

class DepthEngine:
    """MiDaS depth estimator for wall detection."""

    def __init__(self, model_type="MiDaS_small"):
        # Pre-seed trusted repos
        hub_dir = torch.hub.get_dir()
        os.makedirs(hub_dir, exist_ok=True)
        trusted_file = os.path.join(hub_dir, "trusted_list")
        try:
            existing = set(open(trusted_file).read().split())
        except OSError:
            existing = set()
        needed = {"rwightman_gen-efficientnet-pytorch", "isl-org_MiDaS",
                  "intel-isl_MiDaS"}
        if not needed <= existing:
            with open(trusted_file, "a") as f:
                for name in sorted(needed - existing):
                    f.write(name + "\n")

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        for repo in ("isl-org/MiDaS", "intel-isl/MiDaS"):
            try:
                self.model = torch.hub.load(repo, model_type,
                                            trust_repo=True,
                                            skip_validation=True)
                transforms = torch.hub.load(repo, "transforms",
                                            trust_repo=True,
                                            skip_validation=True)
                self.transform = (transforms.dpt_transform
                                  if model_type.startswith("DPT")
                                  else transforms.small_transform)
                self.model.to(device).eval()
                print(f"[MiDaS] {model_type} ready on {device}")
                return
            except Exception as e:
                last_err = e

        raise SystemExit(f"[MiDaS] Failed to load: {last_err}\n"
                         f"  Fix: pip install timm")

    @torch.no_grad()
    def predict(self, img_rgb):
        """RGB uint8 HxWx3 -> relative inverse depth (higher=closer)."""
        x = self.transform(img_rgb).to(self.device)
        pred = self.model(x)
        depth = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=img_rgb.shape[:2],
            mode="bicubic", align_corners=False
        ).squeeze().float().cpu().numpy()
        return depth

    def get_wall_points(self, depth, heading, position_xz, fov_deg=60.0,
                        n_rays=24, max_depth_val=None):
        """
        Convert depth map to wall points in world coordinates.
        Casts rays from the camera position outward, using depth to
        determine wall distance.

        Returns list of (world_x, world_z) wall points.
        """
        h, w = depth.shape
        if max_depth_val is None:
            max_depth_val = depth.max()

        wall_points = []
        fov_rad = np.radians(fov_deg)
        half_fov = fov_rad / 2

        for i in range(n_rays):
            # Sample across the horizontal FOV
            angle_offset = -half_fov + (i / max(1, n_rays - 1)) * fov_rad
            ray_angle = heading + angle_offset

            # Sample depth at this horizontal position
            col = int(np.clip(i / max(1, n_rays - 1) * (w - 1), 0, w - 1))

            # Take median depth in a vertical strip (middle 60%)
            strip_top = int(h * 0.2)
            strip_bot = int(h * 0.8)
            strip = depth[strip_top:strip_bot, max(0, col - 5):min(w, col + 5)]

            if strip.size == 0:
                continue

            median_depth = float(np.median(strip))

            # Convert inverse depth to relative distance
            # MiDaS outputs inverse depth: high value = close
            if median_depth > 1e-3 and max_depth_val > 1e-3:
                # Normalize and invert: close objects have high depth value
                rel_distance = max_depth_val / max(median_depth, 1e-3)
                rel_distance = min(rel_distance, 5.0)  # Cap at 5 units
            else:
                rel_distance = 3.0

            # Project to world coordinates
            wx = position_xz[0] + rel_distance * np.sin(ray_angle)
            wz = position_xz[1] + rel_distance * np.cos(ray_angle)
            wall_points.append((wx, wz))

        return wall_points


# ---------------------------------------------------------------- Fusion GUI

class FusionDisplay:
    """Combined display: camera + room map + depth."""

    # Room map settings
    MAP_SIZE    = 400
    MAP_MARGIN  = 40
    MAP_SCALE   = 60.0  # pixels per meter (auto-adjusts)

    def __init__(self):
        self.trajectory = []  # (x, z, mode) where mode = 'posenet' or 'vo'
        self.wall_points_history = []  # accumulated wall points
        self.max_wall_points = 2000

    def add_position(self, x, z, mode):
        self.trajectory.append((x, z, mode))

    def add_wall_points(self, points):
        self.wall_points_history.extend(points)
        # Limit history
        if len(self.wall_points_history) > self.max_wall_points:
            self.wall_points_history = self.wall_points_history[-self.max_wall_points:]

    def _world_to_map(self, x, z, center_x, center_z, scale):
        px = int(self.MAP_SIZE / 2 + (x - center_x) * scale)
        pz = int(self.MAP_SIZE / 2 + (z - center_z) * scale)
        return (px, pz)

    def draw_room_map(self, current_x, current_z, mode, confidence):
        """Draw the 2D room map with trajectory and wall points."""
        canvas = np.zeros((self.MAP_SIZE, self.MAP_SIZE, 3), dtype=np.uint8)

        # Collect all points for auto-scaling
        all_x = [current_x]
        all_z = [current_z]
        for x, z, _ in self.trajectory:
            all_x.append(x)
            all_z.append(z)
        for x, z in self.wall_points_history:
            all_x.append(x)
            all_z.append(z)

        x_range = max(max(all_x) - min(all_x), 0.5)
        z_range = max(max(all_z) - min(all_z), 0.5)
        scale = (self.MAP_SIZE - 2 * self.MAP_MARGIN) / max(x_range, z_range)
        scale = min(scale, 200.0)  # Don't zoom in too much

        center_x = (max(all_x) + min(all_x)) / 2
        center_z = (max(all_z) + min(all_z)) / 2

        # Grid
        for i in range(0, self.MAP_SIZE, 40):
            cv2.line(canvas, (i, 0), (i, self.MAP_SIZE), (25, 25, 25), 1)
            cv2.line(canvas, (0, i), (self.MAP_SIZE, i), (25, 25, 25), 1)

        # Wall points (blue dots — these form the room boundary)
        for wx, wz in self.wall_points_history:
            px, pz = self._world_to_map(wx, wz, center_x, center_z, scale)
            if 0 <= px < self.MAP_SIZE and 0 <= pz < self.MAP_SIZE:
                cv2.circle(canvas, (px, pz), 2, (180, 100, 40), -1)

        # Trajectory
        if len(self.trajectory) > 1:
            for i in range(1, len(self.trajectory)):
                x1, z1, m1 = self.trajectory[i - 1]
                x2, z2, m2 = self.trajectory[i]
                p1 = self._world_to_map(x1, z1, center_x, center_z, scale)
                p2 = self._world_to_map(x2, z2, center_x, center_z, scale)

                # Color: green for PoseNet, yellow for VO
                color = (0, 255, 0) if m2 == "posenet" else (0, 220, 255)
                cv2.line(canvas, p1, p2, color, 2)

        # Start point
        if self.trajectory:
            sx, sz, _ = self.trajectory[0]
            sp = self._world_to_map(sx, sz, center_x, center_z, scale)
            cv2.circle(canvas, sp, 5, (0, 200, 0), -1)

        # Current position (large red dot)
        cp = self._world_to_map(current_x, current_z, center_x, center_z, scale)
        cv2.circle(canvas, cp, 8, (0, 0, 255), -1)
        cv2.circle(canvas, cp, 10, (255, 255, 255), 1)

        # Mode label on map
        mode_text = f"MODE: {mode.upper()}"
        mode_color = (0, 255, 0) if mode == "posenet" else (0, 220, 255)
        cv2.putText(canvas, mode_text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)

        # Confidence bar
        bar_x, bar_y, bar_w, bar_h = 8, 32, 150, 12
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill = int(confidence * bar_w)
        bar_color = (0, 200, 0) if confidence > 0.3 else (0, 100, 200)
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + fill, bar_y + bar_h), bar_color, -1)
        cv2.putText(canvas, f"Conf: {confidence:.0%}",
                    (bar_x + bar_w + 5, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # Position text
        cv2.putText(canvas, f"Pos: ({current_x:+.2f}, {current_z:+.2f})",
                    (8, self.MAP_SIZE - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Legend
        legend_y = self.MAP_SIZE - 50
        cv2.circle(canvas, (self.MAP_SIZE - 100, legend_y), 4, (0, 255, 0), -1)
        cv2.putText(canvas, "PoseNet", (self.MAP_SIZE - 90, legend_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.circle(canvas, (self.MAP_SIZE - 100, legend_y + 15), 4, (0, 220, 255), -1)
        cv2.putText(canvas, "Vis. Odom.", (self.MAP_SIZE - 90, legend_y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 220, 255), 1)
        cv2.circle(canvas, (self.MAP_SIZE - 100, legend_y + 30), 4, (180, 100, 40), -1)
        cv2.putText(canvas, "Wall pts", (self.MAP_SIZE - 90, legend_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 100, 40), 1)

        return canvas

    def colorize_depth(self, depth):
        """Depth array -> inferno colormap image."""
        d = depth - depth.min()
        m = d.max()
        if m > 1e-9:
            d = d / m
        return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


# ---------------------------------------------------------------- Main

def run_fusion(args):
    import requests

    # --- Initialize engines ---
    print("=" * 60)
    print("  FUSION LIVE — Dual-Mode Indoor Localization")
    print("=" * 60)

    # PoseNet
    posenet = PoseNetEngine(args.weights, confidence_threshold=args.conf_thresh)

    # Visual Odometry
    vo = VisualOdometry(scale=args.vo_scale)

    # MiDaS (optional)
    depth_engine = None
    if not args.no_depth:
        try:
            depth_engine = DepthEngine(args.depth_model)
        except Exception as e:
            print(f"[WARNING] MiDaS failed to load: {e}")
            print("[WARNING] Running without depth. Use --no-depth to suppress.")

    # Display
    display = FusionDisplay()

    # Stream source
    cap = None
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {args.camera}")

    # State
    current_pos = np.array([0.0, 0.0, 0.0])
    current_mode = "initializing"
    fps = 0.0
    frame_count = 0
    depth_frame_skip = args.depth_skip  # Run depth every N frames
    last_depth = None
    last_depth_color = None

    print(f"\n[Fusion] Streaming from {'camera ' + str(args.camera) if cap else args.url}")
    print("[Fusion] Press q to quit\n")
    cv2.namedWindow("Fusion Live — q to quit", cv2.WINDOW_NORMAL)

    while True:
        # --- Grab frame ---
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            try:
                r = requests.get(args.url, timeout=3)
                frame = cv2.imdecode(
                    np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception as e:
                print(f"[Fusion] fetch error: {e}")
                time.sleep(0.5)
                continue

        t0 = time.time()
        frame_count += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- PoseNet prediction ---
        pn_pos, pn_quat, confidence = posenet.predict(img_rgb)

        # --- Visual Odometry ---
        vo_dx, vo_dz, vo_quality, vo_features = vo.process(frame)

        # --- Mode decision ---
        if posenet.is_confident(confidence):
            # PoseNet recognizes this room → use absolute position
            current_mode = "posenet"
            current_pos = pn_pos.copy()
            # Snap VO to PoseNet position to correct drift
            vo.reset_to(pn_pos[0], pn_pos[1], pn_pos[2])
        else:
            # Unknown room → use Visual Odometry
            current_mode = "vo"
            current_pos = vo.position.copy()

        display.add_position(current_pos[0], current_pos[2], current_mode)

        # --- MiDaS depth (every N frames for performance) ---
        if depth_engine is not None and frame_count % depth_frame_skip == 0:
            last_depth = depth_engine.predict(img_rgb)
            last_depth_color = display.colorize_depth(last_depth)

            # Extract wall points
            wall_pts = depth_engine.get_wall_points(
                last_depth,
                heading=vo.heading,
                position_xz=(current_pos[0], current_pos[2]),
                fov_deg=60.0,
                n_rays=16,
            )
            display.add_wall_points(wall_pts)

        dt = time.time() - t0
        fps = 0.9 * fps + 0.1 / max(1e-3, dt)

        # --- Compose display ---

        # Scale up camera frame
        disp_h = 400
        scale_factor = disp_h / frame.shape[0]
        disp_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor,
                                interpolation=cv2.INTER_LINEAR)

        # Overlay text on camera frame
        # Mode indicator
        mode_text = f"MODE: {current_mode.upper()}"
        mode_color = (0, 255, 0) if current_mode == "posenet" else (0, 220, 255)
        cv2.putText(disp_frame, mode_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        # Position
        cv2.putText(disp_frame,
                    f"Pos: ({current_pos[0]:+.2f}, {current_pos[1]:+.2f}, {current_pos[2]:+.2f}) m",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Confidence
        conf_text = f"PoseNet conf: {confidence:.0%} | VO quality: {vo_quality:.0%}"
        cv2.putText(disp_frame, conf_text,
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # FPS + features
        cv2.putText(disp_frame, f"{fps:.1f} FPS | {vo_features} features",
                    (10, disp_frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # --- Depth strip (below camera) ---
        depth_strip_h = 100
        if last_depth_color is not None:
            depth_strip = cv2.resize(last_depth_color,
                                     (disp_frame.shape[1], depth_strip_h))
            cv2.putText(depth_strip, "Depth (bright=close)",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            depth_strip = np.zeros((depth_strip_h, disp_frame.shape[1], 3),
                                   dtype=np.uint8)
            cv2.putText(depth_strip, "Depth: loading...",
                        (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Left panel: camera + depth strip
        left_panel = np.vstack([disp_frame, depth_strip])

        # Right panel: room map (resize to match left panel height)
        room_map = display.draw_room_map(
            current_pos[0], current_pos[2], current_mode, confidence)
        room_map = cv2.resize(room_map,
                              (left_panel.shape[0], left_panel.shape[0]))

        # Combine
        combo = np.hstack([left_panel, room_map])

        cv2.imshow("Fusion Live — q to quit", combo)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    # Save session data
    if display.trajectory:
        arr = np.array([(x, z) for x, z, _ in display.trajectory])
        modes = [m for _, _, m in display.trajectory]
        np.savetxt("fusion_trajectory.csv", arr, delimiter=",",
                   header="x,z", comments="")
        print(f"[Fusion] Trajectory saved: fusion_trajectory.csv "
              f"({len(display.trajectory)} points)")

        # Mode stats
        pn_count = sum(1 for m in modes if m == "posenet")
        vo_count = sum(1 for m in modes if m == "vo")
        total = len(modes)
        print(f"[Fusion] Mode usage: PoseNet {pn_count}/{total} "
              f"({pn_count/total*100:.0f}%), VO {vo_count}/{total} "
              f"({vo_count/total*100:.0f}%)")

        # Plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 7))
            for i in range(1, len(arr)):
                color = "#00ff00" if modes[i] == "posenet" else "#ffcc00"
                ax.plot(arr[i-1:i+1, 0], arr[i-1:i+1, 1], ".-",
                        color=color, ms=2, lw=1)

            ax.scatter(arr[0, 0], arr[0, 1], c="green", s=80, zorder=5,
                       label="start")
            ax.scatter(arr[-1, 0], arr[-1, 1], c="red", s=80, zorder=5,
                       label="end")

            # Wall points
            if display.wall_points_history:
                wp = np.array(display.wall_points_history)
                ax.scatter(wp[:, 0], wp[:, 1], c="steelblue", s=5, alpha=0.3,
                           label="wall points")

            # Legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color="#00ff00", lw=2, label="PoseNet mode"),
                Line2D([0], [0], color="#ffcc00", lw=2, label="VO mode"),
                plt.scatter([], [], c="steelblue", s=20, alpha=0.5,
                            label="Wall points"),
            ]
            ax.legend(handles=legend_elements[:2], loc="upper right")

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.set_title("Fusion Trajectory — Dual-Mode Localization")
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig("fusion_trajectory.png", dpi=120)
            print("[Fusion] Plot saved: fusion_trajectory.png")
        except Exception as e:
            print(f"[Fusion] Plot error: {e}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Source
    p.add_argument("--url", default=None,
                   help="XIAO capture endpoint (e.g., http://192.168.137.134/capture)")
    p.add_argument("--camera", type=int, default=None, help="webcam index")

    # PoseNet
    p.add_argument("--weights", default="posenet_best.pth",
                   help="PoseNet checkpoint path")
    p.add_argument("--conf-thresh", type=float, default=0.15,
                   help="PoseNet confidence threshold (0-1). Below = unknown room")

    # Visual Odometry
    p.add_argument("--vo-scale", type=float, default=0.05,
                   help="VO translation scale (tune for your camera)")

    # MiDaS
    p.add_argument("--no-depth", action="store_true",
                   help="Disable MiDaS depth (faster)")
    p.add_argument("--depth-model", default="MiDaS_small",
                   choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"])
    p.add_argument("--depth-skip", type=int, default=3,
                   help="Run depth every N frames (saves CPU)")

    a = p.parse_args()

    if a.camera is None and a.url is None:
        raise SystemExit("Specify --url or --camera")

    run_fusion(a)


if __name__ == "__main__":
    main()
