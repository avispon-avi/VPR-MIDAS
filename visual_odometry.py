#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Odometry — real-time camera motion tracking (works in ANY room).

Uses sparse optical flow (Shi-Tomasi corners + Lucas-Kanade tracker) to
estimate frame-to-frame camera motion, then integrates into a 2D trajectory.
This is the "draws its own path" behavior — no training, no prior map.

Examples:
    python visual_odometry.py --url http://192.168.137.134/capture
    python visual_odometry.py --camera 0
"""

import argparse
import time
import numpy as np
import cv2

# ---------------------------------------------------------------- VO Engine

class VisualOdometry:
    """Monocular visual odometry using sparse optical flow."""

    # Shi-Tomasi corner detection params
    FEATURE_PARAMS = dict(
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=15,
        blockSize=7,
    )

    # Lucas-Kanade optical flow params
    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    # Minimum features to attempt pose estimation
    MIN_FEATURES = 15

    def __init__(self, focal_length=None, pp=None, scale=1.0):
        """
        focal_length: camera focal length in pixels (estimated if None)
        pp: principal point (cx, cy) (center of image if None)
        scale: multiplier for translation (for display purposes)
        """
        self.focal = focal_length
        self.pp = pp
        self.scale = scale

        # State
        self.prev_gray = None
        self.prev_kp = None
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z in camera coords
        self.heading = 0.0  # yaw angle in radians
        self.trajectory = [(0.0, 0.0)]  # (x, z) floor-plane positions
        self.frame_count = 0
        self.tracking_quality = 0.0

    def _estimate_intrinsics(self, h, w):
        """Estimate camera intrinsics from image dimensions."""
        if self.focal is None:
            # Rough estimate: focal ~ image width (typical for phone/webcam)
            self.focal = w * 0.8
        if self.pp is None:
            self.pp = (w / 2.0, h / 2.0)

    def _detect_features(self, gray):
        """Detect Shi-Tomasi corners."""
        kp = cv2.goodFeaturesToTrack(gray, **self.FEATURE_PARAMS)
        return kp

    def _track_features(self, prev_gray, curr_gray, prev_kp):
        """Track features with Lucas-Kanade optical flow."""
        curr_kp, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_kp, None, **self.LK_PARAMS
        )
        if curr_kp is None or status is None:
            return None, None

        # Keep only good matches
        good = status.ravel() == 1
        if good.sum() < self.MIN_FEATURES:
            return None, None

        prev_good = prev_kp[good]
        curr_good = curr_kp[good]
        return prev_good, curr_good

    def _estimate_motion(self, prev_pts, curr_pts):
        """
        Estimate camera motion from point correspondences using Essential matrix.
        Returns (R, t, inlier_ratio).
        """
        K = np.array([
            [self.focal, 0, self.pp[0]],
            [0, self.focal, self.pp[1]],
            [0, 0, 1],
        ], dtype=np.float64)

        E, mask = cv2.findEssentialMat(
            curr_pts, prev_pts, K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or mask is None:
            return None, None, 0.0

        inlier_ratio = mask.ravel().mean()

        _, R, t, mask2 = cv2.recoverPose(E, curr_pts, prev_pts, K)
        return R, t.ravel(), inlier_ratio

    def process_frame(self, frame_bgr):
        """
        Process one frame. Returns:
            trajectory: list of (x, z) positions
            tracking_quality: float 0-1 (feature tracking quality)
            num_features: int
            flow_vectors: (prev_pts, curr_pts) for visualization
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        self._estimate_intrinsics(h, w)
        self.frame_count += 1

        flow_vectors = (None, None)

        if self.prev_gray is None:
            # First frame — detect features
            self.prev_gray = gray
            self.prev_kp = self._detect_features(gray)
            return self.trajectory, 0.0, 0, flow_vectors

        if self.prev_kp is None or len(self.prev_kp) < self.MIN_FEATURES:
            # Re-detect features
            self.prev_kp = self._detect_features(gray)
            if self.prev_kp is None or len(self.prev_kp) < self.MIN_FEATURES:
                self.prev_gray = gray
                return self.trajectory, 0.0, 0, flow_vectors

        # Track features
        prev_pts, curr_pts = self._track_features(self.prev_gray, gray, self.prev_kp)

        if prev_pts is None:
            # Tracking lost — re-detect
            self.prev_gray = gray
            self.prev_kp = self._detect_features(gray)
            return self.trajectory, 0.0, 0, flow_vectors

        flow_vectors = (prev_pts, curr_pts)
        num_features = len(curr_pts)

        # Estimate motion
        R, t, inlier_ratio = self._estimate_motion(prev_pts, curr_pts)
        self.tracking_quality = inlier_ratio

        if R is not None and inlier_ratio > 0.3:
            # Extract motion on the floor plane (x-z)
            # t is a unit vector — scale it
            dx = float(t[0]) * self.scale
            dy = float(t[1]) * self.scale
            dz = float(t[2]) * self.scale

            # Rotate translation by current heading
            cos_h = np.cos(self.heading)
            sin_h = np.sin(self.heading)
            world_dx = cos_h * dx - sin_h * dz
            world_dz = sin_h * dx + cos_h * dz

            # Update position
            self.position[0] += world_dx
            self.position[1] += dy
            self.position[2] += world_dz

            # Update heading from rotation matrix
            # Extract yaw (rotation around Y axis)
            yaw = np.arctan2(R[0, 2], R[2, 2])
            self.heading += yaw

            self.trajectory.append((self.position[0], self.position[2]))

        # Prepare for next frame — re-detect if features are low
        if num_features < self.MIN_FEATURES * 2:
            self.prev_kp = self._detect_features(gray)
        else:
            self.prev_kp = curr_pts.reshape(-1, 1, 2)
        self.prev_gray = gray

        return self.trajectory, self.tracking_quality, num_features, flow_vectors


# ---------------------------------------------------------------- Visualization

def draw_trajectory_map(trajectory, map_size=300, margin=20):
    """Draw a top-down 2D trajectory on a black canvas."""
    canvas = np.zeros((map_size, map_size, 3), dtype=np.uint8)

    if len(trajectory) < 2:
        # Draw origin cross
        c = map_size // 2
        cv2.drawMarker(canvas, (c, c), (0, 255, 0),
                       cv2.MARKER_CROSS, 10, 1)
        cv2.putText(canvas, "Start", (c + 8, c - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        return canvas

    xs = [p[0] for p in trajectory]
    zs = [p[1] for p in trajectory]
    xs_arr = np.array(xs)
    zs_arr = np.array(zs)

    # Auto-scale to fit canvas
    x_range = max(xs_arr.max() - xs_arr.min(), 0.01)
    z_range = max(zs_arr.max() - zs_arr.min(), 0.01)
    scale = (map_size - 2 * margin) / max(x_range, z_range)

    x_center = (xs_arr.max() + xs_arr.min()) / 2
    z_center = (zs_arr.max() + zs_arr.min()) / 2

    def to_pixel(x, z):
        px = int(map_size / 2 + (x - x_center) * scale)
        pz = int(map_size / 2 + (z - z_center) * scale)
        return (px, pz)

    # Draw grid
    for i in range(0, map_size, 50):
        cv2.line(canvas, (i, 0), (i, map_size), (30, 30, 30), 1)
        cv2.line(canvas, (0, i), (map_size, i), (30, 30, 30), 1)

    # Draw trajectory line
    pts = [to_pixel(x, z) for x, z in trajectory]
    for i in range(1, len(pts)):
        # Color gradient: blue → green as trajectory progresses
        progress = i / len(pts)
        color = (int(255 * (1 - progress)), int(255 * progress), 0)
        cv2.line(canvas, pts[i - 1], pts[i], color, 2)

    # Draw start point (green circle)
    cv2.circle(canvas, pts[0], 5, (0, 200, 0), -1)
    cv2.putText(canvas, "S", (pts[0][0] + 6, pts[0][1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

    # Draw current position (red circle)
    cv2.circle(canvas, pts[-1], 6, (0, 0, 255), -1)

    # Draw heading indicator
    if len(trajectory) >= 2:
        dx = xs[-1] - xs[-2]
        dz = zs[-1] - zs[-2]
        length = np.sqrt(dx * dx + dz * dz)
        if length > 1e-6:
            ndx = dx / length * 15
            ndz = dz / length * 15
            tip = (int(pts[-1][0] + ndx), int(pts[-1][1] + ndz))
            cv2.arrowedLine(canvas, pts[-1], tip, (0, 0, 255), 2, tipLength=0.4)

    return canvas


def draw_flow(frame, prev_pts, curr_pts, max_draw=80):
    """Draw optical flow vectors on the frame."""
    vis = frame.copy()
    if prev_pts is None or curr_pts is None:
        return vis

    step = max(1, len(prev_pts) // max_draw)
    for i in range(0, len(prev_pts), step):
        p0 = tuple(prev_pts[i].astype(int).ravel())
        p1 = tuple(curr_pts[i].astype(int).ravel())
        cv2.arrowedLine(vis, p0, p1, (0, 255, 255), 1, tipLength=0.3)
        cv2.circle(vis, p1, 2, (0, 255, 0), -1)

    return vis


# ---------------------------------------------------------------- Main Loop

def run_stream(camera=None, url=None, vo_scale=1.0):
    """Run visual odometry on a live stream."""
    import requests

    cap = None
    if camera is not None:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {camera}")

    vo = VisualOdometry(scale=vo_scale)
    fps = 0.0

    print("[VO] Visual Odometry started — press q to quit")
    print("[VO] Move the camera slowly. The trajectory draws in real time.")
    cv2.namedWindow("Visual Odometry — q to quit", cv2.WINDOW_NORMAL)

    while True:
        # Grab frame
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            try:
                r = requests.get(url, timeout=3)
                frame = cv2.imdecode(
                    np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue
            except Exception as e:
                print(f"[VO] fetch error: {e}")
                time.sleep(0.5)
                continue

        t0 = time.time()
        trajectory, quality, n_features, (prev_pts, curr_pts) = vo.process_frame(frame)
        dt = time.time() - t0
        fps = 0.9 * fps + 0.1 / max(1e-3, dt)

        # Draw optical flow on frame
        vis_frame = draw_flow(frame, prev_pts, curr_pts)

        # Scale up for display
        vis_frame = cv2.resize(vis_frame, (0, 0), fx=2.0, fy=2.0,
                               interpolation=cv2.INTER_LINEAR)

        # Overlay text
        cv2.putText(vis_frame, f"VO | {fps:.1f} FPS | Features: {n_features}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Quality: {quality:.0%} | Pos: ({vo.position[0]:+.2f}, {vo.position[2]:+.2f})",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        mode_text = "TRACKING" if quality > 0.3 else "LOST - move slowly"
        mode_color = (0, 255, 0) if quality > 0.3 else (0, 0, 255)
        cv2.putText(vis_frame, mode_text,
                    (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        # Draw trajectory map
        traj_map = draw_trajectory_map(trajectory, map_size=280)

        # Add label to map
        cv2.putText(traj_map, "Top-Down Trajectory", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.putText(traj_map, f"{len(trajectory)} pts", (5, 272),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Compose: frame on left, map on right
        h_frame = vis_frame.shape[0]
        map_resized = cv2.resize(traj_map, (h_frame, h_frame))
        combo = np.hstack([vis_frame, map_resized])

        cv2.imshow("Visual Odometry — q to quit", combo)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    # Save trajectory
    if len(trajectory) > 1:
        arr = np.array(trajectory)
        np.savetxt("vo_trajectory.csv", arr, delimiter=",",
                   header="x,z", comments="")
        print(f"[VO] Trajectory saved: vo_trajectory.csv ({len(trajectory)} points)")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(arr[:, 0], arr[:, 1], "b.-", ms=2, lw=1, label="VO path")
        ax.scatter(arr[0, 0], arr[0, 1], c="green", s=80, zorder=5, label="start")
        ax.scatter(arr[-1, 0], arr[-1, 1], c="red", s=80, zorder=5, label="end")
        ax.set_xlabel("X"); ax.set_ylabel("Z")
        ax.set_title("Visual Odometry Trajectory")
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig("vo_trajectory.png", dpi=120)
        print("[VO] Plot saved: vo_trajectory.png")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=None, help="XIAO capture endpoint")
    p.add_argument("--camera", type=int, default=None, help="webcam index")
    p.add_argument("--scale", type=float, default=1.0,
                   help="motion scale multiplier for display")
    a = p.parse_args()

    if a.camera is None and a.url is None:
        raise SystemExit("Specify --url http://... or --camera 0")

    run_stream(a.camera, a.url, a.scale)


if __name__ == "__main__":
    main()
