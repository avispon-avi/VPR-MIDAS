#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-Shot Universal Room Boundary & Relative Positioning Module (Any-Room VPRS).
Works in any room without pre-training or prior maps:
1. Calculates room boundaries (Left Wall, Right Wall, Front Wall/Obstacle, Floor, Ceiling)
   within the camera's field of view using monocular depth geometry.
2. Computes the drone's position relative to the room (Centerline Offset, Wall Distances, Height).
3. Renders a Top-Down Bird's-Eye View (BEV) Room Radar and HUD Overlay.
"""

import cv2
import numpy as np
import time
import struct


class RoomLocalizer:
    def __init__(self, hfov_deg=66.0, vfov_deg=50.0, depth_scale=500.0, ema_alpha=0.3):
        """
        hfov_deg: Horizontal Field of View (degrees). Default 66 deg for XIAO ESP32S3 OV2640.
        vfov_deg: Vertical Field of View (degrees). Default 50 deg.
        depth_scale: Scaling factor converting MiDaS inverse depth to approximate metric meters.
        ema_alpha: Smoothing factor for temporal filtering (0.0 to 1.0).
        """
        self.hfov_rad = np.radians(hfov_deg)
        self.vfov_rad = np.radians(vfov_deg)
        self.depth_scale = depth_scale
        self.ema_alpha = ema_alpha

        # Smoothed state
        self.smoothed_data = None
        self.last_timestamp = time.time()
        self.seq_num = 0

    def process(self, depth_raw, rgb_shape):
        """
        depth_raw: 2D numpy array of inverse depth from MiDaS (HxW).
        rgb_shape: (H, W) or (H, W, 3).
        Returns: dict with room boundaries, positioning, and guidance.
        """
        H, W = rgb_shape[:2]
        if depth_raw.shape != (H, W):
            depth_raw = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_LINEAR)

        # Convert MiDaS inverse depth to metric depth Z (meters)
        safe_depth = np.maximum(depth_raw, 1e-3)
        Z_metric = np.clip(self.depth_scale / safe_depth, 0.25, 8.0)

        # Camera intrinsics
        fx = (W / 2.0) / np.tan(self.hfov_rad / 2.0)
        fy = (H / 2.0) / np.tan(self.vfov_rad / 2.0)
        cx = W / 2.0
        cy = H / 2.0

        # Downsample grid for fast 3D reconstruction (30x40 grid)
        step_y = max(1, H // 30)
        step_x = max(1, W // 40)
        grid_y, grid_x = np.mgrid[0:H:step_y, 0:W:step_x]
        
        z_sample = Z_metric[grid_y, grid_x]
        x_sample = (grid_x - cx) * z_sample / fx  # +X = Right, -X = Left
        y_sample = (grid_y - cy) * z_sample / fy  # +Y = Down (Floor), -Y = Up (Ceiling)

        # 1. Front Obstacle / Wall Boundary (central corridor)
        center_mask = (grid_x >= 0.35 * W) & (grid_x <= 0.65 * W) & \
                      (grid_y >= 0.30 * H) & (grid_y <= 0.70 * H)
        if np.any(center_mask):
            d_front = float(np.percentile(z_sample[center_mask], 25))
        else:
            d_front = float(np.median(Z_metric[int(0.4*H):int(0.6*H), int(0.4*W):int(0.6*W)]))

        # 2. Left Wall / Boundary (rays on the left, X < 0)
        left_mask = (grid_x <= 0.30 * W) & (grid_y >= 0.20 * H) & (grid_y <= 0.80 * H)
        if np.any(left_mask):
            d_left = float(np.median(np.abs(x_sample[left_mask])))
        else:
            d_left = 1.5

        # 3. Right Wall / Boundary (rays on the right, X > 0)
        right_mask = (grid_x >= 0.70 * W) & (grid_y >= 0.20 * H) & (grid_y <= 0.80 * H)
        if np.any(right_mask):
            d_right = float(np.median(x_sample[right_mask]))
        else:
            d_right = 1.5

        # 4. Floor Boundary (Lower sector, Y > 0)
        floor_mask = (grid_y >= 0.75 * H) & (grid_x >= 0.25 * W) & (grid_x <= 0.75 * W)
        if np.any(floor_mask):
            h_floor = float(np.percentile(y_sample[floor_mask], 70))
        else:
            h_floor = 1.0

        # 5. Ceiling Boundary (Upper sector, Y < 0)
        ceil_mask = (grid_y <= 0.25 * H) & (grid_x >= 0.25 * W) & (grid_x <= 0.75 * W)
        if np.any(ceil_mask):
            h_ceil = float(np.percentile(np.abs(y_sample[ceil_mask]), 70))
        else:
            h_ceil = 1.5

        # Sanity clamps
        d_left = max(0.2, min(d_left, 8.0))
        d_right = max(0.2, min(d_right, 8.0))
        d_front = max(0.2, min(d_front, 10.0))
        h_floor = max(0.1, min(h_floor, 5.0))
        h_ceil = max(0.1, min(h_ceil, 5.0))

        # Room metrics
        room_width = d_left + d_right
        room_height = h_floor + h_ceil
        # Lateral position of drone relative to room center (+ = Right, - = Left)
        drone_offset_x = (d_left - d_right) / 2.0
        # Percentage from left wall (0% = at left wall, 50% = centered, 100% = right wall)
        pos_percent = (d_left / max(0.1, room_width)) * 100.0

        # Guidance message
        if abs(drone_offset_x) <= 0.15:
            guidance = "CENTERED"
            guidance_color = (0, 255, 0)
        elif drone_offset_x < -0.15:
            guidance = f"DRIFT LEFT ({abs(drone_offset_x):.2f}m)"
            guidance_color = (0, 215, 255)
        else:
            guidance = f"DRIFT RIGHT ({drone_offset_x:.2f}m)"
            guidance_color = (0, 165, 255)

        raw_result = {
            "d_left": d_left,
            "d_right": d_right,
            "d_front": d_front,
            "h_floor": h_floor,
            "h_ceil": h_ceil,
            "room_width": room_width,
            "room_height": room_height,
            "offset_x": drone_offset_x,
            "pos_percent": pos_percent,
            "guidance": guidance,
            "guidance_color": guidance_color,
            "timestamp": time.time()
        }

        # Temporal smoothing (EWMA filter)
        if self.smoothed_data is None:
            self.smoothed_data = raw_result.copy()
        else:
            a = self.ema_alpha
            for key in ["d_left", "d_right", "d_front", "h_floor", "h_ceil", 
                        "room_width", "room_height", "offset_x", "pos_percent"]:
                self.smoothed_data[key] = (1.0 - a) * self.smoothed_data[key] + a * raw_result[key]
            self.smoothed_data["guidance"] = raw_result["guidance"]
            self.smoothed_data["guidance_color"] = raw_result["guidance_color"]
            self.smoothed_data["timestamp"] = raw_result["timestamp"]

        self.seq_num = (self.seq_num + 1) & 0xFFFF
        return self.smoothed_data

    def render_bev_radar(self, room_info, width=320, height=240, max_range=5.0):
        """
        Renders a Top-Down Bird's-Eye View (BEV) Room Radar showing:
        - Detected room boundary walls (Left, Right, Front)
        - Room centerline and drone position
        - Camera FOV frustum cone
        - Distance grid rings (1m, 2m, 3m, 4m)
        """
        radar = np.zeros((height, width, 3), dtype=np.uint8)
        radar[:] = (18, 22, 28)  # Sleek dark tech background

        # Origin (drone position) in radar pixel coordinates: bottom-center
        ox = width // 2
        oy = int(height * 0.86)
        scale = (height * 0.72) / max_range  # pixels per meter

        # 1. Concentric distance rings
        for r_m in range(1, int(max_range) + 1):
            r_px = int(r_m * scale)
            cv2.circle(radar, (ox, oy), r_px, (40, 50, 65), 1, cv2.LINE_AA)
            cv2.putText(radar, f"{r_m}m", (ox + 4, oy - r_px - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (75, 95, 120), 1, cv2.LINE_AA)

        # 2. Camera FOV Cone
        half_fov = self.hfov_rad / 2.0
        cone_len = max_range * scale
        fov_l_x = int(ox - cone_len * np.sin(half_fov))
        fov_l_y = int(oy - cone_len * np.cos(half_fov))
        fov_r_x = int(ox + cone_len * np.sin(half_fov))
        fov_r_y = int(oy - cone_len * np.cos(half_fov))

        fov_pts = np.array([[ox, oy], [fov_l_x, fov_l_y], [fov_r_x, fov_r_y]], dtype=np.int32)
        fov_overlay = radar.copy()
        cv2.fillPoly(fov_overlay, [fov_pts], (25, 45, 55))
        cv2.addWeighted(fov_overlay, 0.5, radar, 0.5, 0, radar)
        cv2.line(radar, (ox, oy), (fov_l_x, fov_l_y), (60, 110, 130), 1, cv2.LINE_AA)
        cv2.line(radar, (ox, oy), (fov_r_x, fov_r_y), (60, 110, 130), 1, cv2.LINE_AA)

        # 3. Room Boundary Walls
        d_left = room_info["d_left"]
        d_right = room_info["d_right"]
        d_front = room_info["d_front"]

        wall_l_x = int(np.clip(ox - d_left * scale, 5, width - 5))
        wall_r_x = int(np.clip(ox + d_right * scale, 5, width - 5))
        wall_f_y = int(np.clip(oy - d_front * scale, 25, height - 10))

        wall_bot_y = int(oy + 12)

        # Draw Left Wall (Cyan/Yellow)
        cv2.line(radar, (wall_l_x, wall_bot_y), (wall_l_x, wall_f_y), (0, 200, 255), 3, cv2.LINE_AA)
        cv2.putText(radar, f"LEFT {d_left:.1f}m", (max(4, wall_l_x - 62), oy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 255), 1, cv2.LINE_AA)

        # Draw Right Wall (Cyan/Yellow)
        cv2.line(radar, (wall_r_x, wall_bot_y), (wall_r_x, wall_f_y), (0, 200, 255), 3, cv2.LINE_AA)
        cv2.putText(radar, f"RIGHT {d_right:.1f}m", (min(width - 68, wall_r_x + 4), oy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 255), 1, cv2.LINE_AA)

        # Draw Front Wall / Boundary
        cv2.line(radar, (wall_l_x, wall_f_y), (wall_r_x, wall_f_y), (255, 80, 80), 3, cv2.LINE_AA)
        cv2.putText(radar, f"FRONT {d_front:.1f}m", (ox - 35, max(36, wall_f_y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 120, 120), 1, cv2.LINE_AA)

        # 4. Room Centerline (Dashed Green)
        center_x = int((wall_l_x + wall_r_x) / 2)
        for y_dash in range(max(26, wall_f_y), oy, 10):
            cv2.line(radar, (center_x, y_dash), (center_x, min(oy, y_dash + 5)), (180, 220, 100), 1, cv2.LINE_AA)
        cv2.putText(radar, "CENTER", (center_x - 20, oy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 220, 100), 1, cv2.LINE_AA)

        # 5. Drone Marker & Orientation
        cv2.circle(radar, (ox, oy), 6, (0, 255, 120), -1, cv2.LINE_AA)
        cv2.circle(radar, (ox, oy), 9, (255, 255, 255), 1, cv2.LINE_AA)
        # Drone forward heading arrow
        cv2.arrowedLine(radar, (ox, oy), (ox, oy - 16), (0, 255, 120), 2, tipLength=0.35)
        # Drone motor arms
        arm = 6
        cv2.line(radar, (ox - arm, oy - arm), (ox + arm, oy + arm), (0, 255, 120), 1, cv2.LINE_AA)
        cv2.line(radar, (ox - arm, oy + arm), (ox + arm, oy - arm), (0, 255, 120), 1, cv2.LINE_AA)

        # Center Offset Indicator arrow
        if abs(center_x - ox) > 3:
            cv2.arrowedLine(radar, (ox, oy - 22), (center_x, oy - 22),
                            room_info["guidance_color"], 2, tipLength=0.3)

        # Title bar
        cv2.rectangle(radar, (0, 0), (width, 22), (28, 34, 42), -1)
        cv2.putText(radar, "ROOM RADAR (TOP-DOWN)", (8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(radar, f"Offset: {room_info['offset_x']:+.2f}m", (width - 105, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, room_info["guidance_color"], 1, cv2.LINE_AA)

        return radar

    def draw_room_overlay(self, frame, room_info):
        """
        Draws perspective room boundary wireframe and sleek HUD overlay directly on the frame.
        Maintains the exact same (H, W, 3) frame dimensions.
        """
        H, W = frame.shape[:2]
        vis = frame.copy()

        # Perspective boundary horizon lines
        # Floor boundary line
        floor_y = int(np.clip(H * 0.80, 0, H - 1))
        cv2.line(vis, (0, floor_y), (W, floor_y), (0, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(vis, f"FLOOR {room_info['h_floor']:.1f}m", (10, floor_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 120), 1, cv2.LINE_AA)

        # Ceiling boundary line
        ceil_y = int(np.clip(H * 0.22, 0, H - 1))
        cv2.line(vis, (0, ceil_y), (W, ceil_y), (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"CEIL {room_info['h_ceil']:.1f}m", (10, ceil_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 0), 1, cv2.LINE_AA)

        # Left & Right boundary brackets
        # Left edge bracket
        cv2.line(vis, (20, ceil_y), (20, floor_y), (0, 200, 255), 2, cv2.LINE_AA)
        cv2.line(vis, (20, (ceil_y + floor_y)//2), (38, (ceil_y + floor_y)//2), (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"L: {room_info['d_left']:.1f}m", (25, (ceil_y + floor_y)//2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)

        # Right edge bracket
        cv2.line(vis, (W - 20, ceil_y), (W - 20, floor_y), (0, 200, 255), 2, cv2.LINE_AA)
        cv2.line(vis, (W - 20, (ceil_y + floor_y)//2), (W - 38, (ceil_y + floor_y)//2), (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"R: {room_info['d_right']:.1f}m", (W - 80, (ceil_y + floor_y)//2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)

        # Front Central Target Box
        bx_w, bx_h = int(W * 0.26), int(H * 0.24)
        bx_x = (W - bx_w) // 2
        bx_y = (H - bx_h) // 2
        cv2.rectangle(vis, (bx_x, bx_y), (bx_x + bx_w, bx_y + bx_h), (255, 80, 80), 1, cv2.LINE_AA)
        cv2.line(vis, (W//2 - 8, H//2), (W//2 + 8, H//2), (255, 80, 80), 1)
        cv2.line(vis, (W//2, H//2 - 8), (W//2, H//2 + 8), (255, 80, 80), 1)
        cv2.putText(vis, f"FWD {room_info['d_front']:.1f}m", (bx_x + 5, bx_y + bx_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 120, 120), 1, cv2.LINE_AA)

        # Transparent HUD Top Banner
        hud_h = 42
        hud_overlay = vis[:hud_h, :].copy()
        cv2.rectangle(hud_overlay, (0, 0), (W, hud_h), (15, 20, 25), -1)
        cv2.addWeighted(hud_overlay, 0.75, vis[:hud_h, :], 0.25, 0, vis[:hud_h, :])

        # HUD Text
        cv2.putText(vis, "ROOM VPRS (ANY ROOM)", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, room_info["guidance"], (W - 145, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, room_info["guidance_color"], 1, cv2.LINE_AA)

        cv2.putText(vis, f"Offset: {room_info['offset_x']:+.2f}m ({room_info['pos_percent']:.0f}%)",
                    (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(vis, f"W: {room_info['room_width']:.1f}m | H: {room_info['room_height']:.1f}m",
                    (W - 145, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 200, 220), 1, cv2.LINE_AA)

        return vis

    def pack_vprs_packet(self, room_info):
        """
        Packs relative room coordinates into VPRSPacket binary structure for Flight Controller.
        x: lateral offset from room center (meters, + = Right, - = Left)
        y: altitude above floor (meters)
        z: front clearance (meters)
        """
        header = bytes([0xAA, 0x55])
        seq = self.seq_num
        timestamp = int(room_info["timestamp"] * 1000) & 0xFFFFFFFF
        x = float(room_info["offset_x"])
        y = float(room_info["h_floor"])
        z = float(room_info["d_front"])
        vx, vy, vz = 0.0, 0.0, 0.0
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        confidence = 0.95
        status_flags = 0x01  # Valid 3D fix

        # Pack without checksum first (42 bytes)
        payload = struct.pack("<2sHIffffffffffB", header, seq, timestamp,
                              x, y, z, vx, vy, vz, qx, qy, qz, qw,
                              confidence, status_flags)
        # Compute 8-bit checksum
        checksum = sum(payload) & 0xFF
        return payload + struct.pack("<B", checksum)
