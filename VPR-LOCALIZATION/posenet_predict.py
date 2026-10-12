#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPRS - Universal Visual Positioning & Room Boundary System.
Real-time Monocular Depth, Any-Room Boundary Detection, Relative Self-Positioning,
and optional PoseNet regression. Works in any new room within the camera's FOV.
"""

import argparse
import time
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
import torch

from core_localization import CoreLocalizer
from room_localizer import RoomLocalizer

target_xy = None
init_target = False
frame_width = 0

def mouse_callback(event, x, y, flags, param):
    global target_xy, frame_width, init_target
    if event == cv2.EVENT_LBUTTONDOWN:
        if frame_width > 0:
            # Clamp click coordinates to the first panel (RGB feed)
            if x < frame_width:
                target_xy = (x, y)
                init_target = True
                print(f"[Target] Selected at pixel: {target_xy}")

def run_stream(pred=None, camera=None, url=None, depth_scale=500.0):
    global target_xy, frame_width, init_target
    if cv2 is None:
        raise SystemExit("Need OpenCV: pip install opencv-python")
    
    cap = None
    if camera is not None:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise SystemExit(f"cannot open camera {camera}")

    # Load Universal Room Localizer
    room_localizer = RoomLocalizer(depth_scale=depth_scale)

    # Load Depth Model (MiDaS)
    print("Loading MiDaS monocular depth model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    win_name = "VPRS - RGB Boundaries | Depth | Top-Down Room Radar"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback)
    print("\n" + "="*60)
    print(" [VPRS ROOM LOCALIZER READY]")
    print(" -> Left Panel:   RGB feed + 3D Room Wireframe + HUD")
    print(" -> Center Panel: Real-time Depth Map")
    print(" -> Right Panel:  Top-Down Room Radar (BEV) with Room Centerline")
    print(" Controls: 'q' = Quit | 'c' = Clear Target | '+' / '-' = Scale Depth")
    print("="*60 + "\n")

    import requests

    fps = 0.0
    last_time = time.time()
    tracker = None

    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            try:
                r = requests.get(url, timeout=3)
                frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception as e:
                print(f"fetch error: {e}")
                time.sleep(0.5)
                continue

        H, W = frame.shape[:2]
        frame_width = W
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Depth Estimation (MiDaS)
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()

        # 2. Universal Room Boundary Detection & Relative Positioning
        room_info = room_localizer.process(depth_map, (H, W))

        # Colorize depth for display
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min > 0:
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = depth_map
        depth_viz = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

        # 3. Optional PoseNet Prediction
        pos_text_xy = ""
        pos_text_z = ""
        if pred is not None:
            res = pred.estimate_pose(img_rgb)
            pos_text_xy = f"PoseNet: x={res['x']:+.2f} y={res['y']:+.2f}"
            pos_text_z  = f"z={res['z']:+.2f}m"

        # 4. Target Tracking (Click on screen)
        if init_target:
            try:
                tracker = cv2.TrackerCSRT_create()
            except AttributeError:
                tracker = cv2.legacy.TrackerCSRT_create()
                
            box_size = 50
            tx, ty = target_xy
            bbox = (max(0, tx - box_size//2), max(0, ty - box_size//2), box_size, box_size)
            tracker.init(frame, bbox)
            init_target = False

        if tracker is not None:
            ok, bbox = tracker.update(frame)
            if ok:
                target_xy = (int(bbox[0] + bbox[2]//2), int(bbox[1] + bbox[3]//2))
            else:
                tracker = None

        # 5. Render RGB Panel with Room Boundary Wireframe & HUD
        rgb_panel = room_localizer.draw_room_overlay(frame, room_info)

        # Overlay target tracking marker if set
        if target_xy is not None:
            tx, ty = target_xy
            if 0 <= ty < H and 0 <= tx < W:
                # Approximate distance to target
                target_d = room_localizer.depth_scale / max(1e-3, float(depth_map[ty, tx]))
                cam_x, cam_y = W // 2, H
                cv2.line(rgb_panel, (cam_x, cam_y), (tx, ty), (255, 255, 255), 2)
                cv2.circle(rgb_panel, (tx, ty), 6, (0, 0, 255), -1)
                cv2.putText(rgb_panel, f"Target: {target_d:.1f}m", (tx + 8, ty - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 2)

                # Draw on depth panel as well
                cv2.circle(depth_color, (tx, ty), 6, (255, 255, 255), -1)
                cv2.putText(depth_color, f"{target_d:.1f}m", (tx + 8, ty - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2)

        # Overlay PoseNet info if loaded
        if pos_text_xy:
            cv2.putText(rgb_panel, f"{pos_text_xy} {pos_text_z}", (10, H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 255, 180), 1, cv2.LINE_AA)

        # FPS overlay
        t_now = time.time()
        dt = max(1e-3, t_now - last_time)
        fps = 0.9 * fps + 0.1 / dt
        last_time = t_now
        cv2.putText(rgb_panel, f"{fps:.1f} FPS", (W - 65, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

        # 6. Render Depth Panel Header
        cv2.rectangle(depth_color, (0, 0), (W, 22), (20, 20, 20), -1)
        cv2.putText(depth_color, "DEPTH ESTIMATION (MiDaS)", (8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # 7. Render Top-Down Room Radar (BEV) Panel
        radar_panel = room_localizer.render_bev_radar(room_info, width=W, height=H)

        # 8. Combine into 3-Panel Panoramic View
        combined = np.hstack((rgb_panel, depth_color, radar_panel))
        cv2.imshow(win_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            target_xy = None
            tracker = None
            print("[Target] Cleared.")
        elif key in [ord("+"), ord("=")]:
            room_localizer.depth_scale += 50.0
            print(f"[Calibration] Depth Scale: {room_localizer.depth_scale:.0f}")
        elif key in [ord("-"), ord("_")]:
            room_localizer.depth_scale = max(100.0, room_localizer.depth_scale - 50.0)
            print(f"[Calibration] Depth Scale: {room_localizer.depth_scale:.0f}")

    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default=None, help="Path to PoseNet weights (optional)")
    p.add_argument("--onnx", dest="onnx_path", default=None, help="Path to PoseNet ONNX model (optional)")
    p.add_argument("--camera", type=int, default=None, help="Webcam index (e.g. 0)")
    p.add_argument("--url", default=None, help="HTTP stream endpoint (e.g. http://192.168.137.210/capture)")
    p.add_argument("--scale", type=float, default=500.0, help="Depth scale factor (default: 500)")
    
    a = p.parse_args()

    pred = None
    if a.weights or a.onnx_path:
        pred = CoreLocalizer(a.weights, a.onnx_path)
        
    if a.camera is None and a.url is None:
        print("No source specified. Defaulting to camera 0")
        a.camera = 0

    run_stream(pred, a.camera, a.url, depth_scale=a.scale)


if __name__ == "__main__":
    main()
