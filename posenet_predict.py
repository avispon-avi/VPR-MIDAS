#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time PoseNet prediction (7-Scenes 'office') from exported weights.

Runs on a laptop / PC. Load either the PyTorch export or the ONNX export
produced by posenet_train_colab.py:

    checkpoints/posenet_inference.pth   (PyTorch)
    checkpoints/posenet.onnx            (ONNX — input RGB 0..255 float,
                                          output position in meters + quat)

Examples:
    # single image -> prints pose, saves pose_overlay.jpg
    python posenet_predict.py --weights checkpoints/posenet_inference.pth --image frame.png

    # a folder of frames -> poses.csv + trajectory.png
    python posenet_predict.py --onnx checkpoints/posenet.onnx --folder frames/

    # live webcam (press q to quit) -> live overlay + trajectory.png + poses.csv
    python posenet_predict.py --weights checkpoints/posenet_inference.pth --camera 0

    # XIAO ESP32 camera stream (WiFi camera server)
    python posenet_predict.py --onnx checkpoints/posenet.onnx --url http://192.168.4.1/capture

Output pose = camera position (x, y, z) in meters + orientation quaternion
(x, y, z, w) in the 7-Scenes world frame of the training scene.
"""

import argparse
import os
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE_SHORT = 256
CROP = 224


# ---------------------------------------------------------------- model
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
        self.fc_xyz = nn.Linear(2048, 3)
        self.fc_quat = nn.Linear(2048, 4)

    def forward(self, x):
        f = self.fc(self.features(x))
        return self.fc_xyz(f), F.normalize(self.fc_quat(f), p=2, dim=1)


# ---------------------------------------------------------------- image IO
def load_rgb(path):
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.asarray(Image.open(path).convert("RGB"))


def resize_short_side(img, short=RESIZE_SHORT):
    h, w = img.shape[:2]
    scale = short / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if cv2 is not None:
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))


def preprocess(img):
    """RGB uint8 HxWx3 -> dict with (1,3,224,224) tensors:
    'raw' float 0..255 (ONNX graph bakes its own normalization)
    'norm' ImageNet-normalized (PyTorch path)"""
    img = resize_short_side(img)
    h, w = img.shape[:2]
    top, left = (h - CROP) // 2, (w - CROP) // 2
    img = img[top:top + CROP, left:left + CROP]
    x = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32)
    raw = torch.from_numpy(x).unsqueeze(0)
    norm = raw / 255.0
    norm = (norm - torch.from_numpy(IMAGENET_MEAN).view(1, 3, 1, 1)) / \
           torch.from_numpy(IMAGENET_STD).view(1, 3, 1, 1)
    return {"raw": raw, "norm": norm}


# ---------------------------------------------------------------- predictor
class PoseNetPredictor:
    def __init__(self, weights=None, onnx_path=None):
        if bool(weights) == bool(onnx_path):
            raise SystemExit("give exactly one of --weights or --onnx")
        if onnx_path:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"])
            self.mode = "onnx"
            print(f"[predict] ONNX session ready ({onnx_path})")
        else:
            ck = torch.load(weights, map_location="cpu", weights_only=False)
            self.model = PoseNet(ck.get("backbone", "mobilenet_v3_small"))
            self.model.load_state_dict(ck["model_state"])
            self.model.eval()
            self.pos_mean = np.asarray(ck["pos_mean"], dtype=np.float32)
            self.pos_std = np.asarray(ck["pos_std"], dtype=np.float32)
            self.mode = "torch"
            print(f"[predict] PyTorch model ready ({weights})")

    def __call__(self, img_rgb):
        """img_rgb: uint8 HxWx3 -> (xyz meters (3,), quat x,y,z,w (4,))"""
        t = preprocess(img_rgb)
        if self.mode == "onnx":
            pos, quat = self.sess.run(
                None, {"image": t["raw"].numpy()})  # graph outputs meters
            return pos[0], quat[0]
        with torch.no_grad():
            p, q = self.model(t["norm"])
        pos = p.numpy()[0] * self.pos_std + self.pos_mean
        return pos, q.numpy()[0]


# ---------------------------------------------------------------- modes
def load_gt_pose(pose_path):
    """7-Scenes 4x4 camera-to-world pose file -> (xyz meters, quat x,y,z,w)."""
    m = np.loadtxt(pose_path).reshape(4, 4)
    xyz = m[:3, 3].astype(np.float64)
    R = m[:3, :3]
    tr = float(np.trace(R))  # in [-1, 3] for a rotation matrix — no clamping
    w = np.sqrt(max(0.0, 1.0 + tr)) / 2.0
    if w > 1e-8:
        q = np.array([(R[2, 1] - R[1, 2]) / (4 * w),
                      (R[0, 2] - R[2, 0]) / (4 * w),
                      (R[1, 0] - R[0, 1]) / (4 * w), w])
    else:  # ~180-degree rotation
        tr = np.trace(R)
        d = np.diag(R)
        k = int(np.argmax(d))
        t = np.sqrt(max(0.0, 1.0 + 2 * d[k] - tr))
        q = np.zeros(4)
        q[k] = t / 2.0
        for j in range(3):
            if j != k:
                q[j] = (R[k, j] + R[j, k]) / (2.0 * t)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return xyz, q


def orientation_error_deg(q_gt, q_pred):
    d = np.clip(np.abs(np.dot(q_gt, q_pred)), 0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(d)))


def find_gt_pose(image_path):
    """Sibling .pose.txt for a 7-Scenes frame, if present."""
    stem = os.path.splitext(image_path)[0]
    for cand in (stem + ".pose.txt",
                 stem.rsplit(".color", 1)[0] + ".pose.txt"):
        if os.path.isfile(cand):
            return cand
    return None


def save_trajectory(xs, zs, out="trajectory.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(xs, zs, ".-", ms=3, lw=0.5)
    if len(xs):
        ax.scatter(xs[-1], zs[-1], c="r", s=80, zorder=5, label="current")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_title("PoseNet real-time trajectory")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"[predict] trajectory -> {out}")


def run_folder(pred, folder, plot_out):
    """Predict every frame in a folder. If sibling 7-Scenes .pose.txt files
    exist, ALSO validates against ground truth (median/mean position error,
    median orientation error) — same code path as the live camera, so this
    is the end-to-end validation of the real-time pipeline."""
    files = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        raise SystemExit(f"no images in {folder}")
    rows, errs_m, errs_deg = [], [], []
    for i, f in enumerate(files):
        t0 = time.time()
        pos, quat = pred(load_rgb(f))
        dt = (time.time() - t0) * 1000
        line = (f"[{i+1:4d}/{len(files)}] {os.path.basename(f):28s} "
                f"xyz=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}) m  "
                f"quat=({quat[0]:+.2f},{quat[1]:+.2f},{quat[2]:+.2f},{quat[3]:+.2f})  "
                f"[{dt:.0f} ms]")
        gt_file = find_gt_pose(f)
        if gt_file:
            gt_xyz, gt_q = load_gt_pose(gt_file)
            e = float(np.linalg.norm(pos - gt_xyz))
            errs_m.append(e)
            errs_deg.append(orientation_error_deg(gt_q, quat))
            line += f"  | err {e*100:6.1f} cm"
        print(line)
        rows.append(np.concatenate([pos, quat]))
    arr = np.array(rows)
    np.savetxt("poses.csv", arr, delimiter=",",
               header="x_m,y_m,z_m,qx,qy,qz,qw", comments="")
    print("[predict] poses -> poses.csv")
    if errs_m:
        errs_m, errs_deg = np.array(errs_m), np.array(errs_deg)
        print("\n" + "=" * 60)
        print(f" VALIDATION vs ground truth ({len(errs_m)}/{len(files)} frames)")
        print("=" * 60)
        print(f" median position error : {np.median(errs_m)*100:7.1f} cm")
        print(f" mean   position error : {np.mean(errs_m)*100:7.1f} cm")
        print(f" 95th  percentile     : {np.percentile(errs_m,95)*100:7.1f} cm")
        print(f" median orientation   : {np.median(errs_deg):7.1f} deg")
        print(f" within 0.5 m / 1.0 m : "
              f"{np.mean(errs_m<0.5)*100:.0f}% / {np.mean(errs_m<1.0)*100:.0f}%")
        print("=" * 60)
    save_trajectory(arr[:, 0], arr[:, 2], plot_out)


def run_image(pred, image_path):
    img = load_rgb(image_path)
    pos, quat = pred(img)
    print(f"position (m)   : x={pos[0]:+.3f}  y={pos[1]:+.3f}  z={pos[2]:+.3f}")
    print(f"orientation    : quat x,y,z,w = "
          f"({quat[0]:+.3f}, {quat[1]:+.3f}, {quat[2]:+.3f}, {quat[3]:+.3f})")
    if cv2 is not None:
        disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(disp, f"x={pos[0]:+.2f} y={pos[1]:+.2f} z={pos[2]:+.2f} m",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite("pose_overlay.jpg", disp)
        print("[predict] overlay -> pose_overlay.jpg")


def run_stream(pred, camera=None, url=None, ema=0.0):
    if cv2 is None:
        raise SystemExit("--camera/--url need OpenCV: pip install opencv-python")
    import requests
    cap = None
    if camera is not None:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise SystemExit(f"cannot open camera {camera}")
    xs, zs, fps, smooth = [], [], 0.0, None
    print("[predict] streaming — press q in the window to quit")
    cv2.namedWindow("PoseNet — q to quit", cv2.WINDOW_NORMAL)
    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            try:
                r = requests.get(url, timeout=3)
                frame = cv2.imdecode(np.frombuffer(r.content, np.uint8),
                                     cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception as e:
                print(f"fetch error: {e}"); time.sleep(0.5); continue
        
        t0 = time.time()
        pos, quat = pred(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if ema > 0:  # exponential smoothing: higher = smoother but laggier
            smooth = pos.copy() if smooth is None else ema * smooth + (1 - ema) * pos
            pos = smooth
        fps = 0.9 * fps + 0.1 / max(1e-3, time.time() - t0)
        xs.append(pos[0]); zs.append(pos[2])
        disp = frame.copy()
        # Scale up the frame so the UI text and map overlay fit properly
        disp = cv2.resize(disp, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        cv2.putText(disp, f"x={pos[0]:+.2f} y={pos[1]:+.2f} z={pos[2]:+.2f} m",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(disp, f"{fps:.1f} FPS", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # small top-down inset
        if len(xs) > 1:
            px, py, pw, ph = 10, disp.shape[0] - 160, 140, 140
            cv2.rectangle(disp, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
            xs_a, zs_a = np.array(xs), np.array(zs)
            mx = px + pw / 2 + (xs_a - xs_a.mean()) * (pw / 3 / max(1e-3, xs_a.std() + 1))
            my = py + ph / 2 + (zs_a - zs_a.mean()) * (ph / 3 / max(1e-3, zs_a.std() + 1))
            pts = np.stack([mx, my], 1).astype(np.int32)
            cv2.polylines(disp, [pts], False, (0, 255, 0), 1)
            cv2.circle(disp, tuple(pts[-1]), 4, (0, 0, 255), -1)
        cv2.imshow("PoseNet — q to quit", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass  # headless OpenCV has no GUI; outputs are still saved below
    if xs:
        np.savetxt("poses.csv", np.array([xs, zs]).T, delimiter=",",
                   header="x_m,z_m", comments="")
        save_trajectory(xs, zs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=None, help="posenet_inference.pth")
    p.add_argument("--onnx", dest="onnx_path", default=None, help="posenet.onnx")
    p.add_argument("--image", default=None)
    p.add_argument("--folder", default=None)
    p.add_argument("--camera", type=int, default=None, help="webcam index, e.g. 0")
    p.add_argument("--url", default=None, help="http stream endpoint (XIAO)")
    p.add_argument("--ema", type=float, default=0.0,
                   help="position smoothing for streams, e.g. 0.7 "
                        "(higher = smoother, laggier)")
    p.add_argument("--plot", default="trajectory.png")
    a = p.parse_args()
    pred = PoseNetPredictor(a.weights, a.onnx_path)
    if a.image:
        run_image(pred, a.image)
    elif a.folder:
        run_folder(pred, a.folder, a.plot)
    elif a.camera is not None or a.url:
        run_stream(pred, a.camera, a.url, a.ema)
    else:
        raise SystemExit("choose --image, --folder, --camera or --url")


if __name__ == "__main__":
    main()
