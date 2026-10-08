#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time MiDaS monocular depth (zero-shot, works in ANY room) from the
XIAO ESP32-S3 camera stream, a webcam, or image files.

MiDaS predicts relative INVERSE depth (higher value = closer to camera).
It transfers zero-shot to unseen rooms (Ranftl et al., TPAMI 2020) — unlike
PoseNet, no per-room training is needed. It does NOT output metric position
by itself; it is the depth foundation for a room-agnostic relative-localization
layer.

Model choices (--model):
    DPT_Large   best quality  (~1.4 GB, GPU recommended)
    DPT_Hybrid  middle        (~0.5 GB, GPU recommended)
    MiDaS_small fastest      (~82 MB, fine on CPU)
    Default: DPT_Large if CUDA is available, else MiDaS_small.
    All models need:  pip install timm  (the MiDaS code imports it).

Examples:
    python midas_live.py --url http://192.168.4.1/capture          # XIAO stream
    python midas_live.py --camera 0                                # webcam
    python midas_live.py --image frame.png --save-out depth.png    # single frame
    python midas_live.py --folder frames/ --save-dir depth_out/    # batch

Weights download automatically to the torch hub cache the first time:
    C:\\Users\\<you>\\.cache\\torch\\hub\\checkpoints\\
If your network blocks the download, fetch the .pt manually from
https://github.com/isl-org/MiDaS/releases and drop it (unrenamed) into that
folder, then rerun — the script prints the exact expected path on failure.
"""

import argparse
import os
import time

import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image

WEIGHT_FILES = {
    "DPT_Large": "dpt_large_384.pt",
    "DPT_Hybrid": "dpt_hybrid_384.pt",
    "MiDaS_small": "midas_v21_small_256.pt",
}
WEIGHT_URLS = {
    "DPT_Large": "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt",
    "DPT_Hybrid": "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt",
    "MiDaS_small": "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt",
}


# ---------------------------------------------------------------- MiDaS
def _seed_trusted_repos():
    """MiDaS internally does torch.hub.load('rwightman/gen-efficientnet-pytorch',
    ...) for the small model's encoder, which would prompt for trust and hang
    non-interactive runs. Pre-mark the repos it needs as trusted."""
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


def load_midas(model_type, device):
    """torch.hub loader with manual-weights fallback instructions on failure."""
    _seed_trusted_repos()
    last_err = None
    for repo in ("isl-org/MiDaS", "intel-isl/MiDaS"):
        try:
            midas = torch.hub.load(repo, model_type, trust_repo=True,
                                   skip_validation=True)
            transforms = torch.hub.load(repo, "transforms", trust_repo=True,
                                        skip_validation=True)
            transform = (transforms.dpt_transform
                         if model_type.startswith("DPT")
                         else transforms.small_transform)
            return midas.to(device).eval(), transform
        except Exception as e:  # noqa: BLE001 — report the last failure
            last_err = e

    ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    raise SystemExit(
        f"\n[midas] could not load {model_type}: {last_err}\n"
        f"[midas] Fix steps:\n"
        f"[midas]   1. install deps:            pip install timm\n"
        f"[midas]   2. if the weight download was blocked, download\n"
        f"[midas]      {WEIGHT_URLS[model_type]}\n"
        f"[midas]      and put it (unrenamed) in:  {ckpt_dir}\n"
        f"[midas]   3. rerun this command"
    )


def resize_short_side(img, short=256):
    h, w = img.shape[:2]
    scale = short / min(h, w)
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def load_rgb(path):
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.asarray(Image.open(path).convert("RGB"))


class DepthEstimator:
    def __init__(self, model_type, device):
        self.device = device
        self.model, self.transform = load_midas(model_type, device)
        self.model_type = model_type
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[midas] {model_type} ready: {n_params/1e6:.1f}M params on {device}")

    @torch.no_grad()
    def __call__(self, img_rgb):
        """RGB uint8 HxWx3 -> relative inverse depth, same HxW (higher=closer)."""
        x = self.transform(img_rgb).to(self.device)
        pred = self.model(x)
        depth = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=img_rgb.shape[:2],
            mode="bicubic", align_corners=False,
        ).squeeze().float().cpu().numpy()
        return depth


def colorize(depth):
    d = depth - depth.min()
    m = d.max() - d.min()
    if m > 1e-9:
        d = d / m
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def center_depth(depth, frac=0.2):
    """Median inverse depth in the central ROI (relative units, higher=closer)."""
    h, w = depth.shape
    ch, cw = int(h * frac), int(w * frac)
    roi = depth[h // 2 - ch // 2:h // 2 + ch // 2,
                w // 2 - cw // 2:w // 2 + cw // 2]
    return float(np.median(roi))


def annotate(frame_rgb, depth, fps, label):
    disp_rgb = cv2.resize(frame_rgb,
                          (int(frame_rgb.shape[1] * 480 / frame_rgb.shape[0]), 480))
    disp_dep = cv2.resize(colorize(depth), (disp_rgb.shape[1], disp_rgb.shape[0]))
    combo = np.hstack([disp_rgb, disp_dep])
    cd = center_depth(depth)
    cv2.putText(combo, f"{fps:4.1f} FPS  |  {label}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combo, f"center rel-depth {cd:5.2f} (higher = closer)",
                (10, combo.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA)
    return combo


# ---------------------------------------------------------------- modes
def run_image(est, image_path, save_out):
    img = load_rgb(image_path)
    t0 = time.time()
    depth = est(img)
    print(f"[midas] {os.path.basename(image_path)}: inference "
          f"{(time.time()-t0)*1000:.0f} ms | center rel-depth "
          f"{center_depth(depth):.2f} (higher = closer)")
    cv2.imwrite(save_out, annotate(img, depth, 0.0, os.path.basename(image_path)))
    print(f"[midas] visualization -> {save_out}")


def run_folder(est, folder, save_dir):
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        raise SystemExit(f"no images in {folder}")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for i, f in enumerate(files):
        img = load_rgb(os.path.join(folder, f))
        t0 = time.time()
        depth = est(img)
        name = os.path.splitext(f)[0]
        print(f"[{i+1:3d}/{len(files)}] {f:32s} "
              f"{(time.time()-t0)*1000:6.0f} ms | center rel-depth "
              f"{center_depth(depth):.2f}")
        if save_dir:
            out = os.path.join(save_dir, name + "_depth.png")
            cv2.imwrite(out, annotate(img, depth, 0.0, f))
            print(f"      -> {out}")


def run_stream(est, camera=None, url=None, ema=0.0, display_h=480):
    if cv2 is None:
        raise SystemExit("--camera/--url need OpenCV: pip install opencv-python")
    import requests
    cap = None
    if camera is not None:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise SystemExit(f"cannot open camera {camera}")
    fps, smooth = 0.0, None
    csv_rows = [("t_s", "fps", "center_rel_depth")]
    t_start = time.time()
    print("[midas] streaming — press q in the window to quit")
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
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth = est(img)
        if ema > 0:
            smooth = depth.astype(np.float32) if smooth is None \
                else ema * smooth + (1 - ema) * depth.astype(np.float32)
            depth = smooth
        fps = 0.9 * fps + 0.1 / max(1e-3, time.time() - t0)
        cd = center_depth(depth)
        csv_rows.append((f"{time.time()-t_start:.2f}", f"{fps:.2f}", f"{cd:.4f}"))

        combo = annotate(img, depth, fps, "MiDaS relative depth (zero-shot)")
        cv2.imshow("MiDaS live depth — q to quit", combo)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    with open("depth_session.csv", "w") as f:
        for row in csv_rows:
            f.write(",".join(row) + "\n")
    print(f"[midas] session log -> depth_session.csv ({len(csv_rows)-1} frames)")
    if csv_rows[-1][0] != "t_s":
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=None, help="XIAO capture endpoint")
    p.add_argument("--camera", type=int, default=None, help="webcam index")
    p.add_argument("--image", default=None)
    p.add_argument("--folder", default=None)
    p.add_argument("--model", default=None, choices=list(WEIGHT_FILES),
                   help="default: DPT_Large on GPU, MiDaS_small on CPU")
    p.add_argument("--ema", type=float, default=0.0,
                   help="temporal depth smoothing for streams, e.g. 0.5")
    p.add_argument("--save-out", default="midas_depth.png")
    p.add_argument("--save-dir", default=None)
    a = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("[midas] no CUDA — CPU inference (use MiDaS_small for speed)")
    model_type = a.model or ("DPT_Large" if device.type == "cuda"
                             else "MiDaS_small")
    est = DepthEstimator(model_type, device)

    if a.image:
        run_image(est, a.image, a.save_out)
    elif a.folder:
        run_folder(est, a.folder, a.save_dir)
    elif a.camera is not None or a.url:
        run_stream(est, a.camera, a.url, a.ema)
    else:
        raise SystemExit("choose --url, --camera, --image or --folder")


if __name__ == "__main__":
    main()
