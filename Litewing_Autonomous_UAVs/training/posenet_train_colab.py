#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PoseNet training + evaluation on Microsoft 7-Scenes (office scene)
for Google Colab GPU (T4). Runs as a plain terminal script.

Pipeline (mirrors the original caffe-posenet recipe, ported to PyTorch
with a MobileNetV3 backbone for later ESP32 deployment):

  1. Fetch office.zip (Google Drive via gdown, or a local zip path)
  2. Extract outer zip + nested seq-XX.zip files
  3. One-off preprocessing cache: resize shortest side to 256, store
     uint8 frames + poses (pos in meters, quaternion x,y,z,w)
  4. Train PoseNet  =  backbone -> FC 2048 -> ReLU -> Dropout -> (xyz, quat)
     Loss           =  ||x - x^||^2 + beta * ||q - q^||^2      (beta = 500 in
     the original GoogLeNet paper; 100 is a good start for MobileNetV3)
  5. Evaluate exactly like posenet/scripts/test_posenet.py:
     pos error  = ||x - x^||            [meters]
     rot error  = 2*arccos(|q . q^|)    [degrees]
     reported as medians + percentile table + threshold accuracies + plots.

Usage on Colab (GPU runtime):
    pip install -q gdown
    python posenet_train_colab.py --epochs 100
    # smoke test first:
    python posenet_train_colab.py --epochs 1 --limit 400
    # evaluate a saved checkpoint later:
    python posenet_train_colab.py --eval-only

Artifacts land in --work-dir (default /content/posenet_7scenes):
    checkpoints/posenet_best.pth   best model (by median test position error)
    checkpoints/posenet_last.pth   resumable latest model
    checkpoints/posenet_inference.pth  clean deployable weights + norm stats
    checkpoints/posenet.onnx       ONNX graph, input RGB 0-255 (N,3,224,224),
                                   output position (meters) + unit quaternion
    checkpoints/pos_mean.npy / pos_std.npy
    logs/metrics.csv               per-epoch history
    logs/results.txt               per-frame (err_m, err_deg) like the repo
    logs/eval_report.txt           final metric table
    logs/curves.png, logs/trajectory.png, logs/error_hist.png
If Google Drive is mounted at /content/drive, everything is also copied to
MyDrive/posenet_7scenes_office after training (survives Colab resets).
"""

import argparse
import math
import os
import shutil
import sys
import time
import zipfile

import numpy as np

# ----------------------------------------------------------------------------
# Optional deps (Colab has all of them; PIL is the fallback if no cv2)
# ----------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DEFAULT_GDRIVE_ID = "1_m2MMCYiQCyFPBJrJzfgbfqiTED0lPDm"  # user's office.zip


# ============================================================================
# 1. Dataset acquisition
# ============================================================================
def download_office_zip(args, zip_path):
    """Get office.zip from: --zip-path, mounted Drive, or gdown (in that order)."""
    if args.zip_path and os.path.isfile(args.zip_path):
        shutil.copy(args.zip_path, zip_path)
        return zip_path

    # If Drive is mounted, look for an already-uploaded copy (most reliable).
    for cand in [
        "/content/drive/MyDrive/office.zip",
        "/content/drive/MyDrive/7scenes/office.zip",
        os.path.expanduser("~/office.zip"),
    ]:
        if os.path.isfile(cand):
            print(f"[data] using Drive copy: {cand}")
            shutil.copy(cand, zip_path)
            return zip_path

    import gdown
    print(f"[data] downloading office.zip from Google Drive id={args.gdrive_id}")
    try:
        gdown.download(id=args.gdrive_id, output=str(zip_path), quiet=False)
    except Exception:
        url = f"https://drive.google.com/uc?id={args.gdrive_id}"
        gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
    if not os.path.isfile(zip_path) or os.path.getsize(zip_path) < 1e6:
        raise RuntimeError(
            "Download failed. Either enable link-sharing on the Drive file, or "
            "mount Drive and pass --zip-path '/content/drive/MyDrive/office.zip'."
        )
    return zip_path


def parse_split_file(path):
    """Return ['seq-01', ...] from TrainSplit/TestSplit.

    Handles both 'sequence1' (your local copy) and 'office seq-01'
    (official 7-Scenes) line formats.
    """
    seqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok = line.split()[-1]
            if tok.lower().startswith("sequence"):
                n = int("".join(c for c in tok if c.isdigit()))
                tok = f"seq-{n:02d}"
            if not tok.startswith("seq-"):
                continue
            seqs.append(tok)
    return seqs


def find_scene_root(base_dir):
    """Locate the directory that contains TrainSplit.txt (handles any nesting)."""
    for root, _dirs, files in os.walk(base_dir):
        if "TrainSplit.txt" in files:
            return root
    raise FileNotFoundError(f"No TrainSplit.txt found under {base_dir}")


JUNK_NAMES = {"thumbs.db", ".ds_store", "desktop.ini"}


def safe_extract(zf, dest):
    """Extract members one by one: OS junk (Thumbs.db etc.) is skipped outright,
    and members with bad CRCs are skipped with a warning instead of aborting
    the whole extraction."""
    skipped = []
    members = [
        i for i in zf.infolist()
        if not i.is_dir()
        and "__MACOSX" not in i.filename
        and os.path.basename(i.filename).lower() not in JUNK_NAMES
    ]
    for info in members:
        try:
            zf.extract(info, dest)
        except Exception as e:
            skipped.append(f"{info.filename} ({e})")
    return skipped


def prepare_dataset(args):
    """Unzip outer archive + nested seq zips. Returns scene root path
    (recorded in .prepared so later runs reuse it)."""
    marker = os.path.join(args.work_dir, "data", ".prepared")
    if os.path.isfile(marker):
        with open(marker) as f:
            return f.read().strip()

    data_dir = os.path.join(args.work_dir, "data")
    zip_path = os.path.join(data_dir, "office.zip")
    found = [
        r for r, _d, fs in os.walk(data_dir) if "TrainSplit.txt" in fs
    ] if os.path.isdir(data_dir) else []

    if not found:
        os.makedirs(data_dir, exist_ok=True)
        download_office_zip(args, zip_path)
        print("[data] extracting outer zip ...")
        with zipfile.ZipFile(zip_path) as z:
            skipped = safe_extract(z, data_dir)
        for s in skipped:
            print(f"[data]   skipped corrupt member: {s}")
        found = [
            r for r, _d, fs in os.walk(data_dir) if "TrainSplit.txt" in fs
        ]
        if not found:
            raise FileNotFoundError("office.zip did not contain TrainSplit.txt")
    scene_dir = found[0]

    # Nested per-sequence archives (official distribution layout).
    # Extract to a temp dir, then swap in, so a partially-extracted seq dir
    # from an earlier crashed run is never mistaken for a complete one.
    seq_zips = sorted(
        f for f in os.listdir(scene_dir) if f.startswith("seq-") and f.endswith(".zip")
    )
    for sz in seq_zips:
        tgt = os.path.join(scene_dir, sz[:-4])
        tmp = os.path.join(scene_dir, sz[:-4] + ".extracting")
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[data] extracting {sz} ...")
        with zipfile.ZipFile(os.path.join(scene_dir, sz)) as z:
            skipped = safe_extract(z, tmp)
        for s in skipped:
            print(f"[data]   skipped corrupt member: {s}")
        inner = os.path.join(tmp, sz[:-4])
        src = inner if os.path.isdir(inner) else tmp
        n_frames = len([f for f in os.listdir(src) if f.endswith(".color.png")])
        print(f"[data]   {n_frames} color frames")
        if n_frames == 0:
            print(f"[data] WARNING: {sz} yielded no frames — check the zip")
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        shutil.rmtree(tgt, ignore_errors=True)
        shutil.move(src, tgt)
        shutil.rmtree(tmp, ignore_errors=True)
        os.remove(os.path.join(scene_dir, sz))

    with open(marker, "w") as f:
        f.write(scene_dir + "\n")
    if os.path.isfile(zip_path):
        os.remove(zip_path)  # free ~5 GB; extracted data is all we need
    return scene_dir


# ============================================================================
# 2. Pose / image loading + one-off cache
# ============================================================================
def load_pose(pose_path):
    """4x4 camera-to-world matrix -> (xyz[m], quat xyzw canonical, valid)."""
    m = np.loadtxt(pose_path).reshape(4, 4)
    if not np.all(np.isfinite(m)):
        return None
    xyz = m[:3, 3].astype(np.float32)
    # quaternion (x, y, z, w) from rotation matrix
    R = m[:3, :3]
    tr = float(np.trace(R))  # in [-1, 3] for a rotation matrix — no clamping
    w = math.sqrt(max(0.0, 1.0 + tr)) / 2.0
    if w > 1e-8:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
        quat = np.array([x, y, z, w], dtype=np.float32)
    else:  # ~180-degree rotations: w ~ 0, largest-diagonal branch
        d = np.diag(R)
        k = int(np.argmax(d))
        t = math.sqrt(max(0.0, 1 + 2 * d[k] - np.trace(R)))
        q = np.zeros(4)
        q[k] = t / 2
        for j in range(3):
            if j != k:
                q[j] = (R[k, j] + R[j, k]) / (2 * t)
    # canonical hemisphere (q and -q are the same rotation)
    if quat[3] < 0:
        quat = -quat
    quat /= np.linalg.norm(quat)
    return np.concatenate([xyz, quat])  # 7 floats


def load_rgb(path):
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(Image.open(path).convert("RGB"))
    return img


def resize_short_side(img, short=256):
    h, w = img.shape[:2]
    scale = short / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if cv2 is not None:
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))


def build_cache(scene_dir, split_file, cache_dir, limit=0):
    """Pre-resize frames once so every epoch after is GPU-bound, not IO-bound."""
    os.makedirs(cache_dir, exist_ok=True)
    img_f = os.path.join(cache_dir, f"{split_file}_images.npy")
    pose_f = os.path.join(cache_dir, f"{split_file}_poses.npy")
    if os.path.isfile(img_f) and os.path.isfile(pose_f):
        return img_f, pose_f

    seqs = parse_split_file(os.path.join(scene_dir, split_file))
    samples, seq_ids, skipped = [], [], 0
    for sq in seqs:
        seq_dir = os.path.join(scene_dir, sq)
        frames = sorted(
            f for f in os.listdir(seq_dir) if f.endswith(".color.png")
        )
        if limit:
            frames = frames[:limit]
        for fr in frames:
            pose_path = os.path.join(seq_dir, fr[:-len(".color.png")] + ".pose.txt")
            try:
                pose = load_pose(pose_path)
            except (OSError, ValueError):
                pose = None
            if pose is None:
                skipped += 1
                continue
            samples.append((os.path.join(seq_dir, fr), pose))
            seq_ids.append(seqs.index(sq))

    if skipped:
        print(f"[cache] skipped {skipped} frames with invalid poses")

    # probe the first decodable frame to size the cache array
    first = None
    for p, _pose in samples:
        try:
            first = resize_short_side(load_rgb(p))
            break
        except Exception:
            continue
    if first is None:
        raise RuntimeError(f"no readable frames for {split_file}")
    n = len(samples)
    hh, ww = first.shape[:2]
    print(f"[cache] {split_file}: {n} frames -> {hh}x{ww} uint8 cache")
    images = np.empty((n, hh, ww, 3), dtype=np.uint8)
    poses = np.empty((n, 7), dtype=np.float32)
    valid = []
    it = tqdm(samples, desc=f"cache {split_file}") if tqdm else samples
    for i, (p, pose) in enumerate(it):
        try:
            images[i] = first if (i == 0 and p == samples[0][0]) else resize_short_side(load_rgb(p))
            poses[i] = pose
            valid.append(i)
        except Exception as e:
            print(f"\n[cache] dropping unreadable frame {p}: {e}")
    if len(valid) < n:
        print(f"[cache] dropped {n - len(valid)} unreadable frames")
        images, poses = images[valid], poses[valid]
    np.save(img_f, images)
    np.save(pose_f, poses)
    return img_f, pose_f


# ============================================================================
# 3. Dataset
# ============================================================================
class SevenScenesCached(Dataset):
    """Random 224x224 crop (train) / center crop (test); no mirroring —
    a horizontal flip changes the pose, so PoseNet never mirrors."""

    def __init__(self, images_npy, poses_npy, pos_mean, pos_std, train=True,
                 image_size=224):
        self.images = np.load(images_npy, mmap_mode="r")
        self.poses = np.load(poses_npy)
        self.pos_mean, self.pos_std = pos_mean, pos_std
        self.train = train
        self.size = image_size

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        img = np.asarray(self.images[idx])  # HxWx3 uint8
        h, w = img.shape[:2]
        c = self.size
        if self.train:
            top = np.random.randint(0, h - c + 1)
            left = np.random.randint(0, w - c + 1)
        else:
            top, left = (h - c) // 2, (w - c) // 2
        img = img[top:top + c, left:left + c, :]

        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

        xyz, quat = self.poses[idx, :3], self.poses[idx, 3:]
        xyz = (xyz - self.pos_mean) / self.pos_std  # normalized target
        return x, torch.from_numpy(xyz.astype(np.float32)), torch.from_numpy(quat.copy())


# ============================================================================
# 4. Model
# ============================================================================
class PoseNet(nn.Module):
    """backbone -> FC 2048 -> ReLU -> Dropout -> (xyz, unit quat).

    Same head shape as the original GoogLeNet PoseNet's final branch, on a
    MobileNetV3 backbone (chosen for the eventual ESP32 deployment).
    """

    FEAT_DIMS = {
        "mobilenet_v3_small": 576,
        "mobilenet_v3_large": 960,
        "resnet18": 512,
    }

    def __init__(self, backbone="mobilenet_v3_small", dropout=0.5, pretrained=True):
        super().__init__()
        import torchvision.models as tvm
        if backbone == "mobilenet_v3_small":
            base = tvm.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
            self.features = base.features
        elif backbone == "mobilenet_v3_large":
            base = tvm.mobilenet_v3_large(weights="IMAGENET1K_V1" if pretrained else None)
            self.features = base.features
        elif backbone == "resnet18":
            base = tvm.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            self.features = nn.Sequential(*list(base.children())[:-2])
        else:
            raise ValueError(f"unknown backbone {backbone}")
        feat = self.FEAT_DIMS[backbone]

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc_xyz = nn.Linear(2048, 3)
        self.fc_quat = nn.Linear(2048, 4)

    def forward(self, x):
        f = self.fc(self.features(x))
        return self.fc_xyz(f), F.normalize(self.fc_quat(f), p=2, dim=1)


class PoseNetLoss(nn.Module):
    """L = ||x - x^||^2 + beta * ||q - q^||^2  (original: beta = 500)."""

    def __init__(self, beta=500.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred_xyz, pred_q, gt_xyz, gt_q):
        pos_l = (pred_xyz - gt_xyz).pow(2).sum(dim=1).mean()
        quat_l = (pred_q - gt_q).pow(2).sum(dim=1).mean()
        return pos_l + self.beta * quat_l, pos_l.detach(), quat_l.detach()


# ============================================================================
# 4b. Model summary (params / size / MACs) + inference export
# ============================================================================
def profile_macs(model, image_size=224):
    """Multiply-accumulate count for Conv2d + Linear layers at 1x3xSxS input."""
    macs = 0
    handles = []

    def conv_hook(m, _inp, out):
        nonlocal macs
        k = m.kernel_size
        kk = k[0] * k[1] if isinstance(k, tuple) else k * k
        macs += out.numel() * (m.in_channels // m.groups) * kk

    def lin_hook(m, _inp, out):
        nonlocal macs
        macs += m.in_features * m.out_features * (out.numel() // m.out_features)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(lin_hook))
    training = model.training
    model.eval()
    dev = next(model.parameters()).device  # model may already be on CUDA
    with torch.no_grad():
        model(torch.zeros(1, 3, image_size, image_size, device=dev))
    for h in handles:
        h.remove()
    if training:
        model.train()
    return macs


def model_summary(model, backbone, image_size=224):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs = profile_macs(model, image_size)
    n_conv = sum(isinstance(m, nn.Conv2d) for m in model.modules())
    n_lin = sum(isinstance(m, nn.Linear) for m in model.modules())
    lines = [
        "-" * 60,
        f" MODEL SUMMARY  ({backbone}, input {image_size}x{image_size} RGB)",
        "-" * 60,
        f" parameters            : {total:>12,d}  ({total/1e6:.2f} M)",
        f" trainable parameters  : {trainable:>12,d}",
        f" conv / linear layers  : {n_conv:>7d} / {n_lin}",
        f" MACs per inference    : {macs:>12,d}  ({macs/1e6:.1f} M)",
        f" weight size FP32      : {total*4/1e6:>12.2f}  MB",
        f" weight size FP16      : {total*2/1e6:>12.2f}  MB",
        f" weight size INT8      : {total*1/1e6:>12.2f}  MB  (ESP-DL target)",
        "-" * 60,
    ]
    print("\n".join(lines))


class InferenceWrapper(nn.Module):
    """Bakes the whole pre/post-processing into the graph so the exported
    model is self-contained for real-time deployment:

      input  : float32 RGB image, 0..255, shape (N, 3, 224, 224)
      output : position in METERS (N, 3), unit quaternion x,y,z,w (N, 4)
    """

    def __init__(self, net, pos_mean, pos_std):
        super().__init__()
        self.net = net
        self.register_buffer("img_mean", IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("img_std", IMAGENET_STD.view(1, 3, 1, 1))
        self.register_buffer("pos_mean", torch.as_tensor(pos_mean).view(1, 3).float())
        self.register_buffer("pos_std", torch.as_tensor(pos_std).view(1, 3).float())

    def forward(self, image_raw):
        x = image_raw / 255.0
        x = (x - self.img_mean) / self.img_std
        p, q = self.net(x)
        return p * self.pos_std + self.pos_mean, q


def export_for_inference(args):
    """From the best checkpoint, save deployable weights + ONNX + summary."""
    ck_dir = os.path.join(args.work_dir, "checkpoints")
    best = os.path.join(ck_dir, "posenet_best.pth")
    if not os.path.isfile(best):
        print("[export] no posenet_best.pth found — skipping export")
        return
    ck = torch.load(best, map_location="cpu", weights_only=False)
    backbone = ck.get("backbone", args.backbone)
    model = PoseNet(backbone)
    model.load_state_dict(ck["model_state"])
    model.eval()
    pos_mean, pos_std = ck["pos_mean"], ck["pos_std"]
    epoch = ck.get("epoch")

    # 1) clean inference checkpoint (weights + normalization only)
    inf_path = os.path.join(ck_dir, "posenet_inference.pth")
    torch.save({
        "model_state": ck["model_state"],
        "backbone": backbone,
        "pos_mean": pos_mean,
        "pos_std": pos_std,
        "resize_short_side": 256,
        "center_crop": 224,
        "quaternion_order": "x,y,z,w",
        "epoch": epoch,
    }, inf_path)
    print(f"\n[export] inference weights -> {inf_path} "
          f"({os.path.getsize(inf_path)/1e6:.2f} MB, epoch {epoch})")

    # 2) ONNX with pre/post-processing baked in (opset 14: Hardswish support)
    onnx_path = os.path.join(ck_dir, "posenet.onnx")
    try:
        import onnx
        wrapper = InferenceWrapper(model, pos_mean, pos_std).eval()
        dummy = torch.rand(1, 3, 224, 224) * 255.0
        common = dict(
            input_names=["image"], output_names=["position_m", "orientation_quat"],
            dynamic_axes={"image": {0: "batch"},
                          "position_m": {0: "batch"},
                          "orientation_quat": {0: "batch"}},
            opset_version=14,
        )
        try:  # torch >= 2.9 defaults to the dynamo exporter (needs onnxscript)
            torch.onnx.export(wrapper, dummy, onnx_path, dynamo=False, **common)
        except TypeError:  # older torch without the `dynamo` kwarg
            torch.onnx.export(wrapper, dummy, onnx_path, **common)
        onnx.checker.check_model(onnx.load(onnx_path))
        print(f"[export] ONNX graph     -> {onnx_path} "
              f"({os.path.getsize(onnx_path)/1e6:.2f} MB)")
        # numeric verification against PyTorch if onnxruntime is available
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path,
                                        providers=["CPUExecutionProvider"])
            with torch.no_grad():
                t_pos, t_quat = wrapper(dummy)
            o_pos, o_quat = sess.run(None, {"image": dummy.numpy()})
            print(f"[export] ONNX vs PyTorch max|diff|: "
                  f"pos {np.abs(t_pos.numpy()-o_pos).max():.2e} m, "
                  f"quat {np.abs(t_quat.numpy()-o_quat).max():.2e}")
        except ImportError:
            print("[export] (install onnxruntime to auto-verify the graph)")
    except Exception as e:
        print(f"[export] ONNX skipped ({e})")
        print("[export] onnx export is optional — posenet_inference.pth is saved")

    model_summary(model, backbone)


# ============================================================================
# 5. Evaluation (identical math to posenet/scripts/test_posenet.py)
# ============================================================================
def orientation_error_deg(q_gt, q_pred):
    d = np.clip(np.abs(np.sum(q_gt * q_pred, axis=1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


@torch.no_grad()
def evaluate(model, loader, device, pos_mean, pos_std, amp=True):
    model.eval()
    errs_m, errs_deg, val_loss = [], [], 0.0
    for x, xyz, q in loader:
        x = x.to(device, non_blocking=True)
        xyz = xyz.to(device, non_blocking=True)
        q = q.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            p_xyz, p_q = model(x)
            loss = (p_xyz - xyz).pow(2).sum(1).mean() + \
                   (p_q - q).pow(2).sum(1).mean()  # unweighted, for tracking
        val_loss += loss.item() * x.size(0)

        p_xyz = (p_xyz.float().cpu().numpy() * pos_std + pos_mean)
        g_xyz = xyz.cpu().numpy() * pos_std + pos_mean
        errs_m.append(np.linalg.norm(p_xyz - g_xyz, axis=1))
        errs_deg.append(orientation_error_deg(q.cpu().numpy(),
                                              p_q.float().cpu().numpy()))
    errs_m = np.concatenate(errs_m)
    errs_deg = np.concatenate(errs_deg)
    return {
        "val_loss": val_loss / len(loader.dataset),
        "err_m": errs_m,
        "err_deg": errs_deg,
        "median_m": float(np.median(errs_m)),
        "mean_m": float(np.mean(errs_m)),
        "rmse_m": float(np.sqrt(np.mean(errs_m ** 2))),
        "median_deg": float(np.median(errs_deg)),
        "mean_deg": float(np.mean(errs_deg)),
    }


def format_report(m):
    lines = [
        "=" * 62,
        " PoseNet on 7-Scenes 'office' — test-set evaluation",
        "=" * 62,
        f" Samples evaluated          : {len(m['err_m'])}",
        "-" * 62,
        " Position error",
        f"   median                   : {m['median_m']*100:7.2f} cm",
        f"   mean                     : {m['mean_m']*100:7.2f} cm",
        f"   RMSE                     : {m['rmse_m']*100:7.2f} cm",
        f"   75th / 95th percentile   : "
        f"{np.percentile(m['err_m'],75)*100:.2f} / "
        f"{np.percentile(m['err_m'],95)*100:.2f} cm",
        f"   max                      : {np.max(m['err_m'])*100:7.2f} cm",
        " Orientation error",
        f"   median                   : {m['median_deg']:7.2f} deg",
        f"   mean                     : {m['mean_deg']:7.2f} deg",
        "-" * 62,
        " Localization accuracy (fraction of test frames)",
        f"   position < 0.25 m        : {np.mean(m['err_m'] < 0.25)*100:7.2f} %",
        f"   position < 0.50 m        : {np.mean(m['err_m'] < 0.50)*100:7.2f} %",
        f"   position < 1.00 m        : {np.mean(m['err_m'] < 1.00)*100:7.2f} %",
        f"   rotation  < 10 deg       : {np.mean(m['err_deg'] < 10)*100:7.2f} %",
        f"   rotation  < 20 deg       : {np.mean(m['err_deg'] < 20)*100:7.2f} %",
        "=" * 62,
    ]
    return "\n".join(lines)


# ============================================================================
# 6. Plots
# ============================================================================
def make_plots(hist, m, gt_xyz, pred_xyz, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # training curves
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot([e * 100 for e in hist["median_m"]], label="median")
    ax[1].plot([e * 100 for e in hist["mean_m"]], label="mean")
    ax[1].set_title("Position error (cm)"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(hist["median_deg"])
    ax[2].set_title("Median orient. error (deg)"); ax[2].set_xlabel("epoch"); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "curves.png"), dpi=120); plt.close(fig)

    # trajectory GT vs prediction
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(gt_xyz[:, 0], gt_xyz[:, 2], c="g", s=8, alpha=.5, label="ground truth")
    sc = ax.scatter(pred_xyz[:, 0], pred_xyz[:, 2], c=m["err_m"], cmap="coolwarm",
                    s=8, alpha=.8, vmin=0, vmax=np.percentile(m["err_m"], 95))
    plt.colorbar(sc, ax=ax, label="position error (m)")
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 2], "r-", lw=.4, alpha=.4)
    ax.set_title(f"PoseNet predictions — median err {m['median_m']*100:.1f} cm / "
                 f"{m['median_deg']:.1f} deg")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.legend(); ax.grid(alpha=.3)
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "trajectory.png"), dpi=130); plt.close(fig)

    # error histograms + per-frame error
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].hist(m["err_m"], bins=50); ax[0].set_title("Position error (m)")
    ax[0].axvline(m["median_m"], c="r", ls="--", label=f"median {m['median_m']:.2f} m")
    ax[0].legend()
    ax[1].hist(m["err_deg"], bins=50); ax[1].set_title("Orientation error (deg)")
    ax[1].axvline(m["median_deg"], c="r", ls="--", label=f"median {m['median_deg']:.1f} deg")
    ax[1].legend()
    ax[2].plot(m["err_m"], lw=.6)
    ax[2].set_title("Per-frame position error (test order)"); ax[2].set_xlabel("frame")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "error_hist.png"), dpi=120); plt.close(fig)


# ============================================================================
# 7. Train / main
# ============================================================================
def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gdrive-id", default=DEFAULT_GDRIVE_ID)
    p.add_argument("--zip-path", default=None, help="local/Drive path to office.zip")
    p.add_argument("--work-dir", default="/content/posenet_7scenes")
    p.add_argument("--backbone", default="mobilenet_v3_small",
                   choices=list(PoseNet.FEAT_DIMS))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=100.0,
                   help="quat loss weight (original GoogLeNet paper used 500)")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--step-size", type=int, default=30, help="StepLR step")
    p.add_argument("--gamma", type=float, default=0.5, help="StepLR gamma")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0,
                   help="cap frames per split (smoke test)")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint to load (default: best in work-dir)")
    p.add_argument("--resume", default=None, help="resume training from last.pth")
    p.add_argument("--drive-save", type=int, default=1,
                   help="copy artifacts to mounted Google Drive")
    return p.parse_args()


def build_loaders(args):
    scene_dir = prepare_dataset(args)
    cache_dir = os.path.join(args.work_dir, "cache")
    train_img, train_pose = build_cache(scene_dir, "TrainSplit.txt", cache_dir, args.limit)
    test_img, test_pose = build_cache(scene_dir, "TestSplit.txt", cache_dir, args.limit)

    train_poses = np.load(train_pose)
    pos_mean = train_poses[:, :3].mean(axis=0).astype(np.float32)
    pos_std = train_poses[:, :3].std(axis=0).astype(np.float32)
    np.save(os.path.join(args.work_dir, "checkpoints", "pos_mean.npy"), pos_mean)
    np.save(os.path.join(args.work_dir, "checkpoints", "pos_std.npy"), pos_std)

    train_ds = SevenScenesCached(train_img, train_pose, pos_mean, pos_std,
                                 train=True)
    test_ds = SevenScenesCached(test_img, test_pose, pos_mean, pos_std,
                                train=False)
    print(f"[data] train={len(train_ds)} frames, test={len(test_ds)} frames "
          f"| pos mean={pos_mean.round(3)} std={pos_std.round(3)}")

    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **common)
    test_loader = DataLoader(test_ds, batch_size=max(16, args.batch_size // 2),
                             shuffle=False, **common)
    return train_loader, test_loader, pos_mean, pos_std


def save_checkpoint(path, model, opt, sched, epoch, hist, args, pos_mean, pos_std):
    torch.save({
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict() if opt is not None else None,
        "sched_state": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
        "hist": hist,
        "backbone": args.backbone,
        "beta": args.beta,
        "pos_mean": pos_mean,
        "pos_std": pos_std,
    }, path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader, pos_mean, pos_std = build_loaders(args)
    model = PoseNet(args.backbone, dropout=args.dropout).to(device)
    model = model.to(memory_format=torch.channels_last)
    model_summary(model, args.backbone)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, args.step_size, args.gamma)
    criterion = PoseNetLoss(args.beta)
    scaler = torch.GradScaler("cuda", enabled=not args.no_amp)

    hist = {"train_loss": [], "val_loss": [], "median_m": [], "mean_m": [],
            "median_deg": [], "epoch_min": []}
    start_epoch, best_median = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["opt_state"])
        if ck.get("sched_state"):
            sched.load_state_dict(ck["sched_state"])
        hist = ck.get("hist", hist); start_epoch = ck["epoch"] + 1
        best_median = min(hist["median_m"]) if hist["median_m"] else float("inf")
        print(f"[resume] from epoch {start_epoch}")

    ck_dir = os.path.join(args.work_dir, "checkpoints")
    log_dir = os.path.join(args.work_dir, "logs")
    os.makedirs(ck_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "metrics.csv")
    if start_epoch == 0:
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,val_loss,median_m,mean_m,median_deg,lr\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for bi, (x, xyz, q) in enumerate(train_loader):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            xyz = xyz.to(device, non_blocking=True)
            q = q.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=not args.no_amp):
                p_xyz, p_q = model(x)
                loss, pos_l, quat_l = criterion(p_xyz, p_q, xyz, q)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item()
            if bi % args.log_every == 0:
                print(f"  ep {epoch:3d} [{bi:3d}/{len(train_loader)}] "
                      f"loss={loss.item():8.3f} (pos {pos_l.item():6.3f} | "
                      f"quat {quat_l.item():.4f})", flush=True)
        sched.step()

        m = evaluate(model, test_loader, device, pos_mean, pos_std,
                     amp=not args.no_amp)
        hist["train_loss"].append(run / len(train_loader))
        hist["val_loss"].append(m["val_loss"])
        hist["median_m"].append(m["median_m"]); hist["mean_m"].append(m["mean_m"])
        hist["median_deg"].append(m["median_deg"]); hist["epoch_min"].append(epoch)
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{hist['train_loss'][-1]:.5f},{m['val_loss']:.5f},"
                    f"{m['median_m']:.5f},{m['mean_m']:.5f},{m['median_deg']:.3f},"
                    f"{sched.get_last_lr()[0]:.2e}\n")
        print(f"== Epoch {epoch:3d} ({time.time()-t0:5.1f}s) "
              f"train {hist['train_loss'][-1]:8.3f} | val {m['val_loss']:8.3f} | "
              f"median pos {m['median_m']*100:6.1f} cm | median rot "
              f"{m['median_deg']:5.1f} deg", flush=True)

        save_checkpoint(os.path.join(ck_dir, "posenet_last.pth"), model, opt,
                        sched, epoch, hist, args, pos_mean, pos_std)
        if m["median_m"] < best_median:
            best_median = m["median_m"]
            save_checkpoint(os.path.join(ck_dir, "posenet_best.pth"), model, None,
                            None, epoch, hist, args, pos_mean, pos_std)
            print(f"   ^ new best (median {best_median*100:.1f} cm) saved")

    print("\n[done] training finished. Loading best checkpoint for final eval ...")
    return final_eval(args, ck_dir, log_dir, device,
                      os.path.join(ck_dir, "posenet_best.pth"))


def final_eval(args, ck_dir, log_dir, device, ckpt_path):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PoseNet(ck.get("backbone", args.backbone)).to(device)
    model.load_state_dict(ck["model_state"])
    pos_mean, pos_std = ck["pos_mean"], ck["pos_std"]

    cache_dir = os.path.join(args.work_dir, "cache")
    test_img = os.path.join(cache_dir, "TestSplit.txt_images.npy")
    test_pose = os.path.join(cache_dir, "TestSplit.txt_poses.npy")
    test_ds = SevenScenesCached(test_img, test_pose, pos_mean, pos_std, train=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    m = evaluate(model, test_loader, device, pos_mean, pos_std)
    report = format_report(m)
    print("\n" + report)
    with open(os.path.join(log_dir, "eval_report.txt"), "w") as f:
        f.write(report + f"\ncheckpoint: {ckpt_path}\nepoch: {ck['epoch']}\n")

    # per-frame results, same format as the repo's test script
    np.savetxt(os.path.join(log_dir, "results.txt"),
               np.stack([m["err_m"], m["err_deg"]], axis=1), delimiter=" ")

    # GT vs predicted positions for the trajectory plot
    gt_xyz, pred_xyz = [], []
    with torch.no_grad():
        for x, xyz, q in test_loader:
            p_xyz, _ = model(x.to(device, non_blocking=True))
            pred_xyz.append((p_xyz.float().cpu().numpy() * pos_std + pos_mean))
            gt_xyz.append(xyz.numpy() * pos_std + pos_mean)
    make_plots(ck.get("hist", {"train_loss": [], "val_loss": [], "median_m": [],
                               "mean_m": [], "median_deg": []}),
               m, np.concatenate(gt_xyz), np.concatenate(pred_xyz), log_dir)
    print(f"[eval] report, results.txt and plots written to {log_dir}")
    return m


def copy_to_drive(args):
    if not args.drive_save:
        return
    drive_root = "/content/drive/MyDrive"
    if not os.path.isdir(drive_root):
        return
    dst = os.path.join(drive_root, "posenet_7scenes_office")
    os.makedirs(dst, exist_ok=True)
    for sub in ("checkpoints", "logs"):
        src = os.path.join(args.work_dir, sub)
        if os.path.isdir(src):
            for f in os.listdir(src):
                if os.path.isfile(os.path.join(src, f)):
                    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print(f"[drive] artifacts copied to {dst}")


def main():
    args = get_args()
    print(f"[env] python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[env] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[env] WARNING: no GPU detected — training will be very slow. "
              "In Colab: Runtime > Change runtime type > T4 GPU.")

    os.makedirs(os.path.join(args.work_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "logs"), exist_ok=True)

    if args.eval_only:
        ckpt = args.checkpoint or os.path.join(args.work_dir, "checkpoints",
                                               "posenet_best.pth")
        final_eval(args, os.path.join(args.work_dir, "checkpoints"),
                   os.path.join(args.work_dir, "logs"),
                   torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                   ckpt)
        export_for_inference(args)
        copy_to_drive(args)
        return

    train(args)
    export_for_inference(args)
    copy_to_drive(args)


if __name__ == "__main__":
    main()
