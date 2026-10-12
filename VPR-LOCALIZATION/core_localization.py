import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE_SHORT = 256
CROP = 224

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


def resize_short_side(img, short=RESIZE_SHORT):
    h, w = img.shape[:2]
    scale = short / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if cv2 is not None:
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))


def preprocess(img):
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


class CoreLocalizer:
    """
    Core Inference Module.
    Stripped of all GUI, matplotlib, and debugging tools.
    """
    def __init__(self, weights=None, onnx_path=None):
        if bool(weights) == bool(onnx_path):
            raise ValueError("Provide exactly one of weights or onnx_path")
        if onnx_path:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"])
            self.mode = "onnx"
        else:
            ck = torch.load(weights, map_location="cpu", weights_only=False)
            self.model = PoseNet(ck.get("backbone", "mobilenet_v3_small"))
            self.model.load_state_dict(ck["model_state"])
            self.model.eval()
            self.pos_mean = np.asarray(ck["pos_mean"], dtype=np.float32)
            self.pos_std = np.asarray(ck["pos_std"], dtype=np.float32)
            self.mode = "torch"

    def estimate_pose(self, frame_rgb):
        """
        Takes an RGB uint8 image (HxWx3).
        Returns a dictionary representing the pose result.
        """
        t = preprocess(frame_rgb)
        if self.mode == "onnx":
            pos, quat = self.sess.run(None, {"image": t["raw"].numpy()})
            pos, quat = pos[0], quat[0]
        else:
            with torch.no_grad():
                p, q = self.model(t["norm"])
            pos = p.numpy()[0] * self.pos_std + self.pos_mean
            quat = q.numpy()[0]

        return {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "qx": float(quat[0]),
            "qy": float(quat[1]),
            "qz": float(quat[2]),
            "qw": float(quat[3]),
            "confidence": 1.0,
            "timestamp": time.time()
        }
