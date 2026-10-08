# Litewing — Autonomous UAV Indoor Localization System

**Real-time visual indoor localization using a XIAO ESP32-S3 Sense camera, PoseNet, Visual Odometry, and MiDaS depth estimation.**

The system fuses three engines to locate a UAV (or any camera) indoors without GPS:

| Engine | Purpose | Works In |
|--------|---------|----------|
| **PoseNet** | Absolute position (trained room) | Known rooms only |
| **Visual Odometry** | Relative motion tracking | Any room |
| **MiDaS Depth** | Wall/obstacle detection | Any room (zero-shot) |

A fusion controller automatically switches between PoseNet and Visual Odometry based on confidence — when PoseNet recognizes the room, it provides absolute coordinates; when it doesn't, Visual Odometry takes over for relative tracking.

---

## 📐 Architecture

```
┌─────────────────────┐        WiFi (2.4 GHz)       ┌──────────────────┐
│  XIAO ESP32-S3      │ ◄──────────────────────────► │  PC / Laptop     │
│  Sense Camera       │   HTTP JPEG stream           │                  │
│                     │   /capture endpoint           │  ┌────────────┐ │
│  • OV2640 sensor    │                              │  │ PoseNet    │ │
│  • WiFi STA or AP   │                              │  │ (PyTorch)  │ │
│  • HTTP server      │                              │  └─────┬──────┘ │
└─────────────────────┘                              │        │        │
                                                     │  ┌─────▼──────┐ │
                                                     │  │  Fusion    │ │
                                                     │  │  Engine    │ │
                                                     │  └─────┬──────┘ │
                                                     │        │        │
                                                     │  ┌─────▼──────┐ │
                                                     │  │ Visual     │ │
                                                     │  │ Odometry   │ │
                                                     │  └─────┬──────┘ │
                                                     │        │        │
                                                     │  ┌─────▼──────┐ │
                                                     │  │ MiDaS      │ │
                                                     │  │ Depth      │ │
                                                     │  └────────────┘ │
                                                     │                  │
                                                     │  Real-time GUI   │
                                                     │  with 2D map     │
                                                     └──────────────────┘
```

---

## 🔧 Hardware Required

| Component | Details |
|-----------|---------|
| **XIAO ESP32-S3 Sense** | With OV2640 camera expansion board |
| **Antenna** | The tiny WiFi antenna must be connected to the U.FL connector |
| **USB-C cable** | For flashing firmware |
| **PC / Laptop** | Windows, macOS, or Linux with Python 3.8+ |
| **WiFi hotspot** | Phone hotspot or PC hotspot (**must be 2.4 GHz!**) |

---

## Quick Start

### Step 1: Flash the XIAO Camera

1. Install [Arduino IDE](https://www.arduino.cc/en/software) (2.x recommended)
2. Add ESP32 board support:
   - **File → Preferences → Additional Board Manager URLs**, add:
     ```
     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
     ```
   - **Tools → Board → Boards Manager** → search "esp32" → install **"esp32 by Espressif Systems"** (v2.0.14+)
3. Select board settings:
   | Setting | Value |
   |---------|-------|
   | Board | XIAO_ESP32S3 |
   | PSRAM | **OPI PSRAM** ⚠️ Camera WILL fail without this |
   | USB CDC On Boot | **Enabled** (for Serial Monitor over USB) |
4. Edit WiFi credentials in [`firmware/xiao_camera_server/xiao_camera_server.ino`](firmware/xiao_camera_server/xiao_camera_server.ino):
   ```cpp
   WiFiCred wifiList[] = {
     {"YOUR_HOTSPOT_NAME", "YOUR_PASSWORD"},
   };
   ```
5. Upload the sketch
6. Open **Serial Monitor at 115200 baud** — it prints the IP address

> ⚠️ **The XIAO only supports 2.4 GHz WiFi!**
> - **Windows hotspot**: Settings → Mobile hotspot → Edit → Network band → **2.4 GHz**
> - **Android hotspot**: Settings → Hotspot → Band → **2.4 GHz**
> - **iPhone hotspot**: Settings → Personal Hotspot → Maximize Compatibility → **ON**

### Step 2: Install Python Dependencies

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Litewing_Autonomous_UAVs.git
cd Litewing_Autonomous_UAVs

# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run!

```bash
# Replace with the IP shown in Serial Monitor
export XIAO_URL="http://192.168.137.222/capture"

# Option A: Full Fusion System (PoseNet + VO + Depth)
python fusion_live.py --url $XIAO_URL

# Option B: Visual Odometry only (works anywhere, no training needed)
python visual_odometry.py --url $XIAO_URL

# Option C: Depth estimation only
python midas_live.py --url $XIAO_URL

# Option D: PoseNet prediction only
python posenet_predict.py --weights posenet_best.pth --url $XIAO_URL
```

On **Windows PowerShell**, use `$env:XIAO_URL` instead:
```powershell
$env:XIAO_URL = "http://192.168.137.222/capture"
python fusion_live.py --url $env:XIAO_URL
```

---

## 📁 Project Structure

```
Litewing_Autonomous_UAVs/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── SETUP_GUIDE.md             # Detailed setup walkthrough
├── .gitignore                 # Git ignore rules
│
├── fusion_live.py             # 🎯 Main system — fuses all engines
├── visual_odometry.py         # Visual odometry (works in any room)
├── midas_live.py              # MiDaS depth estimation
├── posenet_predict.py         # PoseNet inference (known rooms)
│
├── posenet_best.pth           # Pre-trained PoseNet weights (7-Scenes office)
│                              #   ⚠️ Large file — use Git LFS or download separately
│
├── firmware/
│   └── xiao_camera_server/
│       └── xiao_camera_server.ino   # ESP32 camera firmware
│
└── training/
    └── posenet_train_colab.py       # PoseNet training script (Google Colab)
```

---

## 🧩 Component Details

### Fusion Live (`fusion_live.py`)
The main system that combines all engines into a real-time GUI:
- **Camera feed** with position/mode overlay
- **Depth strip** showing MiDaS wall detection
- **2D room map** with trajectory, wall points, and mode indicator

```bash
python fusion_live.py --url http://<XIAO_IP>/capture [OPTIONS]

Options:
  --url URL              XIAO capture endpoint
  --camera N             Use webcam index N instead
  --weights PATH         PoseNet checkpoint (default: posenet_best.pth)
  --conf-thresh FLOAT    PoseNet confidence threshold (default: 0.15)
  --vo-scale FLOAT       VO translation scale (default: 0.05)
  --no-depth             Disable MiDaS (faster)
  --depth-model MODEL    DPT_Large | DPT_Hybrid | MiDaS_small
  --depth-skip N         Run depth every N frames (default: 3)
```

### Visual Odometry (`visual_odometry.py`)
Standalone VO — tracks camera motion using optical flow. No training required, works in any room.

```bash
python visual_odometry.py --url http://<XIAO_IP>/capture --scale 1.0
```

### MiDaS Depth (`midas_live.py`)
Zero-shot monocular depth estimation. Detects walls and obstacles without training.

```bash
python midas_live.py --url http://<XIAO_IP>/capture
python midas_live.py --image photo.jpg --save-out depth.png
```

### PoseNet Predict (`posenet_predict.py`)
Absolute position estimation in a trained room (7-Scenes office scene).

```bash
python posenet_predict.py --weights posenet_best.pth --url http://<XIAO_IP>/capture
python posenet_predict.py --weights posenet_best.pth --image frame.png
python posenet_predict.py --weights posenet_best.pth --folder frames/
```

---

## 🏋️ Training Your Own PoseNet

To train PoseNet on your own room or a different 7-Scenes scene:

1. Open [Google Colab](https://colab.research.google.com/) (free GPU)
2. Upload `training/posenet_train_colab.py`
3. Run:
   ```bash
   pip install -q gdown
   python posenet_train_colab.py --epochs 100
   ```
4. Download the resulting `posenet_best.pth` and place it in the project root

See [`training/posenet_train_colab.py`](training/posenet_train_colab.py) for full documentation.

---

## 🛠 Troubleshooting

### XIAO won't connect to WiFi
| Symptom | Fix |
|---------|-----|
| `NO_SSID_AVAIL` in Serial Monitor | Your hotspot is on 5 GHz — switch to **2.4 GHz** |
| `CONNECT_FAILED` | Wrong password — check special characters |
| No networks found in scan | Antenna not connected — plug in the tiny cable |
| Falls back to AP mode | All SSIDs failed — connect PC to `XIAO-CAM` / `12345678` |

### PC can't connect to XIAO-CAM AP
- "Forget" the network and reconnect
- Try from your phone first to verify the AP works
- Update your WiFi adapter driver

### Camera init failed
- Set **Tools → PSRAM → OPI PSRAM** in Arduino IDE and re-upload
- Make sure the camera expansion board is firmly seated

### Python errors
| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: timm` | `pip install timm` (needed by MiDaS) |
| `ModuleNotFoundError: torchvision` | `pip install torchvision` |
| MiDaS download hangs | Download weights manually (see `midas_live.py` header) |
| CUDA out of memory | Use `--depth-model MiDaS_small` or `--no-depth` |

### Performance tips
- Use `--no-depth` flag for faster fusion without depth
- Use `--depth-skip 5` to run depth less frequently
- Use `MiDaS_small` instead of `DPT_Large` on CPU
- Close other GPU-heavy apps

---

## 📜 License

This project is for academic and research purposes.

---

