# 📋 Litewing — Detailed Setup Guide

This guide walks you through setting up the entire Litewing indoor localization system from scratch on a new PC.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Setting Up the XIAO ESP32-S3 Camera](#2-setting-up-the-xiao-esp32-s3-camera)
3. [Setting Up the Python Environment](#3-setting-up-the-python-environment)
4. [Connecting XIAO to Your PC](#4-connecting-xiao-to-your-pc)
5. [Running the System](#5-running-the-system)
6. [Training PoseNet for Your Own Room](#6-training-posenet-for-your-own-room)
7. [Common Issues & Fixes](#7-common-issues--fixes)

---

## 1. Prerequisites

### Software
- **Python 3.8+** — Download from [python.org](https://www.python.org/downloads/)
  - ⚠️ During install, check **"Add Python to PATH"**
- **Arduino IDE 2.x** — Download from [arduino.cc](https://www.arduino.cc/en/software)
- **Git** — Download from [git-scm.com](https://git-scm.com/downloads)

### Hardware
- XIAO ESP32-S3 Sense (with OV2640 camera board attached)
- USB-C cable (data cable, not charge-only)
- WiFi antenna connected to the XIAO's U.FL connector

### Optional (for GPU acceleration)
- NVIDIA GPU with CUDA support
- Install CUDA toolkit from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- Then install PyTorch with CUDA:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

---

## 2. Setting Up the XIAO ESP32-S3 Camera

### 2.1 Install Arduino IDE & Board Support

1. Open Arduino IDE
2. Go to **File → Preferences**
3. In **"Additional Board Manager URLs"**, paste:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Go to **Tools → Board → Boards Manager**
5. Search **"esp32"** and install **"esp32 by Espressif Systems"** version 2.0.14 or newer

### 2.2 Configure Board Settings

Connect the XIAO via USB-C, then set:

| Setting | Value | Why |
|---------|-------|-----|
| **Tools → Board** | XIAO_ESP32S3 | Matches our hardware |
| **Tools → PSRAM** | OPI PSRAM | Camera needs PSRAM for frame buffers |
| **Tools → USB CDC On Boot** | Enabled | Allows Serial Monitor over USB |
| **Tools → Port** | (select the COM port that appeared) | Your XIAO's USB port |

### 2.3 Configure WiFi Credentials

Open `firmware/xiao_camera_server/xiao_camera_server.ino` and edit:

```cpp
WiFiCred wifiList[] = {
  {"YOUR_HOTSPOT_NAME", "YOUR_PASSWORD"},     // Primary network
  // {"BackupNetwork", "backup_password"},     // Optional backup
};
```

**Tips:**
- The SSID and password must match **exactly** (case-sensitive, including spaces)
- You can add multiple networks — the XIAO tries each in order
- If all fail, it creates its own WiFi network: `XIAO-CAM` / password: `12345678`

### 2.4 Upload & Verify

1. Click **Upload** (→ button)
2. Wait for "Done uploading"
3. Open **Tools → Serial Monitor** at **115200 baud**
4. Press the **Reset** button on the XIAO
5. You should see:
   ```
   ========================================
     XIAO ESP32-S3 Sense Camera Server
   ========================================

   PSRAM: XXXXXXX bytes free
   Camera: OK
   ===== SCANNING NEARBY WiFi NETWORKS =====
   ...
   [WiFi] CONNECTED! IP: 192.168.137.XXX
   ```
6. Open that IP in a browser: `http://192.168.137.XXX/capture` — you should see a JPEG image

---

## 3. Setting Up the Python Environment

### 3.1 Clone & Create Virtual Environment

```bash
git clone https://github.com/YOUR_USERNAME/Litewing_Autonomous_UAVs.git
cd Litewing_Autonomous_UAVs

# Create isolated Python environment
python -m venv venv

# Activate it
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# macOS/Linux:
source venv/bin/activate
```

### 3.2 Install Dependencies

```bash
pip install -r requirements.txt
```

**If you have an NVIDIA GPU and want CUDA acceleration:**
```bash
# Instead of the above, do:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3.3 Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

### 3.4 Download PoseNet Weights

If `posenet_best.pth` is not included in the repo (too large for GitHub):
- Download it from the releases page
- Or train your own (see Section 6)
- Place it in the project root directory

---

## 4. Connecting XIAO to Your PC

### Method A: PC Mobile Hotspot (Recommended)

1. **Windows**: Settings → Network & internet → Mobile hotspot
2. **Turn ON** the hotspot
3. **Click Edit** → Set **Network band** to **2.4 GHz**
4. Note the hotspot name and password
5. Put those credentials in the Arduino sketch
6. Flash the XIAO → it connects to your PC's hotspot
7. Check Serial Monitor for the assigned IP address

### Method B: Phone Hotspot

1. Enable your phone's hotspot
2. **Force 2.4 GHz band**:
   - Android: Settings → Hotspot → Band → 2.4 GHz
   - iPhone: Settings → Personal Hotspot → Maximize Compatibility → ON
3. Connect both your PC and XIAO to the phone hotspot
4. Check Serial Monitor for the XIAO's IP

### Method C: AP Mode (Fallback)

If the XIAO can't connect to any configured network, it creates its own:
- **Network name**: `XIAO-CAM`
- **Password**: `12345678`
- Connect your PC to this network
- Access the camera at `http://192.168.4.1/capture`

> ⚠️ In AP mode, your PC loses internet access while connected to XIAO-CAM.

---

## 5. Running the System

### Quick Test — Verify Camera Stream

Open a browser and go to `http://<XIAO_IP>/capture` — you should see a JPEG snapshot.

### Visual Odometry (No Training Needed)

Best for getting started — works in any room:

```bash
python visual_odometry.py --url http://<XIAO_IP>/capture
```

Move the camera slowly. You'll see:
- Left: camera feed with optical flow vectors
- Right: top-down trajectory map

### Full Fusion System

```bash
python fusion_live.py --url http://<XIAO_IP>/capture
```

You'll see:
- Camera feed with position/mode overlay
- Depth strip (bright = close)
- 2D room map with trajectory

**Performance options:**
```bash
# Faster — no depth estimation
python fusion_live.py --url http://<XIAO_IP>/capture --no-depth

# Less frequent depth (every 5 frames instead of 3)
python fusion_live.py --url http://<XIAO_IP>/capture --depth-skip 5

# Lightweight depth model
python fusion_live.py --url http://<XIAO_IP>/capture --depth-model MiDaS_small
```

### Using a Webcam Instead

You can use your PC's webcam instead of the XIAO:

```bash
python visual_odometry.py --camera 0
python fusion_live.py --camera 0
python midas_live.py --camera 0
```

---

## 6. Training PoseNet for Your Own Room

The included `posenet_best.pth` is trained on the Microsoft 7-Scenes "office" scene. To train for **your own room**:

### Using Google Colab (Free GPU)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `training/posenet_train_colab.py`
3. Set runtime to GPU: Runtime → Change runtime type → GPU
4. Run in a code cell:
   ```python
   !pip install -q gdown
   !python posenet_train_colab.py --epochs 100
   ```
5. Download the generated `posenet_best.pth`
6. Place it in the Litewing project root

### Collecting Your Own Training Data

To train on your own room, you need:
1. A set of images from different positions/angles in the room
2. Corresponding camera poses (x, y, z position + orientation)
3. Modify the data loading in `posenet_train_colab.py` for your format

---

## 7. Common Issues & Fixes

### "ModuleNotFoundError: No module named 'timm'"
```bash
pip install timm
```

### MiDaS weights won't download
Download manually:
- MiDaS_small: https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt
- DPT_Large: https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt

Place in: `C:\Users\<you>\.cache\torch\hub\checkpoints\` (Windows) or `~/.cache/torch/hub/checkpoints/` (Linux/Mac)

### "FATAL: PSRAM not found"
In Arduino IDE: **Tools → PSRAM → OPI PSRAM** → re-upload the sketch.

### "FATAL: camera init failed"
The camera expansion board isn't seated properly. Remove it, clean the contacts, and re-attach firmly.

### Serial Monitor shows garbage characters
Set baud rate to **115200** (bottom-right dropdown in Serial Monitor).

### WiFi scan shows 0 networks
The antenna cable is disconnected. It's a tiny cable on the XIAO board — make sure it clicks into the U.FL connector.

### "Can't connect to this network" on PC
- "Forget" the network in Windows WiFi settings
- Restart the XIAO (press reset button)
- Try connecting again
- If connecting to XIAO-CAM AP: make sure no other saved WiFi is auto-connecting

### Fusion runs slowly
- Add `--no-depth` to disable MiDaS
- Use `--depth-model MiDaS_small` instead of DPT_Large
- Increase `--depth-skip` to 5 or 10
- Close browser tabs and other GPU apps

---

## ✅ Verification Checklist

Use this to confirm everything works:

- [ ] Arduino IDE installed with ESP32 board support
- [ ] XIAO flashed successfully (Serial Monitor shows camera OK)
- [ ] WiFi connected (Serial Monitor shows IP address)
- [ ] Browser shows image at `http://<IP>/capture`
- [ ] Python venv created and dependencies installed
- [ ] `python visual_odometry.py --url http://<IP>/capture` opens GUI
- [ ] `python fusion_live.py --url http://<IP>/capture` opens fusion GUI

---

*If you're stuck, open an issue on GitHub with your Serial Monitor output and Python error messages.*
