# Traffic Monitor on Jetson Nano – Step-by-Step Guide

> **Target Board:** NVIDIA Jetson Nano 4 GB Developer Kit running JetPack 4.6 (CUDA 10.2 / TensorRT 8.2)
>
> **Image used:** [Qengineering Jetson-Nano-image](https://github.com/Qengineering/Jetson-Nano-image) – a pre-configured SD image that already contains CUDA, cuDNN, TensorRT, OpenCV, TensorFlow 2.4, PyTorch 1.8 and more.
>
> This guide shows how to clone, configure and run the multi-process **Traffic Monitor** pipeline on that image.

---

## 0. Bill of materials

| Item | Notes |
|------|-------|
| Jetson Nano 4 GB dev-kit | Either the A01 or B01 carrier board. |
| 5 V ⎓ 4 A PSU | Barrel-jack recommended (micro-USB often throttles). |
| 64 GB (min 32 GB) micro-SD | Flash the `JetsonNano.img.xz` from Qengineering. |
| USB-Cam **or** RTSP stream | For `--mode live`. Video file works for `--mode offline`. |
| (Optional) USB-SSD | Faster swap & model conversion. |

---

## 1. Flash & first-boot

1. Download `JetsonNano.img.xz` from the Qengineering repo and flash it with balenaEtcher _(keep it compressed – issue #17)._  
2. Insert the SD, connect HDMI/keyboard/ethernet, power on.  
3. Login: **jetson / jetson**.  
4. Grow the root filesystem if you used >32 GB SD:

   ```bash
   sudo apt update && sudo apt install -y gparted
   sudo gparted   # GUI → resize /dev/mmcblk0p1 to full card
   ```

---

## 2. Put the board in max-performance mode

```bash
sudo nvpmodel -m 0   # MAXN (4×A57 @1.5 GHz + GPU @921 MHz)
sudo jetson_clocks   # lock clocks
```

Optional GPU OC (+8 % FPS):

```bash
echo 1000000000 | sudo tee /sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq
```

---

## 3. System packages

```bash
sudo apt update
sudo apt install -y python3-venv git libopenblas-base libjpeg-dev zlib1g-dev
```

JetPack already provides CUDA 10.2, cuDNN 8.2, TensorRT 8.2 & OpenCV 4.5.

---

## 4. Clone the project & create a venv

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/<your-fork>/trafficmonitor.git
cd trafficmonitor
python3 -m venv tm-env
source tm-env/bin/activate
pip install --upgrade pip wheel setuptools
```

---

## 5. Install Python dependencies (Nano-friendly pins)

Create `requirements_jetson.txt` with:

```text
loguru>=0.7.0
click>=8.0
numpy==1.22.4                # last aarch64 wheel for Py3.8
# NVIDIA-built PyTorch & TorchVision wheels
torch==1.13.0+nv22.12         --extra-index-url https://download.pytorch.org/whl/jetson
torchvision==0.14.0+nv22.12   --extra-index-url https://download.pytorch.org/whl/jetson
ultralytics==8.1.38           # latest compatible with torch-1.13
boxmot==13.0.0
opencv-python==4.8.1.78
shapely==1.8.5.post1          # v2 lacks arm64 wheels
fast-plate-ocr==1.0.1
pyyaml
```

Install:

```bash
pip install -r requirements_jetson.txt
```

> **Why these versions?** Newer packages drop aarch64 wheels or require CUDA 11+. The pins above compile cleanly on JetPack 4.6.

---

## 6. Build or download TensorRT engines

The default `settings.yaml` expects:

```
 data/models/vehicle/8n/best.engine  # YOLOv8 vehicle detector
 data/models/plate/5nu/best.engine   # YOLOv5n-u LP detector
```

### Option A – download ready-made engines

Save them under the same paths if you have pre-converted engines.

### Option B – convert yourself (≈20 min each)

```bash
# Vehicle detector – YOLOv8n → TensorRT FP16
source tm-env/bin/activate
cd trafficmonitor

yolo export model=yolov8n.pt format=engine device=0 half=True \
     imgsz=640 batch=1 dynamic=False simplify=True \
     engine=data/models/vehicle/8n/best.engine

# License-plate detector – single-class YOLOv5
ultralytics export model=lp-detector.pt format=engine device=0 \
     engine=data/models/plate/5nu/best.engine
```

> **Tip:** keep batch = 1 and `dynamic=False` to fit Nano GPU RAM.

---

## 7. Prepare input video or camera

```bash
mkdir -p data/videos/input
# copy your clip
cp ~/Downloads/traffic_clip.mp4 data/videos/input/
```

For live mode, plug a USB-cam or use an RTSP URL.

---

## 8. One-shot run examples

### Offline (process a full MP4 and save annotated MP4)

```bash
traffic-monitor \
  --mode offline \
  --source data/videos/input/traffic_clip.mp4 \
  --count-line 0,0.4,1,0.4  # disable counting line → “none”
```

### Live USB-camera (index 0) with no counting lines

```bash
traffic-monitor --mode live --source 0 --count-line none --verbose
```

Outputs land in `data/videos/output/<TIMESTAMP>/`:

* `processed.mp4` – annotated video
* `summary.json`  – per-run metrics
* `traffic_monitor.log` – consolidated Loguru log

---

## 9. Performance tuning

| Knob | Effect |
|------|--------|
| `frame_grabber.resize_resolution` | Lower to 1280×720 to raise FPS. |
| `frame_grabber.process_every_n_frame` | Skip frames for faster-than-RT processing. |
| `vehicle_detector.conf_threshold` | 0.5–0.4 trades accuracy for speed. |
| Disable GUI | `visualizer.enable_gui: false` (default on Nano). |

Monitor utilisation:

```bash
sudo tegrastats   # GPU / RAM / power per 1 s
```

---

## 10. Service at boot (optional)

Create `/etc/systemd/system/traffic-monitor.service`:

```ini
[Unit]
Description=Traffic Monitor
After=network-online.target nvargus-daemon.service

[Service]
WorkingDirectory=/home/jetson/projects/trafficmonitor
ExecStart=/home/jetson/projects/tm-env/bin/traffic-monitor --mode live --source 0 --count-line none
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable traffic-monitor
```

---

## 11. Troubleshooting FAQ

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: tensorrt` | Don’t `pip install tensorrt`. The JetPack system library is already available. |
| `Illegal instruction (core dumped)` when importing torch | You installed upstream wheels. Re-install NVIDIA wheels with the `+nv22.12` tag. |
| `TRT] engine deserialization failed` | Engine was built for a different GPU arch. Re-export on **this** Nano. |
| Package X has no arm64 wheel | Pin to an older version (see § 5) or compile from source. |

---

## 12. References

* Qengineering Jetson-Nano image – <https://github.com/Qengineering/Jetson-Nano-image>
* NVIDIA Jetson Benchmarks – <https://developer.nvidia.com/embedded/jetson-benchmarks>
* Ultralytics export docs – <https://docs.ultralytics.com/models/export>

---

**Enjoy real-time traffic analytics on a $99 board!** 