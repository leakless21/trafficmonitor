# Installation

**Prerequisites:**

- Python 3.10-3.11
- OpenCV
- PyTorch (for YOLO and Re-ID models)
- `pixi` (recommended for dependency management)

**1. Clone the repository:**

```bash
git clone https://github.com/your-username/trafficmonitor.git
cd trafficmonitor
```

**2. Install dependencies using Pixi (Recommended):**

If you have Pixi installed, you can set up the environment and install dependencies with:

```bash
pixi install
pixi shell
```

**3. Manual Installation (if not using Pixi):**

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
pip install -e .
```

**4. Download Models:**

Pre-trained models for YOLO, Re-ID, and OCR are required. You can download them using the provided script:

```bash
python tools/download_model.py
```

Ensure that the `data/models/` directory contains the necessary `.pt` and `.onnx` files as specified in `src/traffic_monitor/config/settings.yaml`.
