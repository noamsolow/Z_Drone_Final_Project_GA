# Monocular Drone 3D Localization (Z-Drone)

This project estimates the Z-distance of a drone from a single image using Monocular Depth Estimation (MiDaS / ZoeDepth / Depth Anything) and YOLO.

## ⚠️ Prerequisites
* **Python 3.9** is required for compatibility with ZoeDepth and MiDaS BEiT models.

## 🛠️ Installation & Setup

### 1. First-Time Setup
If you have just cloned this repository, run these commands in your terminal to set up the correct environment:

```bash
# 1. Create a virtual environment using Python 3.9 (Windows)
# Note: You must have Python 3.9 installed on your machine
py -3.9 -m venv .venv

# 2. Activate the environment
.venv\Scripts\activate

# 3. Install dependencies
# This installs torch, opencv, and the specific version of timm (0.6.12) required
pip install -r requirements.txt

# Run this after installing new packages to save them to the list
pip freeze > requirements.txt