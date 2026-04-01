# ICLR2026 — Functional Map (Anonymous)

This repository contains the code and models for the ICLR 2026 submission:

> **Functional Embeddings Enable Aggregation of Multi-Area SEEG Recordings Over Subjects and Sessions**

_This repo is anonymized for double-blind review._

---

## Quick Start

**Tested on:** Python **3.11.8**, PyTorch **2.5.1 (cu121)**  
> **Note:** Please install the PyTorch build that matches **your** hardware. The command below uses CUDA 12.1 as an example.

### 1) Create & activate a fresh environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -V
```
### 2) Install PyTorch:
Use your exact build (works on CUDA 12.1 systems):
```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121
```
### 3) Install repo dependencies:
```powershell
pip install --upgrade pip
pip install -r requirements.txt          # for full requirement file
# (optional) pip install -r requirements-minimal.txt   # for minimal version
# (optional) pip install -r requirements-viz.txt       # brain plots/3D viz
# (optional) pip install -r requirements-notebooks.txt # Jupyter tooling
```

