# InspectIQ

Unsupervised industrial anomaly detection (PaDiM + frozen ResNet18) with
cross-category structural robustness profiling. Streamlit interface.

## Folder structure
```
InspectIQ/
├── data/mvtec/{bottle,cable,tile}/     <- gitignored, never pushed
├── models/                             <- trained PaDiM stats (.pkl)
├── src/
│   ├── padim_core.py
│   ├── train.py
│   └── evaluate.py
├── app.py
├── requirements.txt
└── .gitignore
```

## Setup (PowerShell)

```powershell
cd InspectIQ
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Step 1 — Copy only the 3 categories you need
```powershell
robocopy "C:\path\to\full_mvtec\bottle" ".\data\mvtec\bottle" /E
robocopy "C:\path\to\full_mvtec\cable"  ".\data\mvtec\cable"  /E
robocopy "C:\path\to\full_mvtec\tile"   ".\data\mvtec\tile"   /E
```

## Step 2 — Train PaDiM stats for all 3 categories
```powershell
python src\train.py --categories bottle cable tile
```

## Step 3 — Evaluate + cross-category robustness matrix
```powershell
python src\evaluate.py --categories bottle cable tile
```

## Step 4 — Launch the app
```powershell
streamlit run app.py
```

## Step 5 — Push to GitHub (data/ never included)
```powershell
git init
git add .
git commit -m "InspectIQ: 3-category PaDiM anomaly detection"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Deploy
Streamlit Community Cloud is the fastest free option for a Streamlit app —
point it at your GitHub repo and `app.py`. Note: Streamlit Cloud does not
persist `models/*.pkl` between sessions unless committed to the repo, so
make sure the 3 trained `.pkl` files (bottle/cable/tile) ARE committed —
they should be small (a few MB each), unlike the raw dataset.

## Deployment pitfalls already fixed for you
- **requirements.txt pins CPU-only torch/torchvision wheels.** Plain `torch`
  pulls the CUDA build (~800MB+), which routinely times out or exceeds the
  size limit on free-tier hosts (Streamlit Cloud, Render free, etc.). You
  don't need CUDA for inference anyway.
- **runtime.txt pins Python 3.11** so the deploy environment matches what
  this was built/tested against — avoids silent version-mismatch errors.
- **opencv-python-headless** (not plain `opencv-python`) — the non-headless
  build needs system GUI libraries that aren't present on cloud hosts and
  is a very common `ImportError: libGL.so.1` cause on first deploy.
- **`.pkl` model files, not the raw dataset, get committed** — keeps the
  repo small enough that GitHub/host size limits are a non-issue.

If deploy still errors, paste me the exact build log — don't just say "it
failed." The log tells us in one look whether it's a missing package, a
size limit, or a Python version mismatch.
