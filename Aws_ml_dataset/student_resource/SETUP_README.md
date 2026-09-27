# Business Entity Resolution — Setup & Run Guide

This guide explains how to set up the environment and run the full ML pipeline end-to-end.

---

## Prerequisites

- **Linux, macOS, or Windows** (all supported)
- Python 3.11 (required — other versions may have compatibility issues)
- At least **32 GB RAM** recommended
- At least **20 GB free disk space** for checkpoints and output files

Check your Python version:
```bash
python3.11 --version   # Linux / macOS
python --version        # Windows (if Python 3.11 is your default)
```

Install Python 3.11 if needed:

**Fedora Linux:**
```bash
sudo dnf install python3.11
```

**Ubuntu/Debian:**
```bash
sudo apt install python3.11 python3.11-venv
```

**Windows:**
Download from https://www.python.org/downloads/release/python-3119/ — install with "Add to PATH" checked.

**macOS:**
```bash
brew install python@3.11
```

---

## Step 1: Clone / Download the repository

Place the project under any directory. The notebook uses absolute paths internally — you will need to update the `BASE` path (see Step 4).

---

## Step 2: Place the dataset

The dataset is **not included** in the repository (too large for Git).

Download the challenge dataset and place it like this:

```
student_resource/
└── dataset/
    ├── train/
    │   ├── train_source1.tsv
    │   ├── train_source2.tsv
    │   ├── train_source3.tsv
    │   └── train_ground_truth.tsv
    └── test/
        ├── test_source1.tsv
        ├── test_source2.tsv
        └── test_source3.tsv
```

All files are tab-separated (`.tsv`). Do **not** open them in Excel or they may be corrupted.

---

## Step 3: Create the Python virtual environment

**Linux / macOS:**
```bash
python3.11 -m venv venv --prompt "entity-resolution"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name "entity-resolution" --display-name "Entity Resolution (Python 3.11)"
```

**Windows (Command Prompt):**
```cmd
py -3.11 -m venv venv --prompt "entity-resolution"
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name "entity-resolution" --display-name "Entity Resolution (Python 3.11)"
```

**Windows (PowerShell):**
```powershell
py -3.11 -m venv venv --prompt "entity-resolution"
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name "entity-resolution" --display-name "Entity Resolution (Python 3.11)"
```

> **Windows note:** If `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` first.

---

## Step 4: Path Configuration

### Auto-detection (recommended — usually no action needed)

**No manual path change is needed in most cases.** The notebook's Section 0 auto-detects the project root by walking up the directory tree from the notebook's working directory until it finds the `dataset/` folder.

As long as the notebook file (`business_entity_resolution.ipynb`) is inside the `student_resource/` folder — which is the default — `BASE` is set correctly on Windows, Linux, and macOS.

### All paths in the notebook (Section 0 constants cell)

Every path in the notebook is derived from a single variable `BASE`. Here is the complete list:

| Variable | Path | Notes |
|----------|------|-------|
| `BASE` | `student_resource/` | Auto-detected project root |
| `TRAIN_S1_PATH` | `BASE/dataset/train/train_source1.tsv` | |
| `TRAIN_S2_PATH` | `BASE/dataset/train/train_source2.tsv` | |
| `TRAIN_S3_PATH` | `BASE/dataset/train/train_source3.tsv` | |
| `TRAIN_GT_PATH` | `BASE/dataset/train/train_ground_truth.tsv` | |
| `TEST_S1_PATH` | `BASE/dataset/test/test_source1.tsv` | |
| `TEST_S2_PATH` | `BASE/dataset/test/test_source2.tsv` | |
| `TEST_S3_PATH` | `BASE/dataset/test/test_source3.tsv` | |
| `MODELS_DIR` | `BASE/models/` | Created automatically |
| `OUTPUT_DIR` | `BASE/output/` | Created automatically |

**You only need to touch one thing:** if auto-detection fails, set `BASE` manually at the **top of the Section 0 constants cell**:

```python
# ── MANUAL OVERRIDE ── uncomment ONE of these if auto-detection fails ─────────
# BASE = Path(r"C:\Users\yourname\projects\student_resource")  # Windows
# BASE = Path("/home/yourname/projects/student_resource")       # Linux
# BASE = Path("/Users/yourname/projects/student_resource")      # macOS
```

### How to know if auto-detection worked

When you run Section 0, you will see:
```
Project root (BASE): /path/to/student_resource
```

If it prints the wrong path, or if you see:
```
AssertionError: dataset/ not found under ...
```
then uncomment the manual override above with the correct path.

---

## Step 5: Open the notebook

**Option A — VS Code:**
1. Open `business_entity_resolution.ipynb` in VS Code
2. Click the kernel picker (top right corner)
3. Select **"Entity Resolution (Python 3.11)"**
4. Run All Cells (`Ctrl+Shift+P` → "Run All Cells") or run cell by cell

**Option B — Jupyter in browser:**

Linux/macOS:
```bash
source venv/bin/activate
jupyter notebook business_entity_resolution.ipynb
```

Windows:
```cmd
venv\Scripts\activate
jupyter notebook business_entity_resolution.ipynb
```

Then select kernel: `Entity Resolution (Python 3.11)`

---

## Step 6: Run the notebook

Run cells **top to bottom** in order. The pipeline has 8 sections:

| Section | What it does | Estimated time |
|---------|-------------|----------------|
| 0 | Imports, constants, checkpoint helpers | < 1 min |
| 1 | Load and validate training data | ~3 min |
| 2 | Text normalization | ~5 min (saved to disk) |
| 3a | Blocking key functions (just defines functions) | < 1 min |
| 3b | Build inverted indices + generate candidates | **~15–20 min** (saved to disk) |
| 3c | Measure blocking recall | ~5 min |
| 4a | Feature extraction core functions | < 1 min |
| 4b | Fit TF-IDF + extract features | **~30–40 min** (vectorizers saved) |
| 5 | Train LightGBM model | **~15–20 min** (model saved) |
| 6 | Threshold optimization | ~5 min (result saved) |
| 7 | Test inference | **~30–40 min** (scores saved) |
| 8 | Write output files + validate | ~5 min |

**Total first run: ~2–2.5 hours**

### Resuming after a restart

Every expensive stage saves checkpoints to `models/`. On any subsequent run, just run all cells top to bottom again — each stage will detect its checkpoint and skip recomputation automatically.

To check what's already done:
```python
ckpt_status()  # run this in any cell after Section 0
```

---

## Step 7: Check the output

After the notebook completes successfully, you will find:

```
student_resource/
├── output/
│   ├── matching_results.tsv    ← submit this to the leaderboard
│   ├── candidate_pairs.tsv     ← include in the final zip
│   └── validation_pass.log     ← validator PASS confirmation
├── models/
│   ├── lgbm_classifier.pkl     ← trained model
│   ├── optimal_threshold.json  ← best threshold value
│   └── ...                     ← other checkpoints
└── docs/
    └── threshold_sweep.tsv     ← F0.5 vs threshold table
```

The validator must print `PASS` — if it doesn't, do not submit.

> **Note:** The notebook runs the validator automatically using `sys.executable` — no need to run it manually. If you do want to run it from the terminal:
> - Linux/macOS: `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
> - Windows: `python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test`

---

## Troubleshooting

**"No module named jellyfish / rapidfuzz / lightgbm"**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**"Kernel not found" in VS Code**
```bash
source venv/bin/activate
python -m ipykernel install --user --name "entity-resolution" --display-name "Entity Resolution (Python 3.11)"
```
Then reload VS Code window.

**Out of memory**
- Reduce `batch_size` in `extract_features_batch()` calls from `4096` to `1024`
- Close other applications

**SyntaxError in notebook cells**
- Make sure you are using the **Python 3.11** kernel, not Python 3.13 or 3.14

**Wrong BASE path**
If you see `FileNotFoundError` or an assertion error about `dataset/` not found:
- Make sure the notebook is opened from inside the `student_resource/` folder
- Or manually set `BASE` at the top of Section 0:
  ```python
  BASE = Path(r"C:\Users\yourname\projects\student_resource")   # Windows
  BASE = Path("/home/yourname/projects/student_resource")        # Linux/macOS
  ```

**Windows: `jellyfish` install error**
If you see a build error on Windows, try:
```cmd
pip install jellyfish --pre
```

**Windows: `rapidfuzz` install error**
```cmd
pip install rapidfuzz --only-binary :all:
```

**Windows: path separator issues**
All paths use `pathlib.Path` which handles `\` vs `/` automatically. You should not encounter any path separator errors.

---

## Dependency versions

All dependencies are pinned in `requirements.txt`:

```
pandas==2.1.0
numpy==1.24.4
scikit-learn==1.3.2
lightgbm==4.1.0
rapidfuzz==3.4.0
jellyfish==1.0.3
scipy==1.11.4
joblib==1.3.2
ipykernel==6.29.5
```

---

## Project structure

```
student_resource/
├── business_entity_resolution.ipynb   ← main notebook (run this)
├── requirements.txt                    ← pinned dependencies
├── SETUP_README.md                     ← this file
├── Documentation_template.md          ← fill in after training
├── dataset/                           ← NOT in git (download separately)
│   ├── train/
│   └── test/
├── models/                            ← NOT in git (generated by notebook)
├── output/                            ← NOT in git (generated by notebook)
├── docs/                              ← EDA results and analysis
│   ├── 02_data_analysis_plan.md
│   └── open_questions.md
├── utils/
│   └── validate_submission.py         ← challenge-provided validator
└── .kiro/specs/                       ← spec documentation
    └── business-entity-resolution/
        ├── requirements.md
        ├── design.md
        └── tasks.md
```
