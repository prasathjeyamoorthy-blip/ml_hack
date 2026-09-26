# Business Entity Resolution — Setup & Run Guide

This guide explains how to set up the environment and run the full ML pipeline end-to-end.

---

## Prerequisites

- Linux or macOS (tested on Fedora Linux)
- Python 3.11 (required — other versions may have compatibility issues)
- At least **32 GB RAM** recommended
- At least **20 GB free disk space** for checkpoints and output files

Check your Python version:
```bash
python3.11 --version
```

If Python 3.11 is not installed on Fedora:
```bash
sudo dnf install python3.11
```

On Ubuntu/Debian:
```bash
sudo apt install python3.11 python3.11-venv
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

From inside the `student_resource/` directory:

```bash
python3.11 -m venv venv --prompt "entity-resolution"
```

Activate it:
```bash
source venv/bin/activate
```

Install all dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Register the kernel so Jupyter/VS Code can see it:
```bash
python -m ipykernel install --user --name "entity-resolution" --display-name "Entity Resolution (Python 3.11)"
```

---

## Step 4: Update the BASE path in the notebook

Open `business_entity_resolution.ipynb` and find **Section 0 → Constants cell**. Change the `BASE` path to match where you placed the project on your machine:

```python
# Change this to your actual path
BASE = Path("/your/path/to/student_resource")
```

For example:
- Linux: `Path("/home/username/projects/student_resource")`
- macOS: `Path("/Users/username/projects/student_resource")`

This is the **only line you need to change** in the entire notebook.

---

## Step 5: Open the notebook

**Option A — VS Code:**
1. Open `business_entity_resolution.ipynb` in VS Code
2. Click the kernel picker (top right corner)
3. Select **"Entity Resolution (Python 3.11)"**
4. Run All Cells (`Ctrl+Shift+P` → "Run All Cells") or run cell by cell

**Option B — Jupyter in browser:**
```bash
source venv/bin/activate
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
If you see `FileNotFoundError`, update the `BASE` path in Section 0 to match your directory.

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
