# Design Document: Business Entity Resolution Pipeline

## Overview

This document describes the technical design of an ML-based Business Entity Resolution pipeline. The system resolves which records across three independent business data sources (Source1, Source2, Source3) refer to the same real-world business entity.

- **Evaluation metric:** F0.5 (macro-averaged, precision-heavy — precision weighted 2× over recall)
- **Implementation target:** Single Jupyter notebook `business_entity_resolution.ipynb`
- **Language:** Python 3.10+
- **No external APIs or business databases permitted**

All design decisions are grounded in EDA findings from `docs/02_data_analysis_plan.md`.

---

## System Architecture

```
TRAINING MODE                              INFERENCE MODE
──────────────────────────────────         ──────────────────────────────────
 train_source1/2/3.tsv                      test_source1/2/3.tsv
 train_ground_truth.tsv                              │
        │                                            │
        ▼                                            ▼
 ┌─────────────────────┐               ┌─────────────────────┐
 │ 1. Data Ingestion   │               │ 1. Data Ingestion   │
 │    & Validation     │               │    & Validation     │
 └──────────┬──────────┘               └──────────┬──────────┘
            │ df_s1, df_s2, df_s3                  │
            ▼                                      ▼
 ┌─────────────────────┐               ┌─────────────────────┐
 │ 2. Text             │               │ 2. Text             │
 │    Normalization    │               │    Normalization    │
 └──────────┬──────────┘               └──────────┬──────────┘
            │ + norm_name, norm_address             │
            ▼                                      ▼
 ┌─────────────────────┐               ┌─────────────────────┐
 │ 3. Train/Val Split  │               │ 3. Multi-Key        │
 │    (seed=42, 80/20) │               │    Blocking         │
 │ + Multi-Key         │               │    (inverted index) │
 │    Blocking         │               └──────────┬──────────┘
 └──────────┬──────────┘                          │ ≤500 candidates/S1
            │ candidate pairs                     ▼
            ▼                          ┌─────────────────────┐
 ┌─────────────────────┐               │ 4. Feature          │
 │ 4. Feature          │               │    Extraction       │
 │    Extraction       │               │    (17 features)    │
 │    + TF-IDF fit     │               └──────────┬──────────┘
 └──────────┬──────────┘                          │ feature matrix
            │ X_train (17 features)               ▼
            ▼                          ┌─────────────────────┐
 ┌─────────────────────┐               │ 5. LightGBM         │
 │ 5. LightGBM         │               │    Inference        │
 │    Training         │  ──model──►  │    (predict_proba)  │
 │    (class weighted) │               └──────────┬──────────┘
 └──────────┬──────────┘                          │ scores [0,1]
            │ trained model                       ▼
            ▼                          ┌─────────────────────┐
 ┌─────────────────────┐               │ 6. Threshold Apply  │
 │ 6. Threshold        │               │    (optimal τ)      │
 │    Optimization     │               └──────────┬──────────┘
 │    (F0.5 sweep)     │                          │
 └──────────┬──────────┘                          ▼
            │ optimal threshold τ      ┌─────────────────────┐
            ▼                          │ 7. Output           │
 ┌─────────────────────┐               │    Generation       │
 │ 7. Validation       │               │    matching +       │
 │    Evaluation       │               │    candidate TSVs   │
 └─────────────────────┘               └──────────┬──────────┘
                                                   ▼
                                        validate_submission.py
                                        → output/validation_pass.log
```

---

## Stage 1: Data Ingestion & Validation

### Purpose
Load all TSV files into pandas DataFrames, enforce schema, and validate referential integrity of the training ground truth.

### Key Design Decisions
- **Fail-fast on wrong separator.** The challenge explicitly warns that reading without `sep="\t"` silently produces a single-column DataFrame. Detect this early.
- **`country` is always an open-set string.** Never encode, filter, or one-hot it.
- **Log summary stats to stdout** as an audit trail every run.

### Function Signatures

```python
REQUIRED_COLS = ["entity_id", "business_name", "business_address", "country"]

def load_source(path: str, expected_prefix: str) -> pd.DataFrame:
    """
    Load a source TSV file.
    - Enforces sep="\t", encoding="utf-8"
    - Raises FileNotFoundError if path does not exist
    - Raises ValueError if REQUIRED_COLS are missing
    - Logs warning for entity_ids not starting with expected_prefix
    - Returns DataFrame with all 4 columns as str dtype
    """

def load_ground_truth(path: str) -> pd.DataFrame:
    """
    Load train_ground_truth.tsv.
    Columns: source1_entity_id (str), matched_entity_ids (str, may be empty)
    """

def validate_training_integrity(df_s1, df_s2, df_s3, df_gt) -> None:
    """
    1. Assert df_gt.source1_entity_id == df_s1.entity_id (exact set match)
    2. Assert all IDs in df_gt.matched_entity_ids exist in df_s2 or df_s3
    Raises DataValidationError with counts and samples on failure.
    """

def log_summary(df: pd.DataFrame, name: str) -> None:
    """Print row count, null%, country value counts, empty address%."""
```

### Data Structures

```python
# After load_source():
df_s1 / df_s2 / df_s3:
  entity_id:        str   # "S1-XXXXXXX", "S2-XXXXXXX", "S3-XXXXXXX"
  business_name:    str   # raw, may have whitespace
  business_address: str   # raw, ~3.3% empty in S2/S3
  country:          str   # "US", "India", "France" — open-set

# After load_ground_truth():
df_gt:
  source1_entity_id: str
  matched_entity_ids: str  # "S2-001,S3-042" or "" for Singletons
```

### Memory Estimate
~2.2M S1 + 10.3M S2+S3 rows × 4 string columns ≈ **4–5 GB peak**

---

## Stage 2: Text Normalization

### Purpose
Produce consistent canonical forms to maximize token overlap between true-match pairs despite surface noise.

### Transformation Order (FIXED — must not be reordered)
1. Convert to lowercase
2. Strip leading/trailing whitespace
3. Normalize punctuation (`&` → `and`, hyphens → space, remove non-alphanumeric-whitespace)
4. Expand legal suffix abbreviations (whole-token matching only)

**Why this order:** Lowercase before dictionary lookup (case-insensitive matching). Strip before punctuation normalization (avoids whitespace edge cases). Punctuation before suffix expansion so `L.L.C.` becomes `llc` before the dictionary runs.

### Normalization Dictionaries

```python
# Legal suffix expansions (English + French — whole-token matching)
LEGAL_SUFFIX_MAP = {
    # English / US / India
    "ltd": "limited", "llc": "limited liability company",
    "llp": "limited liability partnership", "inc": "incorporated",
    "corp": "corporation", "pvt": "private", "co": "company",
    "plc": "public limited company", "lp": "limited partnership",
    "pc": "professional corporation",
    # French (14.98% of test set — no training ground truth but needed at inference)
    "sarl": "societe a responsabilite limitee",
    "sas": "societe par actions simplifiee",
    "sasu": "societe par actions simplifiee unipersonnelle",
    "eurl": "entreprise unipersonnelle a responsabilite limitee",
    "sci": "societe civile immobiliere",
}

# Street type abbreviations (whole-token matching)
STREET_TYPE_MAP = {
    "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "hwy": "highway", "pkwy": "parkway",
}

# Stop words (for blocking key extraction, NOT for feature normalization)
STOP_WORDS = frozenset(["the","and","of","a","an","in","on","at","for","to","with","by","from"])

# Legal suffix tokens in canonical form (removed from blocking keys)
LEGAL_TOKENS = frozenset(["limited","company","corporation","incorporated","partnership",
                           "private","public","associates","services","societe","limitee", ...])
```

### Function Signatures

```python
def normalize_name(name: str) -> str:
    """
    Apply 4-step normalization to a business name.
    - None/NaN → empty string (never raises)
    - Preserves non-ASCII scripts (Devanagari, Arabic, etc.)
    - Examples:
        "ABC Pvt. Ltd."  → "abc private limited"
        "Ram & Sons LLC" → "ram and sons limited liability company"
        "L.L.C."         → "limited liability company"
        "ZNB Club SARL"  → "znb club societe a responsabilite limitee"
    """

def normalize_address(address: str) -> str:
    """
    Apply normalization to an address.
    Steps: lowercase → strip → remove non-alphanumeric-whitespace → expand street types.
    Does NOT expand legal suffixes.
    None/NaN → empty string (never raises).
    """

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns df with two new columns: norm_name, norm_address.
    Original columns are preserved unchanged.
    country is pass-through (strip whitespace only).
    """
```

### Performance Note
Use vectorized pandas `.str` operations for steps 1–3. Use a compiled regex with `re.sub` for suffix expansion to avoid per-row Python loop overhead. At 15M rows, this needs to run in ≤5 minutes.

### Memory Estimate
Adds 2 string columns per DataFrame. Peak: **~6 GB**

---

## Stage 3: Multi-Key Blocking

### Purpose
Reduce 22.77 trillion possible pairs to ≤500 candidates per S1 entity (~866M pairs max) while achieving ≥90% recall ceiling.

### Architecture: Inverted Indices

Build one set of inverted indices for Source2, another for Source3. Each index maps a blocking token to the list of entity_ids containing that token.

```python
InvertedIndex = dict[str, list[str]]   # token → [entity_ids]

# Five separate indices per source:
index = {
    "name_token":  InvertedIndex,   # significant name tokens
    "phonetic":    InvertedIndex,   # Double Metaphone codes
    "addr_token":  InvertedIndex,   # address numeric + significant tokens
    "char_ngram":  InvertedIndex,   # 3-gram hash buckets (non-ASCII names only)
    "domain_name": InvertedIndex,   # stripped domain tokens
}
```

### Five Blocking Keys

| Key | Input | Method | Noise covered |
|-----|-------|--------|---------------|
| `name_token` | norm_name | Tokens ≥2 chars, stop words and legal suffix tokens removed | Legal suffix drop, case, punctuation (84.76% recall alone) |
| `phonetic` | First Latin-script significant token | `jellyfish.metaphone()` or Double Metaphone | Spelling variations, OCR errors |
| `addr_token` | norm_address | Numeric tokens (leading zeros stripped) + tokens ≥3 chars, non-stop | Transliteration misses — **97.4% recovery** for Devanagari pairs |
| `char_ngram` | norm_name (non-ASCII only) | 3-grams, modulo-1024 bucket hash | Devanagari/Telugu/Arabic script names |
| `domain_name` | raw business_name (S2/S3) | Strip `@`/`#`, detect `.com/.in/.net/...`, strip TLD, split on `.`/`-` | 4.0% of S2/S3 records are domain names (31.1% of token-blocking misses) |

**Social handle pre-processing:** Strip leading `@` and `#` from business_name before any key extraction so `@cornerstoneinvestments` → `cornerstoneinvestments` → recovered by `name_token` key.

**Country-aware filtering:** After candidate lookup, keep only candidates where `candidate.country == s1.country` (case-insensitive, strip whitespace). Skip this filter if either country is empty/null. This is critical because 30.25% of S1 names are duplicated across different locations.

### Candidate Cap & Pruning

```python
MAX_CANDIDATES = 500

def prune_candidates(
    s1_norm_name: str,
    candidate_ids: list[str],
    cand_name_lookup: dict[str, str],   # entity_id → norm_name
    tfidf_vectorizer: TfidfVectorizer,
    tfidf_matrix: csr_matrix,           # precomputed for all S2+S3
    max_k: int = MAX_CANDIDATES,
) -> list[str]:
    """
    If len(candidate_ids) > max_k:
      - Score all candidates by TF-IDF cosine(S1_name, candidate_name)
      - Return top max_k by descending score
    Else: return candidate_ids unchanged.
    """
```

### Function Signatures

```python
def build_inverted_indices(df: pd.DataFrame) -> dict[str, InvertedIndex]:
    """Build all 5 inverted indices for a source DataFrame."""

def get_name_token_keys(norm_name: str) -> list[str]:
    """Tokens ≥2 chars from norm_name after removing STOP_WORDS and LEGAL_TOKENS."""

def get_phonetic_key(norm_name: str) -> list[str]:
    """Double Metaphone of first qualifying Latin-script token. Empty if non-Latin."""

def get_addr_token_keys(norm_address: str) -> list[str]:
    """Numeric tokens (int(tok) → str, strips zeros) + ≥3 char non-stop tokens."""

def get_char_ngram_keys(norm_name: str, n: int = 3) -> list[str]:
    """3-gram bucket hashes. Only applied if norm_name contains non-ASCII chars."""

def get_domain_name_keys(raw_name: str) -> list[str]:
    """Detect domain pattern, strip TLD + @/# prefix, split on ./-."""

def lookup_candidates_one_source(
    s1_entity_id: str,
    s1_norm_name: str,
    s1_norm_address: str,
    s1_country: str,
    index: dict[str, InvertedIndex],
    cand_df: pd.DataFrame,
) -> list[str]:
    """
    Union results from all 5 blocking keys.
    Apply country filter.
    Return raw candidate list (before cap pruning).
    """

def generate_all_candidates(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    index_s2: dict[str, InvertedIndex],
    index_s3: dict[str, InvertedIndex],
    tfidf_vectorizer: TfidfVectorizer,
    tfidf_s2_matrix: csr_matrix,
    tfidf_s3_matrix: csr_matrix,
) -> dict[str, list[str]]:
    """
    For each S1 entity:
      1. Lookup from S2 index
      2. Lookup from S3 index
      3. Union S2 + S3 candidates
      4. Prune to ≤500 via TF-IDF if needed
    Returns: {s1_entity_id: [candidate_entity_ids]}
    """
```

### Memory Estimate for Inverted Index
~5M S2 records × ~5 tokens/record = 25M index entries. Similar for S3. Python dicts with list values: **~4–6 GB**. Use `collections.defaultdict(list)`.

---

## Stage 4: Pairwise Feature Extraction

### Purpose
Compute a 17-dimensional float32 feature vector for each (S1, candidate) pair to feed the LightGBM classifier.

### Feature Pre-Normalization (different from blocking normalization)
Only steps 1+3: lowercase → remove non-alphanumeric-whitespace → collapse whitespace. No suffix expansion (to preserve similarity signal from suffix presence/absence).

### 17 Features

| # | Name | Type | Formula |
|---|------|------|---------|
| 0 | `name_jaccard` | float | `\|A∩B\| / \|A∪B\|`, 0.0 if both empty |
| 1 | `name_levenshtein` | float | `1 - edit_dist(a,b) / max(len(a),len(b))`, 1.0 both empty, 0.0 one empty |
| 2 | `name_tfidf_cosine` | float | TF-IDF cosine (vectorizer fit on full training corpus) |
| 3 | `name_token_overlap` | float | `\|A∩B\| / max(\|A\|,\|B\|)`, 0.0 if both empty |
| 4 | `name_char_ngram_sim` | float | Jaccard on char 3-gram sets |
| 5 | `name_length_diff` | float | `abs(len(a)-len(b)) / max(len(a),len(b),1)` |
| 6 | `name_prefix_match` | int | 1 if first token of a == first token of b, else 0 |
| 7 | `name_suffix_match` | int | 1 if last non-legal token of a == last non-legal token of b, else 0 |
| 8 | `addr_token_overlap` | float | `\|A∩B\| / max(\|A\|,\|B\|)` on address token sets, 0.0 if both empty |
| 9 | `addr_levenshtein` | float | Same formula as #1, on norm addresses |
| 10 | `addr_numeric_overlap` | float | `\|nums_A∩nums_B\| / max(\|nums_A\|,\|nums_B\|)`, 0.0 if both empty |
| 11 | `addr_empty_flag` | int | 1 if either address is empty/null, else 0 |
| 12 | `addr_tfidf_cosine` | float | TF-IDF cosine on addresses (separate vectorizer) |
| 13 | `country_match` | int | 1 if country_a.lower().strip() == country_b.lower().strip(), else 0 |
| 14 | `name_addr_product` | float | `name_jaccard × addr_token_overlap` |
| 15 | `max_similarity` | float | `max(name_jaccard, name_tfidf_cosine, name_token_overlap)` |
| 16 | `is_domain_candidate` | int | 1 if candidate name matches domain regex pattern, else 0 |

**Null handling:** If either field is empty, return 0.0 for all similarity features for that field. Never raise.

### Function Signatures

```python
FEATURE_NAMES = [
    "name_jaccard", "name_levenshtein", "name_tfidf_cosine", "name_token_overlap",
    "name_char_ngram_sim", "name_length_diff", "name_prefix_match", "name_suffix_match",
    "addr_token_overlap", "addr_levenshtein", "addr_numeric_overlap", "addr_empty_flag",
    "addr_tfidf_cosine", "country_match", "name_addr_product", "max_similarity",
    "is_domain_candidate"
]

def feature_normalize(text: str) -> str:
    """Lowercase + remove non-alphanumeric-whitespace + collapse whitespace."""

def compute_jaccard(tokens_a: set, tokens_b: set) -> float:
    """Jaccard: |A∩B|/|A∪B|. Returns 0.0 if both empty."""

def compute_levenshtein_sim(a: str, b: str) -> float:
    """Uses rapidfuzz.distance.Levenshtein.normalized_similarity. Edge cases handled."""

def extract_numeric_tokens(address: str) -> set:
    """Extract r'\\d+' tokens, strip leading zeros via int() conversion."""

def extract_feature_vector(
    s1_name: str, s1_addr: str, s1_country: str,
    cand_name: str, cand_addr: str, cand_country: str,
    cand_entity_id: str,
    tfidf_name_vecs: csr_matrix,   # precomputed sparse vectors
    tfidf_addr_vecs: csr_matrix,
    s1_name_vec_idx: int,
    cand_vec_idx: int,
) -> np.ndarray:
    """Compute all 17 features. Returns float32 array of shape (17,)."""

def extract_features_batch(
    candidate_pairs: list[tuple[str, str]],
    s1_lookup: dict,     # entity_id → {norm_name, norm_address, country}
    cand_lookup: dict,   # entity_id → {norm_name, norm_address, country}
    tfidf_name_vectorizer: TfidfVectorizer,
    tfidf_addr_vectorizer: TfidfVectorizer,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Extract features for all pairs in batches.
    Returns np.ndarray shape (N_pairs, 17), dtype float32.
    """
```

### TF-IDF Vectorizer Setup
- Fit `name_tfidf_vectorizer` on `concat(S1+S2+S3)["business_name"]` from training data
- Fit `addr_tfidf_vectorizer` on `concat(S1+S2+S3)["business_address"]` from training data
- Save both vectorizers to disk; reuse at inference time (do NOT refit on test data)
- Use `scipy.sparse.csr_matrix` for the TF-IDF matrices (memory-efficient)

### Memory Estimate
Feature matrix at 15M training pairs × 17 features × float32 = **~1 GB**. Process in batches of 4096 to avoid OOM.

---

## Stage 5: LightGBM Binary Classifier

### Purpose
Score each (S1, candidate) pair with a probability in [0, 1] that the pair represents the same real-world entity.

### Model Choice Rationale
LightGBM on 17 hand-crafted features:
- Handles severe class imbalance (0.69% positive rate) via `scale_pos_weight`
- Fast training (≤15 min on 15M pairs) and inference (≤20 min for 850M test pairs with batching)
- No GPU needed; CPU deployment is sufficient
- MIT license; well within the 8B parameter constraint (it has no "parameters" in that sense)
- Feature importance output enables debugging of blocking quality

### Training Setup

```python
RANDOM_SEED = 42
NEG_SAMPLE_RATIO = 20   # negatives per positive — reduces training set from ~150M to ~15M pairs

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "scale_pos_weight": NEG_SAMPLE_RATIO,  # compensate for undersampling
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}
```

### Train/Validation Split
- **Split unit:** Source1 entity (all candidate pairs for a given S1 entity go to the same split)
- **Ratio:** 80% train, 20% validation (satisfies the "15–20%" validation requirement)
- **Method:** `sklearn.model_selection.train_test_split` with `random_state=RANDOM_SEED`
- **No leakage:** TF-IDF vectorizers are fit on the full training S1+S2+S3 corpus (not just the train split), but ground truth labels from validation S1 entities are never used for training

### Negative Sampling Strategy

```python
def build_training_set(
    pairs: list[tuple[str, str]],
    gt_set: set[tuple[str, str]],  # set of (s1_id, cand_id) true matches
    features: np.ndarray,
    neg_ratio: int = NEG_SAMPLE_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1. Separate positive and negative pairs
    2. Randomly undersample negatives to neg_ratio × n_positives
    3. Return (X, y) with balanced sampling
    """
```

### Function Signatures

```python
def train_classifier(X_train, y_train, X_val, y_val, params=LGBM_PARAMS) -> lgb.LGBMClassifier:
    """Train with early stopping on val logloss. Print feature importances."""

def predict_scores(clf: lgb.LGBMClassifier, X: np.ndarray, batch_size=65536) -> np.ndarray:
    """Batch prediction. Returns float64 array shape (N,) in [0,1]."""
```

### Hyperparameter Tuning Plan

| Parameter | Search values | Priority |
|-----------|---------------|----------|
| `num_leaves` | 63, 127, 255 | Medium |
| `n_estimators` | 300, 500, 1000 | High (use early stopping) |
| `learning_rate` | 0.03, 0.05, 0.1 | Medium |
| `scale_pos_weight` | 10, 20, 50 | High (affects precision/recall balance) |
| `min_child_samples` | 10, 20, 50 | Low |

Start with defaults, run 3–4 grid search experiments. Each run ≤30 min.

---

## Stage 6: Threshold Optimization

### Purpose
Select the confidence score threshold that maximizes macro-averaged F0.5 on the validation set.

### F0.5 Formula

```
Per-entity:
  P = |predicted ∩ true| / |predicted|    (0.0 if predicted is empty)
  R = |predicted ∩ true| / |true|          (0.0 if true is empty)
  F0.5 = (1.25 × P × R) / (0.25 × P + R)
  
Special cases:
  Both empty (correct Singleton prediction): F0.5 = 1.0
  True non-empty, predicted empty:           F0.5 = 0.0
  True empty, predicted non-empty:           P = 0.0, so F0.5 = 0.0

Macro average: mean(F0.5 per entity) over ALL S1 entities in validation split
```

### Threshold Sweep Algorithm

```python
def sweep_thresholds(
    val_scores: np.ndarray,
    val_pairs: list[tuple[str, str]],
    val_s1_ids: list[str],          # ALL S1 entities in val (including Singletons)
    gt: dict[str, set[str]],        # {s1_id: {true_match_ids}}
    t_min=0.0, t_max=1.0, t_step=0.01,
) -> tuple[float, pd.DataFrame]:
    """
    For each threshold t in range(t_min, t_max+t_step, t_step):
      - predictions = {s1_id: {cand_id for (s1_id,cand_id),score if score > t}}
      - macro_f05 = mean(compute_entity_f05(predictions[s1], gt[s1]) for s1 in val_s1_ids)
    
    Select t* = argmax macro_f05; tie-break: select HIGHER threshold (favours precision).
    Return (t*, results_df) where results_df has columns [threshold, precision, recall, f05].
    """
```

**Tie-breaking rule:** If two thresholds produce equal macro F0.5, select the higher one. This systematically favours precision, aligned with the F0.5 metric.

**Expected range from EDA:** With only 5.58% true Singletons and 0.69% positive rate in candidates, the optimal threshold is expected in the 0.6–0.8 range.

### Per-Country Calibration (optional)

```python
def calibrate_per_country(val_scores, val_pairs, val_s1_df, gt, global_t) -> dict[str, float]:
    """
    For each country in val_s1_df['country'].unique():
      - Sweep threshold on that country's val entities only
      - If country_f05_at_country_t > global_f05 + 0.01: use country-specific t
    Returns {country: threshold, "__global__": global_t}.
    """
```

### Output Artifacts
- `docs/threshold_sweep.tsv` — threshold vs F0.5/precision/recall table
- `docs/precision_recall_curve.tsv` — for plotting

---

## Stage 7: Multi-Match Output Generation

### Purpose
Produce both required output TSV files in the exact format required by the challenge validator.

### Output Format

```
matching_results.tsv:
  Header: source1_entity_id\tmatched_entity_ids
  Row (match):     S1-0001234\tS2-100,S3-050
  Row (no match):  S1-0001235\t        ← empty string, NOT "nan"

candidate_pairs.tsv:
  Header: source1_entity_id\tcandidate_entity_ids
  Row:    S1-0001234\tS2-100,S2-200,S3-050
  Row:    S1-0001235\t        ← empty if zero candidates
```

**Critical invariant:** Every entity_id in `matched_entity_ids` MUST also appear in `candidate_entity_ids` for the same S1 entity.

### Function Signatures

```python
def apply_threshold(
    scores: np.ndarray,
    pairs: list[tuple[str, str]],
    threshold: float,
) -> dict[str, list[str]]:
    """
    Match IFF score > threshold (strictly greater than).
    Returns {s1_id: [matched_cand_ids]}, empty list for no-match S1 entities.
    """

def write_matching_results(
    test_s1_ids: list[str],
    matched: dict[str, list[str]],
    path: str = "output/matching_results.tsv",
) -> None:
    """
    Write matching_results.tsv.
    - Exactly one row per test S1 entity
    - Empty string (not nan) for no-match
    - Comma-separated, no trailing comma, no whitespace around commas
    - Sorted entity_ids within each matched list for determinism
    """

def write_candidate_pairs(
    test_s1_ids: list[str],
    candidates: dict[str, list[str]],
    path: str = "output/candidate_pairs.tsv",
) -> None:
    """Write candidate_pairs.tsv. Same format rules as matching_results.tsv."""

def run_validator(
    matching="output/matching_results.tsv",
    candidate="output/candidate_pairs.tsv",
    test_dir="dataset/test",
    log_path="output/validation_pass.log",
) -> bool:
    """
    Run: sys.executable utils/validate_submission.py --matching ... --candidate ... --test-dir ...
    (uses sys.executable for cross-platform compatibility — Windows/Linux/macOS)
    Save stdout to log_path.
    Returns True if exit code 0. Raises SubmissionValidationError after 3 failures.
    """
```

---

## Data Flow Summary

```
Input Files
  → DataFrames (entity_id, business_name, business_address, country)
  → Normalized DataFrames (+ norm_name, norm_address)
  → Inverted Indices (5 per source × 2 sources = 10 indices)
  → Candidate Dict {s1_id: [cand_ids]}  ≤500 per S1
  → Flat Pair List [(s1_id, cand_id), ...]
  → Feature Matrix  (N_pairs × 17, float32)
  → Score Array     (N_pairs,) float64
  → Matched Dict    {s1_id: [matched_cand_ids]}
  → output/matching_results.tsv
  → output/candidate_pairs.tsv
```

---

## Technology Stack

| Library | Version | Use | Rationale |
|---------|---------|-----|-----------|
| `pandas` | ≥1.5 | Data loading, normalization, output | Vectorized str ops; canonical tabular ML |
| `numpy` | ≥1.23 | Feature matrices, numeric ops | Foundation |
| `scikit-learn` | ≥1.1 | TfidfVectorizer, train/val split | Standard; sparse TF-IDF; StratifiedShuffleSplit |
| `lightgbm` | ≥4.0 | Binary classifier | Best gradient boosting for tabular; CPU-fast; MIT license |
| `rapidfuzz` | ≥3.0 | Levenshtein similarity | C-extension; ~100× faster than pure Python |
| `jellyfish` | ≥0.9 | Phonetic encoding (Double Metaphone) | Standard; MIT license |
| `scipy` | ≥1.9 | Sparse TF-IDF matrices, cosine similarity | Memory-efficient sparse matrix operations |
| `re` (stdlib) | — | Regex for domain detection, tokenization | No external dependency |
| `unicodedata` (stdlib) | — | Non-ASCII character detection | Script type identification |
| `collections` (stdlib) | — | defaultdict for inverted index | O(1) amortized insert |

**Not used:**
- `recordlinkage` / `dedupe`: Would hide the 5-key blocking architecture; prohibited by challenge rules
- `faiss`: GPU/HNSW overkill for 5-key inverted index approach
- `torch`/`transformers`: No embedding model needed; hand-crafted features + LightGBM is sufficient and faster

---

## Memory & Performance Budget

| Stage | Wall-clock | Peak RAM | Notes |
|-------|-----------|---------|-------|
| Ingestion (train) | ~3 min | ~4.5 GB | 15M rows pandas |
| Normalization (train) | ~5 min | ~6 GB | Vectorized str ops |
| Build inverted indices | ~8 min | ~8 GB | 5 dicts for S2+S3 |
| TF-IDF fit + transform | ~10 min | ~6 GB | 15M names, sparse |
| Candidate generation | ~15 min | ~10 GB | 2.2M S1 × 5 keys |
| Feature extraction | ~30 min | ~8 GB | ~15M training pairs |
| LightGBM training | ~10 min | ~4 GB | 15M pairs × 17 feats |
| Threshold sweep | ~5 min | ~2 GB | Val split only |
| **Total training** | **~86 min** | **~12 GB peak** | |
| Ingestion + norm (test) | ~6 min | ~5 GB | |
| Candidate gen (test) | ~12 min | ~10 GB | 1.7M S1 |
| Feature extraction (test) | ~25 min | ~8 GB | Batched |
| LightGBM inference (test) | ~20 min | ~4 GB | Batched predict |
| Output generation | ~2 min | ~1 GB | |
| **Total inference** | **~65 min** | **~12 GB peak** | |
| **Grand total** | **~2.5 hours** | **~12 GB** | Well within 4h / 32GB |

### Memory Optimization Strategies
1. Convert DataFrames to `dict[str, dict]` lookup after normalization — O(1) access, lower overhead than repeated DataFrame indexing
2. Use `float32` for feature matrix (halves memory vs float64)
3. Store TF-IDF matrices as `scipy.sparse.csr_matrix`
4. Process candidate generation in S1 entity chunks of 10,000
5. Use `del df_raw; gc.collect()` after raw DataFrames are no longer needed
6. Use `lgb.Dataset(free_raw_data=True)` to release NumPy arrays after LightGBM dataset creation

---

## Notebook Cell Organization

```
business_entity_resolution.ipynb

Section 0: Setup & Constants
  - Imports
  - Path constants
  - Normalization dictionaries (LEGAL_SUFFIX_MAP, STREET_TYPE_MAP, STOP_WORDS)
  - Global config (RANDOM_SEED, MAX_CANDIDATES, NEG_SAMPLE_RATIO, LGBM_PARAMS)

Section 1: Data Ingestion & Validation
  - load_source(), load_ground_truth(), validate_training_integrity(), log_summary()
  - Cell: load all training data, run validation, print summaries

Section 2: Text Normalization
  - normalize_name(), normalize_address(), normalize_dataframe()
  - Cell: normalize all training DataFrames

Section 3: Blocking & Candidate Generation
  - All key extraction functions
  - build_inverted_indices(), lookup_candidates_one_source(), generate_all_candidates()
  - Cell: build indices for S2+S3, generate candidates, log recall stats

Section 4: Feature Extraction
  - feature_normalize(), compute_jaccard(), compute_levenshtein_sim()
  - extract_numeric_tokens(), extract_feature_vector(), extract_features_batch()
  - Cell: fit TF-IDF vectorizers, extract features for training pairs

Section 5: Model Training
  - build_training_set(), train_classifier(), predict_scores()
  - Cell: train LightGBM, print feature importances

Section 6: Threshold Optimization & Validation
  - compute_entity_f05(), compute_macro_f05(), sweep_thresholds()
  - Cell: sweep thresholds, print optimal τ, save threshold table

Section 7: Test Inference
  - Cell: load test data → normalize → build test indices → generate test candidates
          → extract test features → predict scores → apply threshold → write outputs

Section 8: Submission Validation
  - run_validator()
  - Cell: run validate_submission.py, assert PASS, save validation_pass.log
```

---

## Error Handling

| Stage | Error | Behavior |
|-------|-------|----------|
| Ingestion | File not found | `FileNotFoundError` with full path |
| Ingestion | Wrong separator (CSV not TSV) | `ValueError` "sep must be '\\t'" |
| Ingestion | Missing column | `ValueError` with column name |
| Validation | GT ↔ S1 mismatch | `ValueError` with count + 10 samples |
| Validation | Dangling GT references | `ValueError` with count + 10 samples |
| Normalization | Null/empty input | Silent empty string, never raises |
| Blocking | Zero candidates for S1 | Log to list, continue, write empty row in output |
| Feature extraction | Empty field | Return 0.0 for all features of that field |
| Classifier | Training fails | Halt, log error, no output files written |
| Submission validator | Fails after 3 retries | `SubmissionValidationError`, record in open_questions.md |

---

## Experiment Plan

| # | Experiment | Goal | Pass criterion |
|---|-----------|------|----------------|
| 1 | Blocking recall measurement | Confirm ≥90% recall with 5-key blocking | recall_ceiling ≥ 0.90 on 5k S1 sample |
| 2 | Negative sampling ratio | Find optimal neg:pos ratio | Val F0.5 > 0.80 |
| 3 | Initial threshold sweep | Characterize F0.5 vs threshold curve | Peak in 0.5–0.9 range |
| 4 | LightGBM hyperparameter grid | Improve val F0.5 by 0.01–0.02 | Each run ≤30 min |
| 5 | Per-country threshold calibration | Check if France needs different threshold | Improvement > 0.01 F0.5 |
| 6 | Feature importance analysis | Identify top features, debug blocking | Top feature is name-based |
| 7 | Full end-to-end test inference | Confirm ≤4h runtime, PASS validator | validate_submission.py exit 0 |
