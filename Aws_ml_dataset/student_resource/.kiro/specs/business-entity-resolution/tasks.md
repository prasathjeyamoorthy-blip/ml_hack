# Implementation Plan: Business Entity Resolution Pipeline

## Overview

All implementation goes into `business_entity_resolution.ipynb`. Tasks follow the notebook section structure from the design document. Each task maps to a notebook section and its associated functions. Tasks are largely sequential since each pipeline stage depends on the previous one.

## Tasks

- [x] 1. Section 0: Setup, constants and normalization dictionaries
  - Create the notebook's first section with all imports, path constants, and shared data structures
  - Define `LEGAL_SUFFIX_MAP` dict with English suffixes (ltd, llc, llp, inc, corp, pvt, co, plc, lp, pc) and French suffixes (sarl, sas, sasu, eurl, sci)
  - Define `STREET_TYPE_MAP` dict (st, rd, ave, blvd, dr, ln, ct, pl, hwy, pkwy)
  - Define `STOP_WORDS` frozenset and `LEGAL_TOKENS` frozenset (canonical post-expansion forms)
  - Define all path constants: TRAIN_S1_PATH, TRAIN_S2_PATH, TRAIN_S3_PATH, TRAIN_GT_PATH, TEST_S1_PATH, TEST_S2_PATH, TEST_S3_PATH
  - Define global config constants: RANDOM_SEED=42, MAX_CANDIDATES=500, NEG_SAMPLE_RATIO=20, FEATURE_NAMES list (17 items)
  - Define LGBM_PARAMS dict with all hyperparameters as specified in design
  - _Requirements: 1, 4, 8_

- [x] 2. Section 1: Data ingestion and validation functions
  - Implement `load_source(path, expected_prefix)` — reads TSV with sep="\t", utf-8; raises FileNotFoundError if missing; raises ValueError if REQUIRED_COLS absent; logs prefix warnings; returns DataFrame with str dtype
  - Implement `load_ground_truth(path)` — reads train_ground_truth.tsv; returns DataFrame with columns [source1_entity_id, matched_entity_ids]
  - Implement `validate_training_integrity(df_s1, df_s2, df_s3, df_gt)` — checks GT entity_ids == S1 entity_ids (exact set match); checks all matched IDs in GT exist in S2 or S3; raises ValueError with count and samples on failure
  - Implement `log_summary(df, name)` — prints row count, null% per column, unique country values, empty address%
  - Add execution cell: load all 4 training files, run validation, print summaries
  - _Requirements: 1, 2_

- [x] 3. Section 2: Text normalization functions
  - Implement `normalize_name(name)` applying steps in exact order: handle None/NaN → empty string; lowercase; strip whitespace; replace `&` with `and`; replace hyphens with space; remove non-alphanumeric-whitespace chars; expand LEGAL_SUFFIX_MAP tokens via whole-token word-boundary matching; collapse multiple spaces; preserve non-ASCII characters throughout
  - Implement `normalize_address(address)` applying: None/NaN → empty string, lowercase, strip, remove non-alphanumeric-whitespace, expand STREET_TYPE_MAP tokens (whole-token); no legal suffix expansion
  - Implement `normalize_dataframe(df)` — applies both functions to all rows, adds `norm_name` and `norm_address` columns, preserves originals, strips whitespace from country column
  - Add execution cell: normalize all 3 training DataFrames; print before/after samples for 5 rows
  - _Requirements: 4_

- [x] 4. Section 3a: Blocking key extraction functions
  - Implement `get_name_token_keys(norm_name)` — tokenize by whitespace, filter to tokens ≥2 chars, remove STOP_WORDS and LEGAL_TOKENS; return list
  - Implement `get_phonetic_key(norm_name)` — find first significant token; skip if contains non-ASCII chars; apply `jellyfish.metaphone()` on it; return list with 0 or 1 code
  - Implement `get_addr_token_keys(norm_address)` — extract all `\d+` tokens, convert via `str(int(t))` to strip leading zeros; extract tokens ≥3 chars that are not stop words; return combined list
  - Implement `get_char_ngram_keys(norm_name, n=3)` — only called if name has non-ASCII chars; generate all n-grams; apply `hash(gram) % 1024` for bucket; prefix with `"ngram_"`; return list
  - Implement `get_domain_name_keys(raw_name)` — strip leading `@` and `#`; detect domain pattern `\S+\.(com|net|org|in|co\.in|io|biz|fr)`; strip TLD; split on dots and hyphens; return token list or empty list
  - _Requirements: 6_

- [x] 5. Section 3b: Inverted index construction and candidate lookup
  - Implement `build_inverted_indices(df)` — builds 5 `defaultdict(list)` indices; for each row calls all 5 key functions and appends entity_id to each token's list; returns dict of 5 indices
  - Implement `lookup_candidates_one_source(s1_entity_id, s1_norm_name, s1_norm_address, s1_country, index, cand_df)` — builds all 5 key lists for S1 entity; unions entity_id sets from all index lookups; applies country filter (skip if either country is empty); removes s1_entity_id from results; returns list
  - Implement `prune_candidates(s1_norm_name, candidate_ids, cand_name_lookup, tfidf_vectorizer, tfidf_matrix, max_k=500)` — if len > max_k, score by TF-IDF cosine and return top max_k; else return unchanged
  - Implement `generate_all_candidates(df_s1, df_s2, df_s3, index_s2, index_s3, tfidf_vectorizer, tfidf_s2_matrix, tfidf_s3_matrix)` — for each S1: lookup S2, lookup S3, union, prune; return `{s1_id: [candidate_ids]}`
  - Add execution cell: build indices for train S2 and S3; run candidate generation; log total pairs, average candidates per S1, zero-candidate count
  - _Requirements: 5, 6_

- [x] 6. Section 3c: Blocking recall evaluation
  - Add execution cell to measure blocking recall on a 5,000 S1 entity sample from validation split
  - Parse ground truth into set of (s1_id, cand_id) true match pairs
  - Compute `recall_ceiling = |true_pairs where cand_id in candidates[s1_id]| / |total_true_pairs|`
  - Print recall ceiling, total candidate pairs, avg candidates per S1, zero-candidate count
  - If recall_ceiling < 0.90: log warning and add TF-IDF approximate retrieval as 6th blocking key; re-measure
  - _Requirements: 5_

- [x] 7. Section 4a: Feature extraction core functions
  - Implement `feature_normalize(text)` — lowercase + remove non-alphanumeric-whitespace + collapse whitespace + strip; returns empty string for None/NaN
  - Implement `compute_jaccard(tokens_a, tokens_b)` — `|A∩B|/|A∪B|`; returns 0.0 if both empty sets
  - Implement `compute_levenshtein_sim(a, b)` — uses `rapidfuzz.distance.Levenshtein.normalized_similarity`; returns 1.0 if both empty, 0.0 if exactly one empty
  - Implement `compute_char_ngram_sim(a, b, n=3)` — generate char n-gram sets; compute Jaccard; return 0.0 if both empty
  - Implement `extract_numeric_tokens(address)` — find all `\d+` patterns; convert each via `str(int(t))` to strip leading zeros; return as set
  - _Requirements: 7_

- [x] 8. Section 4b: Full feature vector extraction and TF-IDF setup
  - Fit `name_tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=100000)` on concat of all train S1+S2+S3 `norm_name` columns
  - Fit `addr_tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1), max_features=50000)` on concat of train S1+S2+S3 `norm_address` columns
  - Precompute sparse TF-IDF matrices for train S2 and S3 (name and address)
  - Save both vectorizers to `models/name_tfidf.pkl` and `models/addr_tfidf.pkl` using joblib
  - Implement `extract_feature_vector(s1_name, s1_addr, s1_country, cand_name, cand_addr, cand_country, cand_entity_id, ...)` — computes all 17 features; returns float32 array shape (17,)
  - Implement `extract_features_batch(candidate_pairs, s1_lookup, cand_lookup, name_vec, addr_vec, batch_size=4096)` — processes in batches; returns float32 ndarray shape (N_pairs, 17)
  - Add execution cell: extract features for 10,000 training candidate pairs; verify no NaN/Inf; print feature statistics
  - _Requirements: 7_

- [x] 9. Section 5: Model training
  - Perform train/validation split: `train_s1_ids, val_s1_ids = train_test_split(all_s1_ids, test_size=0.20, random_state=42)`
  - Implement `build_training_set(pairs, gt_set, features, neg_ratio=20, seed=42)` — separate positives and negatives; undersample negatives to `neg_ratio × n_positives`; return (X, y)
  - Extract full feature matrix for all training candidate pairs (train split only)
  - Build ground truth set from df_gt for training S1 entities
  - Call `build_training_set` to get balanced (X_train, y_train)
  - Extract feature matrix for validation split pairs → (X_val, y_val)
  - Implement `train_classifier(X_train, y_train, X_val, y_val, params)` — trains LightGBM with early stopping; logs feature importances; saves model to `models/lgbm_classifier.pkl`
  - Implement `predict_scores(clf, X, batch_size=65536)` — batched predict_proba[:, 1]; returns float64 array
  - Add execution cell: train model, print val binary logloss and feature importance table
  - _Requirements: 8, 9_

- [x] 10. Section 6: Threshold optimization and validation evaluation
  - Implement `compute_entity_f05(predicted_ids, true_ids)` — per-entity F0.5: both empty → 1.0; true non-empty + predicted empty → 0.0; true empty + predicted non-empty → 0.0; general formula `(1.25×P×R)/(0.25×P+R)`
  - Implement `compute_macro_f05(val_s1_ids, predictions_dict, gt_dict)` — mean of per-entity F0.5 over all val S1 entities including Singletons
  - Implement `sweep_thresholds(val_scores, val_pairs, val_s1_ids, gt, t_min=0.0, t_max=1.0, t_step=0.01)` — sweeps all thresholds; selects argmax macro F0.5 with higher-threshold tie-break; returns (optimal_t, results_df)
  - Implement `apply_threshold(scores, pairs, threshold)` — match IFF score strictly greater than threshold; return `{s1_id: [matched_ids]}`
  - Add execution cell: run threshold sweep; print optimal threshold and val F0.5/precision/recall; save `docs/threshold_sweep.tsv`
  - Optionally run `calibrate_per_country(...)` if per-country improvement > 0.01 F0.5
  - _Requirements: 9, 10, 11_

- [x] 11. Section 7: Test inference pipeline
  - Load test source files using same `load_source` function (test_source1.tsv, test_source2.tsv, test_source3.tsv)
  - Normalize test DataFrames using same `normalize_dataframe` function
  - Build inverted indices for test S2 and test S3
  - Load saved TF-IDF vectorizers from `models/` (do NOT refit); transform test S2 and S3 into sparse matrices
  - Generate candidates for all test S1 entities using `generate_all_candidates`
  - Extract features for all test candidate pairs using `extract_features_batch`
  - Load saved LightGBM model from `models/lgbm_classifier.pkl`; run `predict_scores`
  - Apply optimal threshold to get `matched` dict
  - Log all 7 stage metrics: records loaded, normalized, candidates generated, features extracted, pairs scored, matches accepted, output rows written
  - _Requirements: 12, 16_

- [x] 12. Section 8: Output file generation and submission validation
  - Create `output/` directory if it does not exist
  - Implement `write_matching_results(test_s1_ids, matched, path)` — one row per test S1 entity in original order; empty string (not nan/None) for no-match; comma-separated matched IDs with no trailing comma and no whitespace around separators; sorted entity_ids within each row for determinism
  - Implement `write_candidate_pairs(test_s1_ids, candidates, path)` — same format rules; every entity_id in matching_results must appear here; empty string for zero-candidate entities
  - Implement `run_validator(matching, candidate, test_dir, log_path)` — runs `python3 utils/validate_submission.py ...` via subprocess; saves stdout to log_path; returns True if exit code 0; raises after 3 failures
  - Add execution cell: write both output files; run validator; assert exit code 0; print PASS confirmation
  - _Requirements: 13, 14, 15_

## Task Dependency Graph

```
Task 1 (Setup & Constants)
  └── Task 2 (Data Ingestion)
        └── Task 3 (Normalization)
              └── Task 4 (Blocking Key Functions)
                    └── Task 5 (Inverted Index & Candidates)
                          ├── Task 6 (Blocking Recall Eval)
                          └── Task 7 (Feature Core Functions)
                                └── Task 8 (Feature Extraction & TF-IDF)
                                      └── Task 9 (Model Training)
                                            └── Task 10 (Threshold Optimization)
                                                  └── Task 11 (Test Inference)
                                                        └── Task 12 (Output & Validation)
```

## Notes

- All code goes into `business_entity_resolution.ipynb` — no separate Python files
- A placeholder notebook already exists at `business_entity_resolution.ipynb`; fill in each section's cells in order
- Save trained models to `models/` directory: `lgbm_classifier.pkl`, `name_tfidf.pkl`, `addr_tfidf.pkl`
- Write docs artifacts to `docs/`: `threshold_sweep.tsv`, `precision_recall_curve.tsv`
- Create `output/` directory before writing submission files
- The `utils/validate_submission.py` validator must return exit code 0 before any submission
- Fixed seed throughout: RANDOM_SEED=42 for all stochastic operations
- Do not refit TF-IDF vectorizers on test data — load from disk
- French legal suffixes (sarl, sas, sasu, eurl, sci) must be in LEGAL_SUFFIX_MAP even though they do not appear in training ground truth (France = 14.98% of test set)
