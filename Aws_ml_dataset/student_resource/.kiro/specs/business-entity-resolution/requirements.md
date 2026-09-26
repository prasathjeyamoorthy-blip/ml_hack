# Requirements Document

## Introduction

This document specifies the requirements for an ML-based **Business Entity Resolution** system for the ML Challenge 2026. The system processes business records from three independent data sources — a deduplicated reference source (Source 1) and two noisy sources (Source 2 and Source 3) — and determines which records across sources refer to the same real-world business entity.

The evaluation metric is **F0.5 (macro-averaged)**, which weights precision at 2× over recall. The training set contains approximately 2.2 million Source 1 entities; the test set contains approximately 1.7 million Source 1 entities, including entities from France which does not appear in training. The system must produce two output files: `matching_results.tsv` (scored) and `candidate_pairs.tsv` (pipeline audit), both in strictly defined TSV formats.

Stage 1 (current) is documentation and architecture only. No implementation code is produced in Stage 1.

**EDA Status:** Full dataset analysis completed. Key findings incorporated into requirements. See `docs/02_data_analysis_plan.md` for the complete analysis.

---

## Glossary

- **Entity_Resolution_System**: The end-to-end ML pipeline that resolves business entities across sources.
- **Source1**: The deduplicated reference source file (`*_source1.tsv`). Each record is the canonical representation of one real-world business.
- **Source2**: A noisy business records file (`*_source2.tsv`). May contain zero, one, or many records matching a given Source1 entity.
- **Source3**: A noisy business records file (`*_source3.tsv`). May contain zero, one, or many records matching a given Source1 entity.
- **Ground_Truth**: The training label file (`train_ground_truth.tsv`) mapping each Source1 entity_id to comma-separated matched entity_ids from Source2 and Source3.
- **Blocker**: The component that reduces the full comparison space by generating a candidate set of plausible matching pairs for each Source1 entity.
- **Classifier**: The ML model that scores each candidate pair and decides whether the pair is a true match.
- **Candidate_Set**: The set of Source2 and Source3 records that the Blocker selects as plausible matches for a given Source1 entity. This is the exact input fed to the Classifier.
- **Singleton**: A Source1 entity for which the ground truth contains no matching Source2 or Source3 records.
- **Normalizer**: The text preprocessing component that standardizes business names and addresses before feature extraction.
- **Feature_Extractor**: The component that computes pairwise similarity features from a candidate pair.
- **Threshold_Optimizer**: The component that selects the classification threshold by optimizing F0.5 on a held-out validation set.
- **Submission_Validator**: The provided utility `utils/validate_submission.py` that checks output format correctness before submission.
- **F0.5**: The evaluation metric defined as `(1.25 × Precision × Recall) / (0.25 × Precision + Recall)`, computed per Source1 entity and macro-averaged across all Source1 entities.
- **Recall_Ceiling**: The maximum achievable recall given the Candidate_Set — the fraction of true matches that appear among the Blocker's candidates.
- **Reduction_Ratio**: The fraction of all possible Source1 × (Source2 ∪ Source3) pairs eliminated by the Blocker without scoring by the Classifier.
- **Open_Set_Country**: The country field is treated as an unbounded string label. France appears only in the test set and must not be filtered or cause pipeline failures.
- **TSV**: Tab-separated values file. The separator is a literal tab character (`\t`). All input files and both output files use this format.
- **matching_results.tsv**: The scored output file containing one row per Source1 entity with comma-separated matched entity_ids.
- **candidate_pairs.tsv**: The audit output file containing one row per Source1 entity with comma-separated candidate entity_ids that the Classifier ran inference over.
- **EDA**: Exploratory Data Analysis — statistical examination of the training data to inform design decisions.
- **DBA**: Doing Business As — a trade name different from the registered legal name of a business.

---

## Requirements

---

### Requirement 1: Data Ingestion

**User Story:** As a pipeline engineer, I want to load all source TSV files reliably, so that downstream components receive well-structured DataFrames with no silent data loss.

#### Acceptance Criteria

1. THE Entity_Resolution_System SHALL read all TSV files using the tab character (`\t`) as the separator and UTF-8 encoding.
2. IF a TSV file is opened with a parser that does not have `sep="\t"` configured, THEN THE Entity_Resolution_System SHALL raise an exception before any data is processed, rather than silently producing a single-column DataFrame.
3. THE Entity_Resolution_System SHALL load all four columns (`entity_id`, `business_name`, `business_address`, `country`) for Source1, Source2, and Source3 files. IF any of these four columns is absent from the file header, THEN THE Entity_Resolution_System SHALL raise an exception identifying the missing column name and the file path.
4. THE Entity_Resolution_System SHALL preserve leading/trailing whitespace in field values as read from the file until the Normalizer processes them.
5. THE Normalizer SHALL strip leading and trailing whitespace from all field values as its first normalization step.
6. IF a source file is missing or unreadable, THEN THE Entity_Resolution_System SHALL raise an exception with an error message that includes the full file path of the missing or unreadable file.
7. THE Entity_Resolution_System SHALL treat the `country` field as an open-set string label and SHALL NOT filter, drop, or one-hot encode any country value.

---

### Requirement 2: Data Validation

**User Story:** As a pipeline engineer, I want to validate the structure and content of all input files before processing, so that I catch data quality issues early rather than propagating corrupt inputs through the pipeline.

#### Acceptance Criteria

1. WHEN training data is loaded, THE Entity_Resolution_System SHALL verify that `train_ground_truth.tsv` contains exactly the same set of Source1 entity_ids as `train_source1.tsv`; IF the sets differ, THEN THE Entity_Resolution_System SHALL halt with an error message listing the count and a sample of mismatched entity_ids.
2. WHEN training data is loaded, THE Entity_Resolution_System SHALL verify that all entity_ids in `train_ground_truth.tsv`'s `matched_entity_ids` column reference entity_ids that exist in `train_source2.tsv` or `train_source3.tsv`; IF any dangling references are found, THEN THE Entity_Resolution_System SHALL halt with an error message listing the count of dangling references and up to 10 example values.
3. IF any entity_id prefix in a source file is not the expected prefix for that file (`S1-` for Source1, `S2-` for Source2, `S3-` for Source3), THEN THE Entity_Resolution_System SHALL log a data quality warning that includes the inconsistent entity_id and the file path; the pipeline SHALL continue processing after logging the warning.
4. WHEN training data is loaded, THE Entity_Resolution_System SHALL write the following summary statistics to standard output for each source file: row count, null rate per column (as a percentage), count of unique country values, and proportion of empty `business_address` values.
5. IF any validation check in criteria 1–3 fails or the null rate for any column exceeds 0% in any source file, THEN THE Entity_Resolution_System SHALL record the finding in `docs/open_questions.md` with the source file name, the check that failed, and the observed value.

---

### Requirement 3: Exploratory Data Analysis

**User Story:** As a data scientist, I want a structured EDA pass over the training data, so that I can make evidence-based decisions about normalization, blocking, and feature engineering before committing to implementation.

#### Acceptance Criteria

1. THE Entity_Resolution_System's EDA component SHALL compute and document: row counts per source, null rates per column, country distribution per source, average and maximum whitespace-delimited token count for `business_name` and `business_address` fields, and character script distribution using Unicode block ranges (bucketed as: ASCII, Latin Extended, Devanagari, Arabic, CJK, Other).
2. THE Entity_Resolution_System's EDA component SHALL compute match cardinality statistics from the ground truth: the full distribution (histogram) of the number of Source2+Source3 matches per Source1 entity, the proportion of Singletons (zero matches), and the proportion of entities with three or more matches.
3. THE Entity_Resolution_System's EDA component SHALL sample and document at least 5 concrete examples per noise category for business names (abbreviations, legal suffix variants, DBA names, typos) and at least 5 concrete examples per noise category for addresses (component reordering, missing PIN codes, landmark references), for a minimum of 5 samples per category.
4. WHEN EDA is complete, THE Entity_Resolution_System's EDA component SHALL output its findings to `docs/02_data_analysis_plan.md`; each finding SHALL be prefixed with either `**Observation:**` (for data-derived facts) or `**Assumption:**` (for inferred or unverified claims), so that the two categories are structurally distinguishable.
5. THE Entity_Resolution_System's EDA component SHALL NOT make claims about normalization strategy, blocking key selection, or model architecture choices without data evidence supporting that specific claim; any such unresolved claim SHALL be recorded in `docs/open_questions.md` rather than embedded in `docs/02_data_analysis_plan.md` as an observation.

---

### Requirement 4: Text Normalization

**User Story:** As a feature engineer, I want consistent, normalized representations of business names and addresses, so that surface-level variations do not prevent true matches from being detected.

#### Acceptance Criteria

1. THE Normalizer SHALL apply transformations in the following fixed order: (1) convert to lowercase, (2) strip leading and trailing whitespace, (3) normalize punctuation, (4) expand legal suffix abbreviations. This order SHALL be documented and applied consistently for both training and inference.
2. THE Normalizer SHALL expand a defined set of legal suffix abbreviations to canonical forms (e.g., "corp" → "corporation", "pvt" → "private", "ltd" → "limited", "llp" → "limited liability partnership") using whole-token matching only, so that a token like "corporate" is never expanded. The dictionary SHALL include French legal suffixes (SARL, SAS, SASU, EURL, SCI) because France represents 14.98% of the test set and these suffixes do not appear in training data.
3. THE Normalizer SHALL normalize punctuation in business names by: replacing `&` with `and`, replacing hyphens with a single space, and removing all characters that are neither alphanumeric nor whitespace.
4. THE Normalizer SHALL normalize address components by: standardizing street type abbreviations (e.g., "rd" → "road", "st" → "street", "ave" → "avenue") using whole-token matching, and removing all characters that are neither alphanumeric nor whitespace.
5. WHERE a business name contains non-ASCII scripts (e.g., Devanagari for Hindi), THE Normalizer SHALL preserve the original script characters rather than transliterating or dropping them, so that script-based blocking keys remain applicable.
6. THE Normalizer SHALL maintain a complete, version-controlled abbreviation expansion dictionary covering all legal suffix and street type expansions; this dictionary SHALL be the single authoritative source used at runtime.
7. IF a `business_name` or `business_address` field is null or empty, THEN THE Normalizer SHALL replace it with an empty string and SHALL NOT raise an exception.
8. THE Normalizer SHALL treat the `country` field as a pass-through string: after stripping leading and trailing whitespace, THE Normalizer SHALL return the `country` value unchanged.

---

### Requirement 5: Blocking Strategy

**User Story:** As a pipeline engineer, I want an efficient blocking stage that generates a compact Candidate_Set while retaining the vast majority of true matches, so that the Classifier only scores a feasible number of pairs without missing true positives.

#### Acceptance Criteria

1. THE Blocker SHALL produce a Candidate_Set for every Source1 entity such that the Recall_Ceiling on the training validation split is ≥ 0.90.
2. THE Blocker SHALL produce a Candidate_Set such that the total number of candidate pairs across all Source1 entities does not exceed 0.1% of the full Cartesian product of Source1 × (Source2 ∪ Source3) entity pairs on the training validation split.
3. THE Blocker SHALL implement at least two blocking keys where each key operates on a distinct field or distinct transformation of a field (e.g., name token key vs. address token key, or name token key vs. phonetic key), so that a noise pattern defeating one key does not defeat both.
4. THE Blocker SHALL cap the Candidate_Set size per Source1 entity at a maximum of 500 candidates; IF the union of all blocking key results for a Source1 entity exceeds 500 candidates, THEN THE Blocker SHALL apply a scoring-based pruning step (e.g., TF-IDF cosine ranking) to retain the top 500 candidates.
5. WHEN generating candidates for a Source1 entity with any country value, THE Blocker SHALL apply the same blocking logic without branching on the country value.
6. THE Blocker SHALL separately generate candidates from Source2 and from Source3, then union the two Candidate_Sets per Source1 entity before applying the 500-candidate cap.
7. THE Blocker SHALL record the Recall_Ceiling and total candidate pair count measured on the training validation split in `docs/04_blocking_strategy.md`. IF the Recall_Ceiling falls below 0.90, THEN THE Blocker SHALL add at least one additional blocking key before proceeding to Classifier training.
8. THE Blocker SHALL produce `candidate_pairs.tsv` containing the final Candidate_Set for every Source1 entity in the format defined in Requirement 14.

---

### Requirement 6: Candidate Generation Keys

**User Story:** As a data scientist, I want a multi-key blocking design that is robust to the noise patterns documented in EDA, so that name abbreviations, address reordering, and multilingual text do not systematically exclude true matches from the Candidate_Set.

#### Acceptance Criteria

1. THE Blocker SHALL implement a name-token blocking key by extracting tokens of 2 or more characters from the normalized business name after removing stop words and legal suffix tokens, and indexing records that share at least one such token. EDA confirms this achieves 84.76% recall alone on a 5,000 entity sample.
2. THE Blocker SHALL implement a phonetic blocking key using a phonetic encoding algorithm (e.g., Soundex, Metaphone, or Double Metaphone) applied to the first Latin-script significant token of the normalized business name; IF the first significant token is non-Latin-script, THEN the phonetic key SHALL be skipped for that entity and the name-token key SHALL be used as the sole name-based key.
3. THE Blocker SHALL implement an address-token blocking key by extracting tokens from the normalized business address that satisfy at least one of: (a) the token is a numeric string after stripping leading zeros (e.g., "00701" → "701"), or (b) the token is 3 or more characters and is not a stop word or common street-type token. This key is critical: EDA confirmed it recovers 97.4% of transliterated (non-ASCII) true matches missed by name-token blocking.
4. WHERE a Source1 entity has business name tokens containing non-ASCII characters (Unicode code points above U+007F), THE Blocker SHALL apply a character n-gram blocking key (n=3) hashed to a fixed-size bucket in addition to the Latin-script keys.
5. THE Blocker SHALL implement a domain-name blocking key: IF a Source2 or Source3 business name matches the regex pattern `\S+\.(com|net|org|in|co\.in|io|biz)`, THEN THE Blocker SHALL strip the TLD and any leading `@` or `#` characters, split on dots and hyphens, and index those tokens as additional name tokens. EDA confirms 4.0% of Source2/S3 records are domain names and represent 31.1% of name-token blocking misses.
6. THE Blocker SHALL strip leading `@` and `#` characters from Source2/S3 business names before applying all blocking keys, so that social handle variants (0.67% of Source2/S3 records) are recovered by the name-token key.
7. THE Blocker SHALL use a union of all applicable blocking keys for each entity pair: a candidate pair qualifies if it matches on at least one blocking key.
8. THE Blocker SHALL document the rationale for each blocking key and its expected noise coverage in `docs/04_blocking_strategy.md`.
9. IF all blocking keys for a Source1 entity yield zero candidates from both Source2 and Source3, THEN THE Blocker SHALL record that entity's entity_id in `docs/04_blocking_strategy.md` as a zero-candidate entity and SHALL still produce an output row for that entity in `candidate_pairs.tsv` with an empty `candidate_entity_ids` field.

---

### Requirement 7: Pairwise Feature Engineering

**User Story:** As a data scientist, I want a rich set of pairwise similarity features between a Source1 entity and a candidate, so that the Classifier can distinguish true matches from near-misses with high precision.

#### Acceptance Criteria

1. THE Feature_Extractor SHALL normalize both records in a candidate pair before computing any features by: converting to lowercase, removing characters that are neither alphanumeric nor whitespace, and collapsing consecutive whitespace to a single space.
2. THE Feature_Extractor SHALL compute Jaccard similarity on the whitespace-delimited token sets of normalized business names using the formula |A ∩ B| / |A ∪ B|, returning 0.0 when both token sets are empty.
3. THE Feature_Extractor SHALL compute normalized Levenshtein similarity on normalized business names using the formula 1 − (edit_distance(a, b) / max(len(a), len(b))), returning 1.0 when both strings are empty and 0.0 when exactly one string is empty.
4. THE Feature_Extractor SHALL compute TF-IDF cosine similarity on business names using a TF-IDF vectorizer whose vocabulary and IDF weights are fit exclusively on the `business_name` fields of the training corpus Source1, Source2, and Source3 files.
5. THE Feature_Extractor SHALL compute token overlap ratio for business addresses using the formula |A ∩ B| / max(|A|, |B|), where A and B are the whitespace-delimited token sets of the normalized addresses, returning 0.0 when both token sets are empty.
6. THE Feature_Extractor SHALL compute normalized Levenshtein similarity on business addresses using the same formula as criterion 3.
7. THE Feature_Extractor SHALL compute a country match binary feature using case-insensitive string equality after applying the same whitespace-stripping normalization as criterion 1: 1 if both records share the same normalized country string, 0 otherwise.
8. THE Feature_Extractor SHALL compute a numeric address token overlap ratio by extracting tokens matching the regex `\d+` from each normalized address and computing |A ∩ B| / max(|A|, |B|), returning 0.0 when both numeric token sets are empty.
9. THE Feature_Extractor SHALL record all feature definitions, including the exact formulas specified in criteria 2–8 and the normalization steps in criterion 1, in `docs/05_feature_engineering.md`.
10. THE Feature_Extractor SHALL NOT use any external lookup, geocoding API, or external database to derive features.
11. IF a field is empty in either record of a candidate pair, THEN THE Feature_Extractor SHALL return exactly 0.0 for all similarity features involving that field rather than raising an exception.

---

### Requirement 8: ML Classifier

**User Story:** As a data scientist, I want an ML classifier trained to distinguish true entity matches from false candidates, so that the system achieves high precision on the F0.5 metric.

#### Acceptance Criteria

1. THE Classifier SHALL be trained exclusively on the provided challenge training data; THE Classifier MAY use general-purpose pretrained models (e.g., trained on Wikipedia or Common Crawl) but SHALL NOT use pretrained models trained on datasets specifically curated for business entity identification, disambiguation, or linking tasks, and SHALL NOT use external entity resolution APIs.
2. THE Classifier SHALL use only models with MIT or Apache 2.0 licenses and at most 8 billion parameters.
3. THE Classifier SHALL be trained on positive examples derived from `train_ground_truth.tsv` and negative examples sampled from the Candidate_Set pairs that are not in the ground truth; the negative sampling ratio relative to positive examples SHALL be documented in `docs/06_model_strategy.md`.
4. WHEN the training set is class-imbalanced, THE Classifier SHALL apply one of the following strategies: class weighting (setting class weights inversely proportional to class frequency), oversampling of the minority class, or undersampling of the majority class; THE chosen strategy and its rationale SHALL be documented in `docs/06_model_strategy.md`.
5. THE Classifier SHALL output a continuous confidence score in [0, 1] for each candidate pair, representing the probability that the pair is a true match.
6. THE Classifier SHALL be evaluated exclusively on a held-out validation split comprising at least 10% of Source1 entities, stratified at the Source1-entity level so that all candidate pairs for a given Source1 entity are in either training or validation but not both; THE Classifier SHALL NOT use the test set during model selection or hyperparameter tuning.
7. Before Classifier training begins, THE Classifier's planned architecture and hyperparameter search space SHALL be recorded in `docs/06_model_strategy.md`. After training, THE final selected configuration (architecture, hyperparameters, training duration, and validation F0.5) SHALL be appended to `docs/06_model_strategy.md`.

---

### Requirement 9: Validation Methodology

**User Story:** As a data scientist, I want a rigorous internal validation framework that accurately estimates test-set performance, so that I avoid overfitting to the training set and can make reliable decisions about model selection and threshold tuning.

#### Acceptance Criteria

1. THE Entity_Resolution_System SHALL hold out a validation split comprising 15–20% of Source1 entities, sampled with a fixed random seed recorded in `docs/07_validation_strategy.md`, before any model training or hyperparameter search begins.
2. THE Entity_Resolution_System SHALL NOT use the validation split to generate Blocker training data, fit TF-IDF vocabularies, or compute any statistics used as training features.
3. THE Entity_Resolution_System SHALL compute F0.5 on the validation split using the formula `(1.25 × Precision × Recall) / (0.25 × Precision + Recall)`, computed per Source1 entity and then macro-averaged across all Source1 entities in the validation split, including Singletons.
4. WHEN evaluating Singleton entities in validation: a Source1 entity with no true matches that is predicted to have no matches SHALL contribute a per-entity F0.5 of 1.0 to the macro-average; a Source1 entity with true matches that is predicted to have no matches SHALL contribute a per-entity F0.5 of 0.0 to the macro-average.
5. THE Entity_Resolution_System SHALL write precision, recall, and macro-averaged F0.5 (as three named scalar values) plus the validation split size (in number of Source1 entities) to `docs/07_validation_strategy.md` after each evaluation run, so that the precision–recall trade-off is visible across runs.
6. THE Entity_Resolution_System's validation framework SHALL be documented in `docs/07_validation_strategy.md`, including the train/validation split strategy, the fixed random seed, and the exact F0.5 computation formula.

---

### Requirement 10: Threshold Optimization

**User Story:** As a data scientist, I want a principled method for selecting the classification threshold, so that the chosen operating point maximizes F0.5 on the validation set and reflects the precision-heavy nature of the metric.

#### Acceptance Criteria

1. THE Threshold_Optimizer SHALL sweep classifier output thresholds over the range [0.0, 1.0] at a resolution of at most 0.01 steps.
2. THE Threshold_Optimizer SHALL select the threshold that achieves the highest macro-averaged F0.5 on the validation set; IF two thresholds produce equal F0.5, THE Threshold_Optimizer SHALL select the higher threshold value to favour precision.
3. WHEN applying the threshold, THE Entity_Resolution_System SHALL declare a candidate pair a match if and only if the Classifier's confidence score for that pair is strictly greater than the selected threshold.
4. THE Threshold_Optimizer SHALL record the selected threshold value, the validation-set precision, recall, and F0.5 at that threshold in `docs/08_threshold_strategy.md`.
5. THE Threshold_Optimizer SHALL save a threshold-vs-F0.5 table (one row per threshold step) and a precision–recall table (one row per threshold step) as TSV files in the `docs/` folder, so that the sensitivity of the operating point is auditable.
6. IF per-country threshold calibration produces a macro-averaged F0.5 that exceeds the global-threshold F0.5 by more than 0.01 points on the validation set, THEN THE Threshold_Optimizer SHALL apply separate thresholds per country value; for Source1 entities with country values not seen during threshold calibration, THE Threshold_Optimizer SHALL apply the global threshold.

---

### Requirement 11: Singleton Handling

**User Story:** As a data scientist, I want the system to correctly identify and score Singleton entities, so that the precision gains from correct no-match predictions are captured in the F0.5 evaluation.

#### Acceptance Criteria

1. THE Entity_Resolution_System SHALL produce an output row in `matching_results.tsv` for every Source1 entity in the test set, including entities for which the Candidate_Set is empty.
2. WHEN the Candidate_Set for a Source1 entity is empty after blocking, THE Entity_Resolution_System SHALL write an empty `matched_entity_ids` field (an empty string, not "nan", not "None") for that entity.
3. WHEN the Classifier assigns a confidence score at or below the selected threshold for all candidates of a Source1 entity, THE Entity_Resolution_System SHALL write an empty `matched_entity_ids` field for that entity; this is consistent with the strict-greater-than match rule: a candidate is a match if and only if its score is strictly greater than the threshold.
4. A Source1 entity's `matched_entity_ids` field SHALL contain exactly the set of candidate entity_ids whose Classifier score is strictly greater than the selected threshold; no candidate SHALL be included solely because it exists in the Candidate_Set.
5. Singleton entities in the validation split SHALL each contribute a per-entity F0.5 value of 1.0 to the macro-average when the system correctly predicts no matches for them, and 0.0 when the system incorrectly predicts one or more matches for them.

---

### Requirement 12: Inference Pipeline

**User Story:** As a pipeline engineer, I want a deterministic, reproducible inference pipeline that processes the full test set end-to-end, so that anyone with the training data and code can regenerate the submission files.

#### Acceptance Criteria

1. THE Entity_Resolution_System SHALL process the full test set without requiring human intervention after a single command-line invocation.
2. THE Entity_Resolution_System SHALL execute the inference pipeline in the following fixed order: (1) data ingestion, (2) normalization, (3) blocking/candidate generation, (4) feature extraction, (5) classification scoring, (6) threshold application, (7) output file generation.
3. THE Entity_Resolution_System SHALL be deterministic: given the same trained model, the same input data, and the same fixed random seeds for all stochastic operations, the pipeline SHALL produce byte-for-byte identical output files on repeated runs, including identical row order and identical within-list entity_id ordering.
4. THE Entity_Resolution_System SHALL log the following observable metrics at each pipeline stage to standard output or a log file: (1) records loaded per source file; (2) records normalized; (3) total candidates generated; (4) features extracted (pair count); (5) pairs scored; (6) matches accepted (above threshold); (7) output rows written.
5. THE Entity_Resolution_System's inference pipeline architecture SHALL be documented in `docs/09_inference_pipeline.md` including: a data flow diagram, peak memory consumption per stage in gigabytes, and expected wall-clock runtime per stage in minutes.
6. IF an exception occurs during any pipeline stage, THEN THE Entity_Resolution_System SHALL halt immediately, log the stage name and exception details, and SHALL NOT write partial output files.

---

### Requirement 13: Output Format — matching_results.tsv

**User Story:** As a competition participant, I want the matching_results.tsv file to conform exactly to the challenge specification, so that the submission is not rejected by the validator.

#### Acceptance Criteria

1. THE Entity_Resolution_System SHALL write `output/matching_results.tsv` as a UTF-8 encoded, tab-separated file with the header row `source1_entity_id\tmatched_entity_ids`.
2. THE Entity_Resolution_System SHALL include exactly one row per Source1 entity from `test_source1.tsv`; no Source1 entity SHALL be omitted, no duplicate rows SHALL appear, and no entity_id absent from `test_source1.tsv` SHALL appear as a row key.
3. THE Entity_Resolution_System SHALL write `matched_entity_ids` as a comma-separated list of Source2 and Source3 entity_ids with no quoting, no trailing comma, and no whitespace characters adjacent to any entity_id or comma separator.
4. THE Entity_Resolution_System SHALL write an empty `matched_entity_ids` field (an empty string, not null, not "nan", not "None") for Source1 entities that have no predicted matches, producing a line of the form `S1-XXXXXXX\t`.
5. THE Entity_Resolution_System SHALL NOT include any Source1 entity_id in the `matched_entity_ids` column.
6. THE Entity_Resolution_System SHALL NOT include entity_ids that do not exist in `test_source2.tsv` or `test_source3.tsv` in the `matched_entity_ids` column.
7. THE Entity_Resolution_System SHALL NOT produce duplicate entity_ids within a single `matched_entity_ids` list.

---

### Requirement 14: Output Format — candidate_pairs.tsv

**User Story:** As a competition participant, I want the candidate_pairs.tsv file to correctly capture the final blocking stage output, so that pipeline auditors can verify recall ceiling and reduction ratio.

#### Acceptance Criteria

1. THE Entity_Resolution_System SHALL write `output/candidate_pairs.tsv` as a UTF-8 encoded, tab-separated file with the header row `source1_entity_id\tcandidate_entity_ids`.
2. THE Entity_Resolution_System SHALL include exactly one row per Source1 entity from `test_source1.tsv`; no Source1 entity SHALL be omitted; no duplicate rows SHALL appear; no entity_id absent from `test_source1.tsv` SHALL appear as a row key; `candidate_entity_ids` SHALL be written as a comma-separated list with no quoting, no trailing comma, and no whitespace characters adjacent to any entity_id or comma separator; for Source1 entities with an empty Candidate_Set, THE Entity_Resolution_System SHALL write an empty string (not "nan"), producing a line of the form `S1-XXXXXXX\t`; no entity_id SHALL appear more than once within a single `candidate_entity_ids` list.
3. THE Entity_Resolution_System SHALL ensure that every entity_id present in `matching_results.tsv`'s `matched_entity_ids` column for a given Source1 entity also appears in `candidate_pairs.tsv`'s `candidate_entity_ids` column for that same Source1 entity.
4. THE Entity_Resolution_System SHALL populate `candidate_pairs.tsv` from the final Candidate_Set — the exact set of Source2 and Source3 records the Classifier ran inference over — and SHALL NOT include earlier-stage candidates that were subsequently filtered before reaching the Classifier.
5. THE Entity_Resolution_System SHALL NOT include Source1 entity_ids or entity_ids outside `test_source2.tsv` and `test_source3.tsv` in the `candidate_entity_ids` column.

---

### Requirement 15: Submission Validation

**User Story:** As a competition participant, I want to run the challenge's provided validator before uploading, so that format errors are caught locally rather than wasting a scored submission.

#### Acceptance Criteria

1. THE Entity_Resolution_System SHALL run `utils/validate_submission.py` with the command `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` before any submission is made to the challenge portal.
2. WHEN `utils/validate_submission.py` exits with code 0, THE Entity_Resolution_System SHALL proceed to submission.
3. WHEN `utils/validate_submission.py` exits with a non-zero code or reports a FAIL issue, THE Entity_Resolution_System SHALL correct the identified issues and re-run the validator; THE Entity_Resolution_System SHALL NOT submit until the validator exits with code 0.
4. IF the validator still does not exit with code 0 after 3 correction attempts, THEN THE Entity_Resolution_System SHALL halt and record the unresolved validator errors in `docs/open_questions.md` before any submission attempt.
5. THE Entity_Resolution_System SHALL store the complete standard output of the final passing validator run as a file at `output/validation_pass.log` alongside the submission files for audit purposes.

---

### Requirement 16: Open-Set Country Handling

**User Story:** As a data scientist, I want the pipeline to handle unseen country values transparently, so that France (and any other future country) does not cause failures, dropped rows, or degraded precision.

#### Acceptance Criteria

1. THE Entity_Resolution_System SHALL process every Source1 entity in `test_source1.tsv` regardless of the `country` field value; no row SHALL be skipped or dropped due to an unrecognized country value.
2. THE Normalizer SHALL apply normalization rules conditionally based on the observed country value; for country values not seen during training, THE Normalizer SHALL apply only the universal normalization rules (lowercase, whitespace stripping, punctuation normalization, suffix expansion) and SHALL NOT apply country-specific rules designed only for `US` or `India`.
3. WHEN the Blocker processes a Source1 entity with any country value including unseen values (e.g., France), THE Blocker SHALL apply all blocking keys defined in Requirement 6; THE Blocker SHALL document in `docs/04_blocking_strategy.md` how address and name patterns from countries not represented in training data are handled by the defined keys.
4. THE Feature_Extractor SHALL compute the country match feature as a case-insensitive string equality comparison after whitespace stripping, which handles any country value without modification.
5. IF the `country` field of a Source1 entity was not present in the validation split during threshold calibration, THEN THE Threshold_Optimizer SHALL apply the global threshold for that entity.
6. IF the `country` field of a Source1 entity or candidate record is null or empty, THEN every pipeline component (Normalizer, Blocker, Feature_Extractor, Threshold_Optimizer) SHALL treat it as the string value `"unknown"` rather than raising an exception.

---

### Requirement 17: Scalability and Performance

**User Story:** As a pipeline engineer, I want the pipeline to complete on the full test set within a practical time frame, so that iteration cycles and final submission generation are not bottlenecked by compute.

#### Acceptance Criteria

1. THE Blocker SHALL generate candidates using an indexing structure (e.g., inverted index, LSH, or TF-IDF sparse retrieval) such that the total number of candidate pairs across all Source1 entities does not exceed 0.1% of the full Cartesian product of Source1 × (Source2 ∪ Source3) entity pairs.
2. THE Entity_Resolution_System SHALL complete the full inference pipeline (data ingestion through output file generation) for the test set within 4 hours wall-clock time on the documented reference hardware.
3. THE Entity_Resolution_System SHALL document estimated peak memory consumption (in GB) and expected wall-clock runtime (in minutes) for each pipeline stage in `docs/09_inference_pipeline.md`, based on observed data sizes.
4. THE Entity_Resolution_System SHALL support batch processing in the Feature_Extractor and Classifier stages with a configurable batch size in the range [128, 65536] records, so that the pipeline can run within a 32 GB RAM memory budget on a single machine.
5. THE Entity_Resolution_System SHALL document in `docs/12_development_plan.md` the minimum assumed CPU core count, assumed RAM in GB, and whether a GPU is required or optional for the pipeline to meet the 4-hour runtime bound.

---

### Requirement 18: Reproducibility and Submission Package

**User Story:** As a competition auditor, I want the submission package to be fully self-contained and reproducible, so that the top teams' results can be independently verified.

#### Acceptance Criteria

1. THE Entity_Resolution_System SHALL be packaged as `<team_name>_submission.zip` containing `output/`, `code/business_entity_resolution/`, and the completed `Documentation_template.md` with all required sections filled in.
2. THE Entity_Resolution_System's `code/business_entity_resolution/README.md` SHALL provide exact command-line instructions to regenerate both `matching_results.tsv` and `candidate_pairs.tsv`, including the required input data paths (`dataset/train/` and `dataset/test/`) and the full sequence of commands.
3. THE Entity_Resolution_System's `code/business_entity_resolution/requirements.txt` SHALL pin all Python dependency versions using the `==` operator and SHALL include a comment specifying the Python version (e.g., `# python==3.10`).
4. THE Entity_Resolution_System SHALL produce byte-for-byte identical `matching_results.tsv` and `candidate_pairs.tsv` output files when the documented run instructions are followed on a machine with the same OS, Python version, and hardware as specified in `docs/12_development_plan.md`, given identical random seeds.
5. THE Entity_Resolution_System SHALL record all architectural decisions in `docs/decisions.md` with rationale; THE Entity_Resolution_System SHALL record all unresolved questions in `docs/open_questions.md`.

---

### Requirement 19: Documentation and Architecture

**User Story:** As the system architect, I want a complete, structured documentation suite covering every stage of the pipeline, so that implementation by a separate engineering agent (Claude Code) can proceed without ambiguity.

#### Acceptance Criteria

1. THE Entity_Resolution_System's documentation SHALL include all eighteen documents: `docs/01_problem_understanding.md` through `docs/15_claude_code_protocol.md` (fifteen numbered documents), plus `docs/decisions.md`, `docs/open_questions.md`, and `README.md`; each document SHALL be non-empty and SHALL contain the section headings defined in the project brief.
2. WHEN a document presents a design decision that has not yet been validated by EDA or experiment, THE document SHALL prefix that statement with the inline tag `[ASSUMPTION]` immediately preceding the claim, so that assumptions are structurally distinguishable from verified facts.
3. THE Entity_Resolution_System's `docs/15_claude_code_protocol.md` SHALL define the collaboration protocol between Kiro (architecture owner) and Claude Code (implementation agent), including: which files Claude Code is authorized to create, which decisions require Kiro approval, and how findings are communicated via `docs/claude_code_notes.md` and `docs/decisions.md`.
4. WHEN Stage 1 documentation is complete, THE system architect SHALL record written approval in `docs/decisions.md` before any Stage 2 (Dataset Inspection) work begins. IF Stage 2 work begins without this approval record, THEN all Stage 2 artifacts SHALL be considered out of scope for the current review cycle.
5. THE Entity_Resolution_System's `docs/14_risk_and_failure_modes.md` SHALL enumerate at least three failure modes per pipeline stage, classify each failure mode's likelihood as Low, Medium, or High, and provide a mitigation strategy of at least one sentence per failure mode.

---

### Requirement 20: No External Data or APIs

**User Story:** As a challenge integrity officer, I want the system to strictly use only the provided challenge data, so that the submission is compliant with the fair-play rules and is not disqualified.

#### Acceptance Criteria

1. THE Entity_Resolution_System SHALL NOT query any external API, database, web service, or data source to augment, verify, or resolve business entities at any pipeline stage (training, validation, inference, or feature extraction).
2. THE Entity_Resolution_System SHALL NOT use geocoding APIs, government business registries, commercial entity resolution services, or any internet-sourced data at any pipeline stage; this prohibition does not apply to pretrained model weights governed by criterion 4.
3. THE Entity_Resolution_System SHALL derive all training signals from the following files only: `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, and `train_ground_truth.tsv` during training and validation; `test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv` during inference.
4. THE Entity_Resolution_System SHALL use only pretrained models with MIT or Apache 2.0 licenses and at most 8 billion parameters; fine-tuning on or use of pretrained models trained on datasets specifically curated for business entity identification, disambiguation, or linking tasks is prohibited.
5. WHEN a pretrained model is used (e.g., for embedding generation), THE Entity_Resolution_System SHALL record the model name, version or commit hash (with release date as fallback if no formal version string exists), license, and parameter count in `docs/06_model_strategy.md`.
