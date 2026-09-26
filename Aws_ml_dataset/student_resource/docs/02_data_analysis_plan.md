# Document 02: Data Analysis Plan — EDA Results

> Status: COMPLETED (actual dataset analysis performed)
> All findings below are derived from running analysis scripts on the real dataset.
> Findings are labeled **Observation** (data-derived) or **Assumption** (inferred/unverified).

---

## 1. Record Counts

**Observation:** Row counts across all files:

| File | Rows |
|------|------|
| train_source1.tsv | 2,206,821 |
| train_source2.tsv | 5,034,616 |
| train_source3.tsv | 5,285,603 |
| train_ground_truth.tsv | 2,206,821 |
| test_source1.tsv | 1,732,544 |
| test_source2.tsv | 4,887,273 |
| test_source3.tsv | 5,082,316 |

**Observation:** Ground truth row count matches Source1 exactly — every Source1 entity has exactly one ground truth row (possibly empty matched_entity_ids).

**Observation:** The combined Source2+Source3 pool is ~10.3M records for training and ~9.97M for test.

---

## 2. Null / Empty Rates

**Observation:** Null rates per column:

| Source | entity_id | business_name | business_address | country |
|--------|-----------|---------------|------------------|---------|
| Source1 (train) | 0% | 0% | 0% | 0% |
| Source2 (train) | 0% | ~0% (2 records) | **3.36% (168,967)** | 0% |
| Source3 (train) | 0% | ~0% (13 records) | **3.33% (175,916)** | 0% |
| test_source1 | 0% | 0% | 0% | 0% |
| test_source2 | 0% | — | **2.65% (129,408)** | 0% |
| test_source3 | 0% | — | **2.68% (136,098)** | 0% |

**Observation:** Source1 has zero missing addresses. Source2 and Source3 have ~3.3% empty addresses in training and ~2.65–2.68% in test. This is a significant fraction — ~344K training records have no address.

**Optimization implication:** Address-based blocking must gracefully fall back to name-only blocking when address is absent. Features must return 0.0 (not NaN/error) for empty addresses.

---

## 3. Country Distribution

**Observation:** Training data contains only 2 countries:

| Country | Source1 | Source2 | Source3 |
|---------|---------|---------|---------|
| US | 1,323,633 (59.98%) | 3,016,817 (59.92%) | 3,170,056 (59.97%) |
| India | 883,188 (40.02%) | 2,017,799 (40.08%) | 2,115,547 (40.02%) |

**Observation:** Test data introduces a third country:

| Country | test_source1 | test_source2 | test_source3 |
|---------|-------------|-------------|-------------|
| India | 809,986 (46.75%) | 2,312,565 (47.32%) | 2,405,000 (47.33%) |
| US | 663,106 (38.27%) | 1,871,330 (38.29%) | 1,945,701 (38.28%) |
| **France** | **259,452 (14.98%)** | **703,378 (14.39%)** | **731,615 (14.39%)** |

**Observation:** France represents ~15% of test_source1 — this is a substantial portion, not a rare edge case. France has its own legal suffixes (SARL, SAS, SASU, EURL, SCI) that are not present in training.

**Optimization implication:** Legal suffix normalization MUST include French suffixes: SARL → société à responsabilité limitée, SAS, SASU, EURL, SCI. These should be added to the normalization dictionary before training.

---

## 4. Match Cardinality

**Observation:** Match count distribution from ground truth:

| Matches | Count | Percentage |
|---------|-------|------------|
| 0 (Singleton) | 123,247 | **5.58%** |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.93% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10 | 534 | 0.02% |
| 11 | 37 | <0.01% |

**Observation:**
- Only **5.58% of Source1 entities are Singletons** (no matches) — this is much lower than expected. The vast majority (94.42%) have at least one match.
- **72.01% of entities have 3 or more matches** — the system must support robust multi-match output.
- **Mean matches per Source1 entity: 3.461**
- **Max matches: 11**
- Total true match pairs in training: **7,638,365**

**Optimization implication:** Since 94% of entities have matches, the classifier must produce high precision to avoid flooding. The low singleton rate also means the threshold must be tuned carefully — being too conservative (predicting no matches) will hurt significantly.

---

## 5. Source2 vs Source3 Match Distribution

**Observation:**

| Metric | Value |
|--------|-------|
| S1 entities with ≥1 S2 match | 1,919,076 (86.96%) |
| S1 entities with ≥1 S3 match | 1,940,545 (87.93%) |
| S1 entities with BOTH S2 and S3 | 1,776,047 (80.48%) |
| S1 entities with ONLY S2 match | 143,029 |
| S1 entities with ONLY S3 match | 164,498 |

**Observation:** S2 and S3 are nearly symmetric in coverage — both sources must be blocked independently.

---

## 6. Business Name Statistics

**Observation:** Token count statistics (whitespace-delimited):

| Source | Min | Mean | Median | Max |
|--------|-----|------|--------|-----|
| Source1 | 1 | 3.55 | 4 | 16 |
| Source2 | 1 | 3.50 | 4 | 15 |
| Source3 | 1 | 3.53 | 4 | 18 |

**Observation:** Source1 business names are 100% ASCII. Source2 has 15.19% non-ASCII records; Source3 has 11.48% non-ASCII records.

**Observation:** Devanagari-script names in Source2: **269,424 records** (primarily India entities). Telugu and other Indic scripts also observed.

**Optimization implication:** Source1 is pure ASCII but Source2/S3 contain Devanagari transliterations of Source1 names. Address blocking is the primary recovery mechanism for these (97.4% recovery rate confirmed, see Section 10).

---

## 7. Address Statistics

**Observation:** Token count statistics (whitespace-delimited):

| Source | Min | Mean | Median | Max | Empty |
|--------|-----|------|--------|-----|-------|
| Source1 | 2 | 8.03 | 7 | 43 | 0 |
| Source2 | 0 | 7.29 | 6 | 46 | 168,967 |
| Source3 | 0 | 7.17 | 6 | 43 | 175,916 |

---

## 8. Duplicate Analysis

**Observation:**

| Source | Duplicate entity_ids | Duplicate exact names | Duplicate normalized names |
|--------|---------------------|----------------------|---------------------------|
| Source1 | 0 | 667,592 (30.25%) | 668,017 (30.27%) |
| Source2 | 0 | 632,607 (12.57%) | 747,146 (14.84%) |
| Source3 | 0 | 633,994 (11.99%) | 706,520 (13.37%) |

**Observation:** No duplicate entity_ids in any source — IDs are unique. However, **30.25% of Source1 records share a business name with another Source1 record** (they are different locations/entities with the same name). This is a critical blocker design concern — name-only blocking will generate many false candidates for common names.

**Optimization implication:** Country + name token combination blocking is essential to limit cross-country false candidates. Address tokens must be weighted heavily for disambiguation.

---

## 9. Legal Suffix Presence

**Observation:** Records containing at least one legal suffix token (approx):
- Source1: ~87.8%
- Source2: ~57.4%
- Source3: ~62.0%

**Observation:** Source2 and Source3 frequently STRIP the legal suffix entirely (e.g., "Celentano Bioscience" vs "Celentano Bioscience LLC"). This is a common noise pattern.

**Optimization implication:** Legal suffix tokens should be removed from blocking keys (confirmed in spec). They are poor discriminators and cause false negatives if required.

---

## 10. Noise Pattern Catalogue (from matched pair analysis)

### 10.1 Business Name Noise Patterns (observed in true matches)

| Pattern | Example | Frequency |
|---------|---------|-----------|
| Legal suffix dropped | "Celentano Bioscience" vs "Celentano Bioscience LLC" | Very common |
| Legal suffix abbreviated | "Pvt Ltd" vs "Private Limited" | Very common |
| ALL CAPS variant | "PRIMARY CARE DESERT MEDICINE LLC" vs original | Very common |
| Punctuation in suffix | "L.L.C." vs "LLC" | Common |
| Word order change | "Jaipur Ltd. Pvt. Consultancy" vs "Jaipur Consultancy Pvt. Ltd." | Common |
| Noise prefix | "--", "@", "Smt", "#" prepended | Common (4.3% social handles in S2/S3) |
| Domain name | "ghaziabadconstructions.com" vs business name | **4.0% of S2/S3 records** |
| Transliteration | "राम हॉस्पिटैलिटी लिमिटेड" vs "Ram Hospitality Limited" | Significant for India |
| OCR/typo | "Prvifte", "gi1lette", "Eletci", "Ássociates" | Present |
| Accent diacritics | "Ássociates" vs "Associates", "Córp" vs "Corp" | Present |
| Number words | "Twelfth" vs "12th", "Ninth" vs "9th" | Present |
| Extra/duplicate tokens | "Harbor Dental Dental [Associates]" vs original | Present |
| DBA short name | "Gillette" vs "Gillette Electric Inc" | Present |

### 10.2 Address Noise Patterns (observed in true matches)

| Pattern | Example |
|---------|---------|
| Street type abbreviation | "DR" vs "Drive", "ST" vs "Street", "AVE" vs "Avenue" |
| Component reordering | "IN, 613 9TH ST, WEST TERRE HAUTE" vs "613 9th St, West Terre Haute, IN" |
| Missing street number | Address starts with street name |
| Zero-padded numbers | "00701" vs "701", "03851" vs "3851" |
| Number words | "Twelfth Ave" vs "12th Ave" |
| Mixed script | "महाराष्ट्र" vs "Maharashtra" in same address field |
| Special char prefix | "##90" vs "90", "1124." vs "1124" |
| Missing address (empty) | ~3.3% of S2/S3 records |
| Landmark references | "H.NO C-16 A", "Plot ##311" variations |

---

## 11. Blocking Recall Simulation

**Observation:** Pure name-token blocking (≥2 char tokens, stop words and legal suffixes removed) achieves **84.76% recall** on a 5,000 entity sample (18,331 true pairs tested).

**Observation:** The 15.24% missed pairs break down as:
- **Domain name variants**: 31.1% of misses (e.g., "rabunboral.com" vs "Rabun Boral LLC")
- **Non-ASCII / transliteration**: **48.3% of misses** (Devanagari, Telugu, etc.)
- **Social handles (@/#)**: 4.3% of misses
- **Other (typo/OCR/fully different name)**: ~16%

**Observation:** For transliterated pairs (non-ASCII S2/S3 vs ASCII S1), address token blocking recovers **97.4% of them**. This is a critical finding.

**Observation:** Domain name variants (e.g., "familyempirepartners.com") require a specialized blocking key — strip TLD and dots, then match on concatenated tokens.

---

## 12. Scale Implications

**Observation:**

| Metric | Value |
|--------|-------|
| Full Cartesian product (train) | ~22.77 trillion pairs |
| 0.1% of Cartesian | ~22.77 billion pairs |
| Candidate pairs at cap 500 per S1 | ~1.1 billion pairs |
| Cap 500 as % of Cartesian | 0.0048% |
| Total true match pairs | 7,638,365 |
| Positive rate in full Cartesian | 0.000034% |
| Positive rate at cap-500 candidate set | ~0.69% (extreme class imbalance) |

**Observation:** Even at 500 candidates per S1 entity, the positive rate is only ~0.69%. The classifier must handle severe class imbalance. Negative sampling strategy is critical.

---

## 13. France-Specific Observations

**Observation:** France records use distinct legal suffixes absent from training:
- SARL (Société à Responsabilité Limitée — equivalent to LLC)
- SAS / SASU (Société par Actions Simplifiée)
- EURL (Entreprise Unipersonnelle à Responsabilité Limitée)
- SCI (Société Civile Immobilière)

**Observation:** France addresses include French-language components ("Rue", "Boulevard", "Avenue", "Allée", region names "Hauts-de-France", "Nouvelle-Aquitaine").

**Observation:** French names contain accent characters (é, è, à, â, etc.) — not Devanagari, but non-ASCII Latin Extended.

**Optimization implication:** The normalization dictionary must be extended with French legal suffixes before training even though they won't appear in ground truth. The address blocking should treat French address tokens ("Rue", "Boulevard", "Allée") like English street type tokens — include them in address-token blocking but down-weight them relative to rare city/district names.

---

## 14. Key Optimizations Identified from EDA

The following architectural optimizations are directly supported by data evidence:

### OPT-1: Address blocking is the primary fallback for transliterated names
**Evidence:** 97.4% of transliterated (non-ASCII) true matches share at least one address token with their S1 counterpart.
**Action:** Address token blocking is not optional — it is the critical recovery mechanism for India entities.

### OPT-2: Domain name stripping blocking key is needed
**Evidence:** 4.0% of Source2/S3 records are domain names (31.1% of token-blocking misses). These cannot be recovered by any token or phonetic key.
**Action:** Add a domain name normalization step: strip TLD (`.com`, `.in`, `.net`, etc.), remove hyphens, match on resulting concatenated string. Use this as an additional blocking key.

### OPT-3: Social handle stripping
**Evidence:** 0.67% of S2/S3 records start with `@` or `#`.
**Action:** Strip leading `@` and `#` characters before tokenization. This enables token blocking to recover these pairs.

### OPT-4: France legal suffixes must be in normalization dictionary
**Evidence:** France = 15% of test set. SARL/SAS/SASU/EURL/SCI are common and not in training data.
**Action:** Add French legal suffixes to the expansion dictionary before training. Even though they won't appear in training ground truth, they will affect blocking and feature extraction at test time.

### OPT-5: Zero-padded number normalization for address blocking
**Evidence:** "00701" vs "701", "03851" vs "3851" observed in matched pairs.
**Action:** Strip leading zeros from numeric tokens in address blocking keys. e.g., `int("00701") → "701"`.

### OPT-6: Country-stratified blocking
**Evidence:** 30.25% of Source1 names are duplicated. Many businesses (e.g., "Allied Federation") have the same name across different locations.
**Action:** Apply country as a blocking filter — only generate candidates where S1 and S2/S3 share the same country string. Exception: if one or both country fields are empty.

### OPT-7: Negative sampling ratio must be carefully controlled
**Evidence:** Positive rate at cap-500 is only 0.69% (~1:144 ratio). Training with all negatives would overwhelm the classifier.
**Action:** Undersample or weight negatives. Recommended starting point: 1:10 or 1:20 positive:negative ratio with class weighting.

### OPT-8: Threshold will be well above 0.5 (precision-heavy metric)
**Evidence:** Only 5.58% of entities are true singletons, but with 0.69% positive rate in candidates, the classifier will produce many low-confidence scores on true negatives. Given F0.5 is precision-heavy, the optimal threshold is expected to be in the 0.6–0.8 range.
**Action:** Sweep thresholds from 0.3 to 0.95 with 0.01 resolution on the validation set.

---

## 15. Open Questions Generated from EDA

See `docs/open_questions.md` for the full list. Key unresolved items:

- Q1: What is the actual blocking recall achievable with name-token + address-token + domain-strip + country-filter combined? Target ≥ 90%. **(Requires implementation)**
- Q2: What is the optimal negative sampling ratio? **(Requires experiment)**
- Q3: Does per-country threshold calibration improve F0.5 for France entities? **(Requires experiment)**
- Q4: How many S1 entities will have zero candidates after blocking (true hard singletons vs. blocking failures)? **(Requires implementation)**
- Q5: What is the effective candidate recall ceiling achievable within the 500-candidate cap? **(Requires implementation)**
