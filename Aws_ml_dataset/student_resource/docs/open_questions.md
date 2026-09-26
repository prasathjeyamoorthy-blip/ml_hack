# Open Questions

All unresolved questions. Answers must be filled in after experiments by Claude Code.

---

## Blocking

**Q1:** What is the combined blocking recall (name-token + address-token + domain-strip + social-handle-strip + country-filter) on the training validation split?
- Target: ≥ 90% recall ceiling
- Current evidence: name-token alone = 84.76% (simulation on 5k S1 sample)
- Status: **Unresolved — requires implementation**

**Q2:** How many Source1 entities end up with zero candidates after all blocking keys are applied?
- Concern: These will be predicted as Singletons — but most S1 entities (94.4%) have true matches
- Status: **Unresolved — requires implementation**

**Q3:** Does adding a TF-IDF retrieval key (top-k BM25 or cosine) on top of token blocking improve recall ceiling above 90% within the 500-candidate cap?
- Status: **Unresolved — requires experiment**

**Q4:** What is the effective reduction ratio at the 500-candidate cap?
- The 500 cap = 0.0048% of Cartesian, which satisfies the ≥0.01 reduction ratio requirement
- But does the cap cause true match losses for high-cardinality entities?
- Status: **Unresolved — requires implementation**

---

## Normalization

**Q5:** Which French address tokens ("Rue", "Boulevard", "Allée", "Impasse", "Cité") should be treated as street-type tokens and excluded from blocking keys (like "St", "Ave", "Rd")?
- Status: **Unresolved — requires manual curation or frequency analysis on France test data**

**Q6:** Does stripping domain TLDs and splitting on dots/hyphens recover domain-name variants reliably?
- e.g., "ghaziabadconstructions.com" → "ghaziabad constructions" → matches token blocking
- Status: **Unresolved — requires experiment**

---

## Features

**Q7:** Which features have the highest feature importance for the classifier?
- Hypothesis: name Jaccard, name TF-IDF cosine, numeric address overlap will rank highest
- Status: **Unresolved — requires feature ablation experiment**

**Q8:** Does Jaro-Winkler similarity add value beyond Levenshtein for short business names?
- Status: **Unresolved — requires ablation**

**Q9:** Does character n-gram similarity (e.g., n=3) on business names help with OCR/typo pairs missed by token overlap?
- Status: **Unresolved — requires experiment**

---

## Model

**Q10:** Which classifier performs best: LightGBM, XGBoost, HistGradientBoosting, or Logistic Regression?
- Status: **Unresolved — requires experiment**

**Q11:** What is the optimal negative sampling ratio (positive:negative)?
- Current evidence: positive rate in candidate set ≈ 0.69% (1:144 uncapped)
- Proposed starting points: 1:10, 1:20
- Status: **Unresolved — requires experiment**

**Q12:** Does hard negative mining (sampling near-miss false positives) improve precision?
- Status: **Unresolved — requires experiment**

---

## Threshold

**Q13:** What threshold value maximizes validation F0.5?
- Given precision-heavy F0.5, expected range: 0.6–0.8
- Status: **Unresolved — requires threshold sweep experiment**

**Q14:** Does per-country threshold calibration (separate thresholds for US, India, France) improve macro F0.5 by more than 0.01 points?
- Status: **Unresolved — requires experiment**

---

## France / Open-Set Country

**Q15:** Do France entities match using the same blocking keys as US/India? Or do French address patterns require a separate street-type token list?
- Status: **Unresolved — no France training ground truth available. Must be inferred from test data structure.**

**Q16:** What is the expected positive rate for France entities? Is it similar to US/India?
- Status: **Unresolved — no ground truth for France**

---

## Answers (to be filled in by Claude Code after experiments)

| Q# | Answer | Date | Experiment |
|----|--------|------|-----------|
| — | — | — | — |
