# TRUST-RAG: Final Comprehensive Experiment Audit, Feature Correction, and Verification Report

**Audit Date:** October 12, 2026  
**Auditor Role:** Senior ML Research Engineer and Research-Paper Auditor  
**Target Repository:** `TRUST-RAG`  
**Target Manuscripts:** [`paper_manuscript_final.md`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/paper_manuscript_final.md) & [`paper_manuscript_final.tex`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/paper_manuscript_final.tex)  
**Report File:** `reports/final_experiment_audit.md`  

---

## 1. Executive Summary & Audit Verdict

### Overall Verdict: **VERIFIED & PUBLICATION-READY (ALL CHECKS PASSED)**

A rigorous, end-to-end experimental and methodological audit of the TRUST-RAG repository was completed. All identified defects—specifically the constant `n_unique_pmids` retrieval feature, cross-validation preprocessing data leakage, ablation data fallback, and numerical discrepancies in the manuscript—have been diagnosed at the root-cause level, repaired in source code, verified with unit tests, regenerated across all affected downstream splits, and synchronized across all machine-readable files, figures, tables, and IEEE manuscript text.

### Key Milestones Achieved
1. **Diagnosis & Repair of `n_unique_pmids`:** Identified that generator evidence passage slicing ($K=3$) was conflated with the candidate retrieval pool ($K=5$). The feature was constantly evaluating to $3/5.0 = 0.6000$. Created `retrieve_with_candidates()` in `src/retriever.py`, updated `src/trust_features.py`, added a 6-case unit test suite (`tests/test_trust_features.py`, 100% passing), and regenerated train ($N=600$) and val ($N=200$) features. The feature now exhibits true discrete distribution ranging from $0.20$ to $1.00$ ($\mu = 0.6557, \sigma = 0.1580$).
2. **Elimination of Cross-Validation Preprocessing Leakage:** Audited `src/trust_model.py`. Replaced manual whole-partition feature scaling with a scikit-learn `Pipeline([('scaler', StandardScaler()), ('classifier', base_model)])` inside cross-validation, guaranteeing strict fold-isolated scaling. Verified that all models fit only on the designated 450 fitting samples, calibrate only on the designated 150 calibration samples, and evaluate on the untouched 200 validation holdout.
3. **Rigorous Explanation of the Semantic-Only Calibration Anomaly:** Demonstrated mathematically and empirically why the semantic-only ablation produced AUROC = 0.3098. On the 150-sample calibration subsplit, the empirical correlation between the uncalibrated semantic logit and binary label was slightly negative ($-0.0060, p = 0.9418$). Scikit-learn's unconstrained sigmoid calibration fitted a positive slope parameter ($A = +0.0308$), inverting the monotonic ranking of probabilities: $\text{AUROC}_{\text{calibrated}} = 1 - 0.6902 = 0.3098$.
4. **End-to-End Regeneration & Synchronization:** Re-ran trust model evaluation, feature ablations, figure generation (300 DPI), and error taxonomy reconciliation. Updated all tables (Tables I–VII), learned coefficients, and discussion narratives across both `paper_manuscript_final.md` and `paper_manuscript_final.tex`.
5. **Automated Consistency Verification:** Developed `scripts/verify_reproducibility_audit.py`, executing 10 automated checks (splits, overlap, schemas, leakage absence, calibration consistency, metrics, taxonomy sums, McNemar/Binomial tests, and LaTeX/Markdown typography alignment). All 10 checks passed with zero errors.

---

## 2. Phase-by-Phase Audit Findings and Root Cause Analysis

### Phase 1: Baseline Establishment & Work Preservation
- **Repository Baseline:** Backed up existing manuscripts, tables, and results to `backups/pre_phase2/`.
- **Dataset Partition Verification:**
  - PubMedQA (`pqal`): Train ($N=600$), Validation ($N=200$), Test ($N=200$).
  - Sample ID Overlap: Exactly 0 PMIDs overlap between train, val, and test splits.
  - Class Distribution across 1,000 samples: Yes (55.2%), No (33.8%), Maybe (11.0%).
  - Validation Split Distribution: Yes (55.5%), No (33.5%), Maybe (11.0%).

### Phase 2: Diagnosis and Resolution of `n_unique_pmids = 0.6`
- **Defect Observation:** In previous runs, `n_unique_pmids` in `qwen_train_trust.jsonl` and `qwen_val_trust.jsonl` was constant at 0.6 across 100% of samples.
- **Root Cause Established:** 
  1. The pipeline calls `retriever.retrieve(query)`.
  2. The retriever fused BM25 and MedCPT into candidates, but `retrieve()` directly returned only the top `evidence_top_k=3` passages formatted for prompt generation.
  3. When `TrustFeatureBuilder.build()` processed `retrieved_pmids`, it received these 3 evidence blocks. Because deduplication occurred prior to generation, `set(retrieved_pmids)` always had exactly 3 unique PMIDs.
  4. The formula `len(unique_pmids) / 5.0` evaluated to $3 / 5.0 = 0.6000$ identically for every query.
- **Implementation Fix:**
  - In [`src/retriever.py`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/src/retriever.py): Implemented `retrieve_with_candidates()`, returning:
    1. `candidate_pool`: Raw top-20 candidates from BM25 and MedCPT.
    2. `retained_candidates`: Top-5 fused RRF candidate passages (`candidate_top_k=5`).
    3. `evidence_passages`: Top-3 passages supplied to the LLM generator prompt (`evidence_top_k=3`).
  - In [`src/trust_features.py`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/src/trust_features.py): Updated `build()` to accept `candidate_pmids` representing the retained top-5 candidates, with graceful fallback to `retrieved_pmids`. Implemented explicit normalization $\text{unique\_PMIDs\_in\_top\_5} / 5.0$ capped at 1.0, handling empty retrieval gracefully ($0.0$).
  - In [`tests/test_trust_features.py`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/tests/test_trust_features.py): Created 6 rigorous unit tests:
    - 5 unique PMIDs in 5 candidates $\to 1.0$
    - Duplicate PMIDs (3 unique in 5 candidates) $\to 0.6$
    - Fewer than 5 candidates (3 unique in 3 candidates) $\to 0.6$
    - Empty candidate list $\to 0.0$
    - Multiple chunks with same PMID $\to$ deduplicated properly
    - Candidate pool ($K=5$) separated from generator evidence ($K=3$).
    *Result: All 6 tests pass cleanly.*
- **Feature Distribution Verification Post-Regeneration:**
  - **Train Split ($N=600$):** Min = 0.2000, Max = 1.0000, Mean = 0.6557, Std = 0.1580. Counts: `{0.2: 3, 0.4: 81, 0.6: 299, 0.8: 180, 1.0: 37}`.
  - **Validation Split ($N=200$):** Min = 0.2000, Max = 1.0000, Mean = 0.6510, Std = 0.1575. Counts: `{0.2: 3, 0.4: 23, 0.6: 105, 0.8: 58, 1.0: 11}`.

### Phase 3: Regeneration in Strict Dependency Order
Regeneration followed the exact dependency DAG without running unneeded costly inference:
1. Retrieval cache updated with retained candidate pools (`scripts/update_retrieval_with_candidates.py`).
2. Generator outputs (`results/generation/`) preserved (source prompts and LLM answers unchanged).
3. Claim extraction and DeBERTa-v3 NLI verification results preserved.
4. Trust feature files (`qwen_train_trust.jsonl` and `qwen_val_trust.jsonl`) regenerated with corrected candidate PMIDs (`scripts/regenerate_trust_features.py`).
5. Supervised trust models fit and calibrated on strict 450 fit / 150 cal / 200 val partitions.
6. Ablations, error taxonomy, publication figures, and manuscripts regenerated.

### Phase 4: Cross-Validation Preprocessing Leakage Elimination
- **Defect Observation:** In `src/trust_model.py`, feature scaling was executed globally on `X_train` before invoking `cross_val_score(self._model, X_scaled, y_train, ...)`. In k-fold cross-validation, the validation fold for each split was included in the calculation of `StandardScaler` mean and variance.
- **Implementation Fix:**
  - Replaced manual pre-scaling with a scikit-learn pipeline:
    ```python
    from sklearn.pipeline import Pipeline
    cv_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clone(self._model))
    ])
    cv_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=stratified_cv, scoring=scoring)
    ```
  - Scaler is now fit exclusively on the $(k-1)$ training folds within each fold iteration.
  - 5-fold CV AUROC on fitting partition ($N=450$): $0.5787 \pm 0.0275$.

### Phase 5: Re-evaluation of Trust Estimators and Calibration
All models were re-evaluated on the untouched 200 validation samples using uniform 5-bin calibration:

| Estimator | AUROC | AUPRC | ECE (5-bin) | MCE | Brier Score | High-Trust Acc | Low-Trust Acc | Dangerous Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Heuristic Baseline** | 0.5151 | 0.4495 | 0.1691 | 0.3806 | 0.2728 | 40.00% (6/15) | 43.24% (80/185) | 4.50% (9/200) |
| **Uncalibrated LogReg** | 0.6544 | 0.5717 | 0.0353 | 0.1804 | 0.2283 | 60.00% (24/40) | 38.75% (62/160) | 8.00% (16/200) |
| **Calibrated LogReg (Platt)** | 0.6544 | 0.5717 | 0.0743 | 0.0824 | 0.2366 | 100.00% (1/1) | 42.71% (85/199) | 0.00% (0/200) |
| **Calibrated LogReg (Isotonic)** | 0.6275 | 0.5263 | 0.0655 | 0.1116 | 0.2303 | 64.29% (9/14) | 41.40% (77/186) | 2.50% (5/200) |
| **Gaussian Naive Bayes** | 0.6166 | 0.5368 | **0.0263** | **0.0447** | 0.2402 | 62.50% (5/8) | 42.19% (81/192) | 1.50% (3/200) |

- **Learned Standardized Coefficients (Fit on $N=450$):**
  - `question_answer_sim`: **+0.3930** (strong positive driver of semantic relevance)
  - `answer_evidence_sim`: **-0.1948** (conditional suppressor penalizing rote text copying)
  - `max_contradiction`: **-0.1882** (strong penalty against hallucination)
  - `n_unique_pmids`: **-0.1617** (penalizes candidate dispersion across disparate documents; replaces 0.0000!)
  - `mean_max_entailment`: **+0.1227** (positive evidence grounding)
  - `frac_entailed_claims`: **+0.0435**
  - `retrieval_score_margin`: **+0.0411** (margin separation certainty)
  - `retrieval_top1_score`: **-0.0347**
  - `frac_contradicted_claims`: **+0.0295**
  - *Intercept*: **-0.4138**

### Phase 6: Feature Group Ablations
Audited feature groups in `experiments/run_ablation.py`:

| Feature Subset | $n$ Feats | AUROC | $\Delta$ AUROC | AUPRC | ECE | Brier Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **All 9 Features** | 9 | 0.6544 | — | 0.5717 | 0.0743 | 0.2366 |
| **w/o Semantic Features** | 7 | 0.5283 | −0.1261 | 0.4814 | 0.0703 | 0.2499 |
| **w/o NLI Features** | 5 | 0.6664 | +0.0120 | **0.5930** | 0.0828 | 0.2374 |
| **w/o Retrieval Features** | 6 | **0.6739** | +0.0195 | 0.5744 | 0.1204 | 0.2429 |
| **NLI Features Only** | 4 | 0.5285 | −0.1259 | 0.4732 | 0.0452 | 0.2455 |
| **Retrieval Features Only** | 3 | 0.5047 | −0.1497 | 0.4518 | 0.0530 | 0.2505 |
| **Semantic Features Only** | 2 | 0.3098* | −0.3446 | 0.3179 | **0.0234** | 0.2466 |

### Phase 7: Proof of the Semantic-Only Anomaly (AUROC = 0.3098)
- **Question Investigated:** Why did semantic-only features produce an inverted AUROC of 0.3098 when uncalibrated models have positive discriminative capacity?
- **Empirical Proof:**
  1. An uncalibrated logistic regression trained solely on `question_answer_sim` and `answer_evidence_sim` achieves an out-of-sample AUROC of **0.6902** and AUPRC of **0.6001** on the 200 validation samples.
  2. However, when Platt scaling (`CalibratedClassifierCV(method='sigmoid')`) was fit on the disjoint 150-sample internal calibration set, the empirical point-biserial correlation between uncalibrated decision logits and labels was slightly negative: $r = -0.0060$ ($p = 0.9418$).
  3. Scikit-learn's sigmoid calibration optimizes $\sigma(-(A \cdot f + B))$. Because $r < 0$, the unconstrained optimizer converged to a positive slope parameter: $A = +0.0308 > 0$.
  4. With $A > 0$, the transformation is strictly monotonically decreasing: higher raw confidence produces lower calibrated probability.
  5. This precisely inverted the ranking of the validation predictions, producing $\text{AUROC} = 1 - 0.6902 = \mathbf{0.3098}$.
- **Conclusion:** This serves as a vital case study in the paper demonstrating the fragility of unconstrained parametric calibration on small sample splits ($N=150$) when relying on low-dimensional unimodal features.

### Phase 8: Statistical Comparisons & Error Taxonomy
- **Paired Model Comparison on 200 Shared Validation Queries:**
  - Both correct: 34 (17.0%)
  - Qwen correct, MedGemma incorrect: 52 (26.0%)
  - MedGemma correct, Qwen incorrect: 22 (11.0%)
  - Both incorrect: 92 (46.0%)
  - Discordant pairs: $52 + 22 = 74$
  - **McNemar's Test (Edwards continuity correction):** $\chi^2 = \frac{(|52 - 22| - 1)^2}{74} = \mathbf{11.3649}$, $\mathbf{p = 7.48 \times 10^{-4}}$.
  - **Two-Sided Exact Binomial Test:** $\mathbf{p = 6.428 \times 10^{-4}}$.
  *Qwen significantly outperforms MedGemma.*
- **Error Taxonomy Distribution:**
  - **Qwen 2.5 7B (114 errors):**
    - $E_1$ Retrieval Failure (omitted from prompt context): 1 (0.88%)
    - $E_2$ Ranking Suboptimality: 1 (0.88%)
    - $E_5$ Contradiction / Generative Hallucination: 78 (68.42%)
    - $E_6$ Unsupported Misinterpretation: 34 (29.82%)
    - Sum = 114 (100.0%)
  - **MedGemma 1.5 4B (144 errors):**
    - $E_1$: 1 (0.69%), $E_2$: 1 (0.69%), $E_3$: 1 (0.69%), $E_4$: 10 (6.94%), $E_5$: 64 (44.44%), $E_6$: 67 (46.53%). Sum = 144 (100.0%).
  - Retrieval failures account for $< 1.8\%$ of total failures. Over $98\%$ of failures originate in generative reasoning.

### Phase 9: Retrieval Performance Audit
- BM25: Recall@1 = 0.9550, Recall@3 = 0.9800, Recall@5 = 0.9800, MRR = 0.9675
- MedCPT: Recall@1 = 0.9800, Recall@3 = 0.9950, Recall@5 = 1.0000, MRR = 0.9885
- **Hybrid RRF ($k=60$):** Recall@1 = **0.9850**, Recall@3 = **0.9950**, Recall@5 = **1.0000**, MRR = **0.9910**
- *Reconciliation:* In candidate retrieval ($K=5$), Recall is 1.0000 ($E_1 = 0$). In operative generation prompts ($K=3$), Recall is 0.9950 ($E_1 = 1$, query 16147837 omitted).

---

## 3. Metrics Before and After Feature Correction & Leakage Elimination

| Evaluation Metric / Finding | Initial Baseline (Constant Feature & Stale Splits) | Audited & Corrected Implementation | Status |
|---|:---:|:---:|:---:|
| `n_unique_pmids` Variance | $\sigma = 0.0000$ (Constant 0.6000) | $\mu = 0.6557, \sigma = 0.1580$ ($[0.2, 1.0]$) | **CORRECTED** |
| `n_unique_pmids` Feature Coefficient | $0.0000$ (Degenerate) | **-0.1617** (Valid dispersion penalty) | **CORRECTED** |
| Cross-Validation Protocol | Scaler fit across all folds | `Pipeline([('scaler', StandardScaler()), ...])` | **LEAKAGE REMOVED** |
| Trust Model AUROC (LogReg) | 0.6718 (stale 600-sample fit) | **0.6544** (strict 450 fit / 200 holdout) | **VERIFIED** |
| Trust Model AUPRC (LogReg) | 0.5785 | **0.5717** | **VERIFIED** |
| Expected Calibration Error (Isotonic) | 0.0583 | **0.0655** | **VERIFIED** |
| Expected Calibration Error (GNB) | 0.0619 | **0.0263** | **VERIFIED** |
| Isotonic High-Trust Group Accuracy | 66.67% (8/12) | **64.29%** (9/14) | **SYNCHRONIZED** |
| Isotonic High-Trust Coverage | 6.00% (12/200) | **7.00%** (14/200) | **SYNCHRONIZED** |
| Dangerous Prediction Rate (Isotonic) | 2.00% (4/200) | **2.50%** (5/200) | **SYNCHRONIZED** |
| In-Group High-Trust Error Rate | 33.33% (4/12) | **35.71%** (5/14) | **SYNCHRONIZED** |
| Uncalibrated High-Trust Acc ($\tau \ge 0.5$) | 57.89% (22/38) | **60.00%** (24/40) | **SYNCHRONIZED** |
| Heuristic High-Trust Acc ($\tau \ge 0.5$) | 40.00% (6/15) | **40.00%** (6/15) | **VERIFIED** |

---

## 4. Automated Reproducibility Audit Results

The audit script [`scripts/verify_reproducibility_audit.py`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/scripts/verify_reproducibility_audit.py) was executed to verify all 10 core constraints:

```text
================================================================================
TRUST-RAG COMPREHENSIVE REPRODUCIBILITY & CONSISTENCY AUDIT
================================================================================
[CHECK 1] Dataset splits: Train=600, Val=200, Test=200, Overlap=0 -> PASS
[CHECK 2] Feature schema & integrity: Train unique n_unique_pmids=5, Val=5 -> PASS
[CHECK 3] Leakage-free preprocessing (Pipeline in CV, strict splits) -> PASS
[CHECK 4] No silent fallback to validation data in run_ablation.py -> PASS
[CHECK 5] Verified Model Comparison Metrics (AUROC uncal=0.6544, ECE iso=0.0655, ECE gnb=0.0263) -> PASS
[CHECK 6] Feature Ablation Groups & Dimensions (9, 5, 6, 7, 4, 3, 2) -> PASS
[CHECK 7] Semantic-Only Calibration Anomaly Verified (AUROC=0.3098, AUPRC=0.3179) -> PASS
[CHECK 8] Error Taxonomy Sums & Counts (Qwen=114, MedGemma=144, E5=78/64, E6=34/67) -> PASS
[CHECK 9] Paired Comparison (Discordant 52 vs 22, McNemar chi2=11.3649, p_binom=0.000643) -> PASS
[CHECK 10] Manuscript Synchronization (.md and .tex aligned with verified results) -> PASS
================================================================================
ALL 10 REPRODUCIBILITY AND CONSISTENCY CHECKS PASSED PERFECTLY!
================================================================================
```

---

## 5. Artifact & File Change Registry

### Source Code & Test Files
- [`src/retriever.py`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/src/retriever.py): Added `retrieve_with_candidates()` preserving top-5 candidate pools and top-3 evidence passages.
- [`src/trust_features.py`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/src/trust_features.py): Updated feature builder to accept candidate PMIDs; clarified docstrings; handled empty edge cases.
- [`src/trust_model.py`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/src/trust_model.py): Implemented `Pipeline` for fold-isolated preprocessing in CV.
- [`experiments/run_ablation.py`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/experiments/run_ablation.py): Eliminated validation fallback; ensured correct 7-group ablation subsets.
- [`experiments/generate_figures.py`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/experiments/generate_figures.py): Updated coefficient plots to render verified learned weights.
- [`tests/test_trust_features.py`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/tests/test_trust_features.py): 6 unit tests covering all retrieval edge cases (100% passing).
- [`scripts/verify_reproducibility_audit.py`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/scripts/verify_reproducibility_audit.py): Automated audit test suite (100% passing).

### Machine-Readable Data & Results
- [`results/trust/qwen_train_trust.jsonl`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/results/trust/qwen_train_trust.jsonl): Regenerated 600 training samples with dynamic `n_unique_pmids`.
- [`results/trust/qwen_val_trust.jsonl`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/results/trust/qwen_val_trust.jsonl): Regenerated 200 validation samples with dynamic `n_unique_pmids`.
- [`results/final/trust_model_comparison_verified.json`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/results/final/trust_model_comparison_verified.json): Machine-readable verified trust model metrics.
- [`results/ablation/qwen_ablation_summary.json`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/results/ablation/qwen_ablation_summary.json): Machine-readable feature ablation results.
- [`results/final/model_comparison_table.csv`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/results/final/model_comparison_table.csv): Updated CSV table.
- [`results/final/model_comparison_table.tex`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/results/final/model_comparison_table.tex): Updated LaTeX snippet table.

### Publication Figures (300 DPI)
- [`results/figures/fig1_reliability_comparison_verified.png`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/results/figures/fig1_reliability_comparison_verified.png)
- [`results/figures/fig2_discrimination_curves_verified.png`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/results/figures/fig2_discrimination_curves_verified.png)
- [`results/figures/fig3_feature_coefficients_verified.png`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/results/figures/fig3_feature_coefficients_verified.png)
- [`results/figures/fig4_confusion_matrix_qwen_verified.png`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/results/figures/fig4_confusion_matrix_qwen_verified.png)
- [`results/figures/fig5_error_taxonomy_verified.png`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/results/figures/fig5_error_taxonomy_verified.png)

### Academic Manuscripts
- [`paper_manuscript_final.md`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/paper_manuscript_final.md): Fully synchronized IEEE Markdown manuscript.
- [`paper_manuscript_final.tex`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/paper_manuscript_final.tex): Fully synchronized IEEE LaTeX manuscript.

---

## 6. Exact Reproduction Commands

To reproduce the verified experimental results from scratch:
```bash
# 1. Run unit test suite
python3 -m unittest tests/test_trust_features.py

# 2. Update retrieval caches with candidate pools
python3 scripts/update_retrieval_with_candidates.py

# 3. Regenerate train and validation trust features
python3 scripts/regenerate_trust_features.py

# 4. Fit and calibrate trust models (strict 450 fit / 150 cal / 200 val)
python3 scripts/train_and_evaluate_trust.py

# 5. Run 7-group feature ablations
python3 experiments/run_ablation.py --model qwen --split val --skip-retrieval

# 6. Regenerate publication-quality figures
python3 experiments/generate_figures.py

# 7. Execute automated 10-point reproducibility audit
python3 scripts/verify_reproducibility_audit.py
```

---

## 7. Limitations & Honest Scope

1. **Single-Benchmark Evaluation:** All experiments evaluate on PubMedQA (`pqal`). Multi-turn clinical conversations and complex EHR notes require future prospective validation.
2. **Computational Overhead of NLI:** Sentence decomposition and DeBERTa-v3 inference add $\sim 1.5$s per query. Distilled student cross-encoders will be necessary for high-throughput clinical deployments.
3. **Prompt Conservatism:** Zero-shot prompting yields high "Maybe" prediction rates (55% Qwen, 66% MedGemma); few-shot chain-of-thought prompting is recommended for future work.
4. **Decoupled Selective Classification vs. Autonomous Safety:** Calibrated trust models flag ungrounded answers for clinician review (identifying high-trust subgroups with 64.29% accuracy at 7.00% coverage with a 2.50% dangerous error rate), but do not constitute autonomous clinical safety. Mandatory clinician verification remains imperative.

---

## 8. Final Audit Sign-Off

All phases requested by the audit charter are complete. Code, data, results, figures, and manuscripts are unified, verified, and ready for publication submission.
