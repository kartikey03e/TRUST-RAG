# TRUST-RAG Results & Discussion Change Log

This document records all modifications made when transforming the draft `results_section_verified.md` into the publication-ready `results_section_final.md`. Every change is grounded in empirical artifacts from the repository.

---

## 1. Summary of Changes by Section

### Title and Section Structure
- **Old:** `# Section 4: Experimental Results and Discussion` with colloquial Markdown callout banners.
- **New:** Formatted in formal IEEE academic style as `V. RESULTS AND DISCUSSION` with structured subsections A through G.
- **Evidence/Rationale:** Conforms to IEEE manuscript formatting guidelines and eliminates informal Markdown callouts.

---

### Subsection A: Evidence Retrieval Performance
- **Old:** Claimed that Hybrid RRF "strictly dominates both individual methods across all metrics."
- **New:** Corrected to note that Hybrid RRF ties dense retrieval (MedCPT) on Recall@3 (0.9950) and Recall@5 (1.0000), while achieving an advantage on Recall@1 (0.9850 vs. 0.9800) and MRR (0.9910 vs. 0.9885).
- **Old:** Stated that because Recall@5 is 1.0000, generation reasoning failures cannot be caused by missing evidence.
- **New:** Added explicit distinction between Recall@5 and Recall@3. Noted that because the generator only receives top-3 passages (`evidence_top_k=3`), Recall@3 (0.9950) is the operative context boundary.
- **Evidence:** `results/retrieval/val_retrieval_metrics.json` and `results/ablation/medgemma_ablation_summary.json`.

---

### Subsection B: Zero-Shot Question Answering Performance
- **Old:** Combined answer generation metrics with heuristic trust metrics (ECE, Brier, AUROC, Dangerous Rate) in a single benchmark table.
- **New:** Separated the answer generation performance (Accuracy, Macro-F1, Weighted-F1, Bootstrap CI) into Table II, while moving trust estimation metrics to Table IV.
- **Old:** Used emotive language ("pathological affirmation bias", "negative-evidence reasoning deficit").
- **New:** Replaced with objective academic descriptions of observed prediction skew (132 and 110 "Maybe" predictions vs. 22 gold cases).
- **Evidence:** `results/final/qwen_val_evaluation.json`, `results/final/medgemma_val_evaluation.json`, and bootstrap percentile CI calculations in `src/evaluation.py`.

---

### Subsection C: Trust Estimation Calibration & Discrimination
- **Old:** Evaluated models using sub-split validation samples (150 train, 50 cal) due to missing `qwen_train_trust.jsonl`.
- **New:** Resolved completely: generated `results/trust/qwen_train_trust.jsonl` across the full PubMedQA training partition ($N=600$) using the end-to-end TRUST-RAG feature extraction pipeline. Models were fit on the training partition ($N=450$), calibrated on an internal stratified calibration subsplit ($N=150$, 25%), and evaluated strictly out-of-sample on the untouched validation split ($N=200$).
- **New Verified Metrics ($N=600$ Train Split, $N=200$ Untouched Val):**
  - Calibrated LogReg (Platt Scaling): AUROC = **0.6718**, AUPRC = **0.5785**, ECE = 0.1252, Brier = 0.2409.
  - Uncalibrated LogReg: AUROC = **0.6499**, AUPRC = **0.5599**, ECE = 0.0759, Brier = **0.2296**.
  - Gaussian Naive Bayes: AUROC = **0.6302**, AUPRC = **0.5438**, ECE = 0.0619, MCE = **0.0768**, Brier = 0.2417.
  - Calibrated LogReg (Isotonic): AUROC = **0.6239**, AUPRC = **0.5529**, ECE = **0.0583**, Brier = 0.2368.
  - Heuristic Baseline: AUROC = 0.5151, AUPRC = 0.4495, ECE = 0.1691, Brier = 0.2728.
- **Evidence:** `results/trust/qwen_train_trust.jsonl`, `results/final/trust_model_comparison_verified.json`, and `results/ablation/qwen_ablation_summary.json`.

---

### Subsection D: Trust–Correctness Contingency Analysis
- **Old:** Contingency table for isotonic model was based on validation-fitted predictions.
- **New:** Updated with true out-of-sample predictions on untouched validation split ($N=200$):
  - Heuristic: High-trust accuracy = 40.00% (6/15), Low-trust accuracy = 43.24% (80/185).
  - Uncalibrated LogReg: High-trust accuracy = **54.05%** (20/37), Low-trust accuracy = **40.49%** (66/163).
  - Isotonic LogReg: High-trust accuracy = **66.67%** (8/12), Low-trust accuracy = **41.49%** (78/188).
  - Dangerous Failure Rate under Isotonic model reduced to **2.00%** (4/200), providing a +23.67 percentage point boost over base accuracy (43.00%).
- **Evidence:** `results/final/trust_model_comparison_verified.json`.

---

### Subsection E: Feature Importance and Signal Ablation
- **Old:** Feature ablation code had string name mismatches (`frac_entailed`, `retrieval_entropy`, `lexical_overlap`).
- **New:** Repaired all variable names in `experiments/run_ablation.py`. Executed clean feature group ablations using the decoupled train/calibration/validation protocol across the complete 600-sample training dataset.
- **New:** Standardized coefficients trained on the complete training partition ($N=600$):
  - `mean_max_entailment`: **+0.3280**
  - `question_answer_sim`: **+0.2960**
  - `frac_contradicted_claims`: **+0.1405**
  - `retrieval_score_margin`: **+0.1063**
  - `retrieval_top1_score`: **−0.0424**
  - `answer_evidence_sim`: **−0.0619**
  - `frac_entailed_claims`: **−0.1813**
  - `max_contradiction`: **−0.2093**
  - `n_unique_pmids`: **0.0000**
  - *Intercept*: **−0.3993**
- **Evidence:** `results/ablation/qwen_ablation_summary.json` and `results/figures/fig3_feature_coefficients_verified.png`.

---

## 2. Regenerated Figures

All figures were regenerated at 300 DPI and saved to `results/figures/`:
1. `fig1_reliability_comparison_verified.png` & `figure1_reliability_qwen.png`: Compares Heuristic baseline vs. Isotonic-calibrated model on untouched validation split.
2. `fig2_discrimination_curves_verified.png` & `figure2_roc_pr_qwen.png`: Empirical ROC and PR curves for all model variants on untouched validation split.
3. `fig3_feature_coefficients_verified.png` & `figure5_feature_importance.png`: Standardized logistic regression coefficients for all 9 approved features learned from the complete 600-sample training partition.
4. `fig4_confusion_matrix_qwen_verified.png` & `figure6_confusion_qwen.png`: Qwen 2.5 7B decision confusion matrix against gold labels on validation split.
5. `fig5_error_taxonomy_verified.png` & `figure4_errors_qwen.png`: 6-category error taxonomy distribution for Qwen and MedGemma.

---

## 3. Status of Previous Limitations

1. **Full 600-Sample Training Scaling: FULLY RESOLVED.** The complete 600-sample training split (`data/processed/train.jsonl`) has been fully processed through the retrieval, generation, decision extraction, claim-level NLI verification, and trust feature building pipeline. All 600 feature vectors are saved in `results/trust/qwen_train_trust.jsonl`. Trust models were retrained and recalibrated without any validation label leakage.
2. **Independent Test Split Evaluation:** While validation split evaluation is now strictly decoupled from training and calibration, evaluating on the held-out 200-sample test split (`data/processed/test.jsonl`) remains available for final confirmation.
