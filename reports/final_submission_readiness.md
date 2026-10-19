# TRUST-RAG: Final Submission Readiness and Experimental Verification Report

**Audit Date:** October 15, 2026  
**Auditor Role:** Senior ML Research Engineer & Academic Manuscript Auditor  
**Repository:** `TRUST-RAG`  
**Target Manuscripts:** [`paper_manuscript_final.md`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/paper_manuscript_final.md) & [`paper_manuscript_final.tex`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/paper_manuscript_final.tex)  
**Verification Script:** [`scripts/verify_reproducibility_audit.py`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/scripts/verify_reproducibility_audit.py)  
**Deep Inspection Script:** [`scripts/deep_inspection.py`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/scripts/deep_inspection.py)  

---

## 1. Executive Recommendation

### Verdict: **READY FOR SUBMISSION**

**Condition:** Submission must preserve the established framing of TRUST-RAG as an **evidence-grounded selective decision-support filter with mandatory clinician verification**, rather than an autonomous clinical decision-making agent.

Every numerical assertion, feature dimension, calibration curve, error category, and statistical p-value in both the Markdown and LaTeX manuscripts has been cross-checked against raw, machine-readable artifacts and verified via automated reproducible scripts. Zero synthetic, unverified, or fabricated metrics exist in the reported literature.

---

## 2. Comprehensive Itemized Audit Checks

| Audit Dimension | Status | Verified Evidence & Methodological Finding |
|---|:---:|---|
| **1. Dataset Partition Integrity** | **PASS** | 600 training, 200 validation, and 200 test samples verified in `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`. Set intersection confirms exactly **0** PMID overlap between any split pair. Partition metadata frozen in `results/final/train_calibration_split_metadata.json`. |
| **2. Preprocessing Data Leakage** | **PASS** | `src/trust_model.py` integrates `StandardScaler` and `LogisticRegression` within a scikit-learn `Pipeline` during cross-validation. Scalers are fit strictly on internal training folds ($N=450$). Calibration estimators fit strictly on internal calibration folds ($N=150$). Validation partition ($N=200$) is strictly untouched during all feature normalization, model fitting, and calibration. |
| **3. Elimination of Silent Data Fallback** | **PASS** | In `experiments/run_ablation.py`, the silent fallback `X_train, y_train = X_val, y_val` was eliminated. If training features are missing, an explicit `FileNotFoundError` is raised. |
| **4. Feature Schema & Variance (`n_unique_pmids`)** | **PASS** | Resolved the root cause where `n_unique_pmids` evaluated constantly to $3/5.0 = 0.6000$ due to generator evidence truncation ($K=3$). Created `retrieve_with_candidates()` in `src/retriever.py` preserving top-5 candidate pools. Regenerated train and validation features. Distribution verified: Train ($N=600$): Min 0.20, Max 1.00, Mean 0.6557, Std 0.1580; Val ($N=200$): Min 0.20, Max 1.00, Mean 0.6510, Std 0.1575. Unit test suite `tests/test_trust_features.py` (6 tests) passes 100%. |
| **5. Model Comparison Metrics** | **PASS** | Recomputed from raw scores in `results/final/trust_model_comparison_verified.json`. Uncalibrated LogReg: AUROC 0.6544, AUPRC 0.5717, ECE 0.0353, Brier 0.2283. Platt LogReg: AUROC 0.6544, AUPRC 0.5717, ECE 0.0743, Brier 0.2366. Isotonic LogReg: AUROC 0.6275, AUPRC 0.5263, ECE 0.0655, Brier 0.2303. GNB: AUROC 0.6166, AUPRC 0.5368, ECE 0.0263, Brier 0.2402. Heuristic: AUROC 0.5151, AUPRC 0.4495, ECE 0.1691. |
| **6. Platt-Calibrated 100% Accuracy Transparency** | **PASS** | Inspected raw predictions. At $\tau = 0.5$, Platt scaling classifies **exactly 1 sample** as high trust ($0.50\%$ coverage), which happens to be correct ($1/1 = 100.0\%$). Added Platt scaling explicitly to Table V and documented in prose that 100% accuracy is an artifact of severe score shrinkage toward the prior ($P \approx 0.43$), preventing misleading claims of selector efficacy. |
| **7. Feature Group Ablations** | **PASS** | Executed in `results/ablation/qwen_ablation_summary.json`. Verified feature dimensions: All 9 ($n=9$, AUROC 0.6544), w/o Semantic ($n=7$, AUROC 0.5283), w/o NLI ($n=5$, AUROC 0.6664), w/o Retrieval ($n=6$, AUROC 0.6739), NLI Only ($n=4$, AUROC 0.5285), Retrieval Only ($n=3$, AUROC 0.5047), Semantic Only ($n=2$, AUROC 0.3098). |
| **8. Honest Reporting of Non-Monotonic AUROC** | **PASS** | Explicitly investigated why the model without retrieval features achieves slightly higher AUROC (0.6739) than the full 9-feature model (0.6544). Near-ceiling retrieval recall (Recall@1=0.9850, Recall@5=1.0000) causes retrieval scores to have low inter-class variance, acting as weak linear predictors that add mild estimation variance. However, omitting retrieval features severely degrades calibration error (ECE jumps from 0.0743 to 0.1204, a +62.0% increase) and strips the document dispersion penalty (`n_unique_pmids = -0.1617`). The manuscript documents this transparently without claiming every feature group monotonically boosts AUROC. |
| **9. Semantic-Only Anomaly & AUPRC Resolution** | **PASS** | Verified that uncalibrated semantic-only LogReg achieves valid out-of-sample AUROC of **0.6902** and AUPRC of **0.5942** (via `auc(recall, precision)` in `src/evaluation.py`; 0.6001 via `average_precision_score`). On the 150-sample calibration set, point-biserial correlation was slightly negative ($r = -0.006013, p = 0.9418$). Scikit-learn's unconstrained sigmoid calibration optimized $1 / (1 + \exp(a \cdot f + b))$ and converged to a positive slope parameter ($a = +0.030799, b = 0.390486$). Positive $a$ makes calibrated probabilities strictly decreasing with decision scores, inverting the ROC ranking: $\text{AUROC}_{\text{cal}} = 1 - 0.6902 = \mathbf{0.3098}$ (AUPRC = 0.3179). Both manuscripts now report 0.5942 with integration explanation. |
| **10. Paired Statistical Comparisons** | **PASS** | Recomputed on the exact 200 shared validation queries between Qwen and MedGemma. Contingency matrix: Both correct = 34, Qwen-only = 52, MedGemma-only = 22, Both incorrect = 92. Discordant pairs = 74. Edwards continuity-corrected McNemar: $\chi^2 = \mathbf{11.3649}, \mathbf{p = 7.48 \times 10^{-4}}$ (raw $\chi^2 = 12.1622, p = 4.88 \times 10^{-4}$). Exact two-sided Binomial test: $\mathbf{p = 6.428 \times 10^{-4}}$. |
| **11. Error Taxonomy Operational Reconciliation** | **PASS** | Classified 114 Qwen errors (E1=1, E2=1, E5=78, E6=34) and 144 MedGemma errors (E1=1, E2=1, E3=1, E4=10, E5=64, E6=67). Sums equal total error counts exactly. Reconciled candidate pool ($K=5$: Recall@5 = 1.0000, $E_1=0$) with prompt context ($K=3$: Recall@3 = 0.9950, $E_1=1$). Described explicitly as operational classifications based on measurable rank and NLI thresholds, not proof of causal etiology. |
| **12. Candidate-Pool Retrieval Framing** | **PASS** | Verified that perfect candidate-pool Recall@5 (1.0000) is not overstated as proof that retrieval cannot contribute to errors. The manuscript clarifies that high candidate recall shifts operational failures to generative reasoning, but retrieval ranking, passage order, and distractor context still influence generation. |
| **13. Cross-Manuscript Synchronization** | **PASS** | `paper_manuscript_final.md` and `paper_manuscript_final.tex` were fully synchronized across all 7 tables, feature weights, discussion sections, and conclusion metrics. Automated verification script `scripts/verify_reproducibility_audit.py` passes 10/10 checks. |
| **14. Non-Causal Feature Interpretation** | **PASS** | Replaced causal assertions in Section VI-E with qualified correlational language. Standardized regression coefficients and ablation differences are framed as empirical associations rather than proven causal mechanisms. |
| **15. Clinical Scope & Humility** | **PASS** | High-trust accuracy of 64.29% at 7.00% coverage (Isotonic) includes 5 incorrect predictions (35.71% in-group error rate, 2.50% cohort dangerous rate). All claims of "autonomous safety" or "hospital deployment readiness" were removed in favor of selective decision-support filtering with mandatory clinician verification. |

---

## 3. Discrepancy Register and Resolutions

1. **`n_unique_pmids` Constant Value (0.6000):**
   - *Cause:* Deduplicated top-3 evidence passages were passed to the feature builder instead of the top-5 candidate pool.
   - *Resolution:* Implemented `retrieve_with_candidates()` in `src/retriever.py`, updated `src/trust_features.py`, added unit tests, and regenerated feature files. Real distribution verified ($\mu = 0.6557, \sigma = 0.1580$).
2. **Cross-Validation Preprocessing Leakage:**
   - *Cause:* Feature scaler was fit globally on `X_train` before calling `cross_val_score`.
   - *Resolution:* Wrapped scaler and estimator in a scikit-learn `Pipeline` inside `src/trust_model.py`.
3. **Ablation Training Fallback:**
   - *Cause:* `run_ablation.py` assigned `X_train, y_train = X_val, y_val` when training features were missing.
   - *Resolution:* Replaced with an explicit `FileNotFoundError`.
4. **Apparent Platt Discrimination Anomaly:**
   - *Cause:* Platt model had AUROC 0.6718 while uncalibrated model reported 0.6499 in early drafts due to differing fit partitions.
   - *Resolution:* Standardized base model fitting to the strict 450-sample partition. Both achieve mathematically invariant AUROC of 0.6544.
5. **Semantic-Only AUROC Inversion (0.3098):**
   - *Cause:* Near-zero negative correlation on the 150-sample calibration set caused unconstrained sigmoid calibration to fit a positive slope $a = +0.0308$, inverting probabilities.
   - *Resolution:* Proven empirically via `scripts/deep_inspection.py` and documented as a case study on calibration instability.
6. **Semantic-Only AUPRC Discrepancy (0.5942 vs. 0.6001):**
   - *Cause:* `src/evaluation.py` line 270 computes `auc(recall, precision)` via trapezoidal integration, yielding **0.5942**, whereas `average_precision_score` computes rectangular step-function area, yielding **0.6001**.
   - *Resolution:* Synchronized both manuscripts to cite 0.5942 (matching repository `Evaluator` output) with an explanatory note regarding the 0.6001 step-function value.
7. **Platt High-Trust 100% Accuracy Transparency:**
   - *Cause:* Platt high-trust tier had 100% accuracy, but only 1 sample exceeded $\tau = 0.5$.
   - *Resolution:* Added Platt scaling to Table V and explicitly noted the $n=1$ sample count and $0.50\%$ coverage in the manuscript text.
8. **Disparity between Full 9-Feature Model (0.6544) and Without Retrieval (0.6739):**
   - *Cause:* Low inter-class variance in retrieval scores on a near-ceiling retrieval benchmark adds mild estimation noise to discriminative ranking.
   - *Resolution:* Transparently documented in Section VI-E; emphasized that retrieval features are necessary to prevent severe calibration degradation (ECE increases from 0.0743 to 0.1204 without them).
9. **Causal Interpretation of Regression Weights & Ablations:**
   - *Cause:* Section VI-E previously used causal verbs ("drives trust", "penalizing", "causes").
   - *Resolution:* Replaced with correlational and linear weighting terminology ("exhibits strongest positive association", "assigned large negative weight").
10. **Error Taxonomy Causality Overstatement & Operational Framing:**
   - *Cause:* Previous text could be read as asserting that retrieval never contributes to generation failures due to near-ceiling retrieval recall (Recall@5 = 1.0000).
   - *Resolution:* Reframed as operational classifications; distinguished candidate pool ($K=5$) from operative prompt pool ($K=3$); and adopted the exact non-causal framing across Section VI-F, Section VIII, and the abstract:
     > *"Under the proposed operational error taxonomy, 98.24% of Qwen's errors were classified as generation-level hallucination or unsupported misinterpretation. These findings suggest that, under high-recall retrieval on PubMedQA, downstream answer generation and interpretation warrant particular attention. However, the taxonomy does not establish definitive causal attribution, and retrieval ranking and evidence composition may still influence model behavior. Post-generation trust estimation may support selective review, but its effectiveness and generalizability require further evaluation."*

---

## 4. Manuscript & Figure Changes

### Updated Files
1. [`paper_manuscript_final.md`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/paper_manuscript_final.md):
   - Table IV, Table V (added Platt row), Table VI, and Table VII updated.
   - Learned standardized coefficients synchronized (`question_answer_sim`: +0.3930, `n_unique_pmids`: -0.1617, Intercept: -0.4138).
   - Added honest discussion of feature exclusion ($\Delta$ AUROC = +0.0195 without retrieval vs ECE degradation).
   - Resolved semantic-only AUPRC to 0.5942 with step-function integration note (0.6001).
   - Replaced causal assertions in Section VI-E with qualified empirical interpretations.
   - Refined error taxonomy narrative to emphasize operational categorization rather than causal proof.
   - Synchronized selective prediction metrics (Isotonic: 64.29% accuracy, 7.00% coverage, 35.71% in-group error, 2.50% cohort dangerous rate).
2. [`paper_manuscript_final.tex`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/paper_manuscript_final.tex):
   - Mirrored all table structures, coefficient listings, narrative refinements, non-causal phrasing, and reference numbers from Markdown.
3. Publication Figures (300 DPI regenerated):
   - `results/figures/fig1_reliability_comparison_verified.png`
   - `results/figures/fig2_discrimination_curves_verified.png`
   - `results/figures/fig3_feature_coefficients_verified.png`
   - `results/figures/fig4_confusion_matrix_qwen_verified.png`
   - `results/figures/fig5_error_taxonomy_verified.png`

---

## 5. Artifact Provenance and Reproduction Commands

### Exact Artifact Paths
- Training Features: `results/trust/qwen_train_trust.jsonl` ($N=600$)
- Validation Features: `results/trust/qwen_val_trust.jsonl` ($N=200$)
- Model Comparison Results: `results/final/trust_model_comparison_verified.json`
- Feature Ablation Summary: `results/ablation/qwen_ablation_summary.json`
- Split Metadata: `results/final/train_calibration_split_metadata.json`
- Error Taxonomy: `results/final/qwen_val_errors.json`, `results/final/medgemma_val_errors.json`

### Reproduction Commands
```bash
# 1. Execute Unit Test Suite for Trust Feature Extraction
python3 -m unittest tests/test_trust_features.py

# 2. Update Retrieval Caches with Candidate Pools
python3 scripts/update_retrieval_with_candidates.py

# 3. Regenerate Training and Validation Trust Features
python3 scripts/regenerate_trust_features.py

# 4. Train, Calibrate, and Evaluate Trust Estimators
python3 scripts/train_and_evaluate_trust.py

# 5. Execute 7-Group Feature Ablation Study
python3 experiments/run_ablation.py --model qwen --split val --skip-retrieval

# 6. Regenerate High-Resolution Publication Figures
python3 experiments/generate_figures.py

# 7. Run Deep Parameter Inspection on Calibration & Inversion
python3 scripts/deep_inspection.py

# 8. Run Automated 10-Point Reproducibility Audit
python3 scripts/verify_reproducibility_audit.py
```

---

## 6. Remaining Limitations & Honest Scope

1. **Single-Benchmark Evaluation:** All experiments evaluate on PubMedQA (`pqal`). Multi-turn clinical conversations and complex EHR notes require future prospective validation.
2. **Computational Overhead of NLI:** Sentence decomposition and DeBERTa-v3 inference add $\sim 1.5$s per query. Distilled student cross-encoders will be necessary for high-throughput clinical deployments.
3. **Zero-Shot Prompt Conservatism:** Zero-shot prompting yields high "Maybe" prediction rates (55% Qwen, 66% MedGemma); few-shot chain-of-thought prompting is recommended for future work.
4. **Decoupled Selective Classification vs. Autonomous Safety:** Calibrated trust models flag ungrounded answers for clinician review (identifying high-trust subgroups with 64.29% accuracy at 7.00% coverage with a 2.50% dangerous error rate), but do not constitute autonomous clinical safety. Mandatory clinician verification remains imperative.

---

## 7. Final Sign-Off

The TRUST-RAG research paper and codebase have undergone a complete, rigorous experimental audit. All identified defects have been eliminated, all numbers are backed by raw artifacts, and the manuscript text is sound, nuanced, and ready for publication submission.
