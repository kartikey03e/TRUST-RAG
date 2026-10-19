# V. RESULTS AND DISCUSSION

This section presents the empirical evaluation of the TRUST-RAG framework on the PubMedQA (`pqal`) labeled validation split ($N = 200$). The evaluation is organized into six experimental subsections followed by a critical discussion: (A) evidence retrieval performance across sparse, dense, and hybrid methods; (B) zero-shot question answering performance and decision distributions across two instruction-tuned language models; (C) trust estimation discrimination and probability calibration across multiple modeling paradigms; (D) contingency analysis of trust assignments versus empirical correctness; (E) feature importance and group-level signal ablation; (F) hierarchical error taxonomy analysis; and (G) a synthesis of findings and experimental limitations.

---

## A. Evidence Retrieval Performance

The performance of the evidence retrieval module is evaluated to determine whether relevant biomedical literature is successfully captured prior to answer generation. Table I reports the retrieval metrics across sparse lexical retrieval (BM25), dense semantic retrieval (MedCPT with FAISS inner-product search), and hybrid Reciprocal Rank Fusion (RRF, $k=60$). A retrieval hit is recorded if the gold PubMed abstract PMID is present within the top-$K$ PMID-level documents ($K \in \{1, 3, 5\}$).

### TABLE I
### Evidence Retrieval Performance on PubMedQA Validation Split ($N=200$)

| Retrieval Method | Recall@1 | Recall@3 | Recall@5 | Mean Reciprocal Rank (MRR) | Unretrieved Queries ($n$) |
|---|:---:|:---:|:---:|:---:|:---:|
| BM25 (Sparse Lexical) | 0.9550 | 0.9800 | 0.9800 | 0.9675 | 4 |
| MedCPT (Dense Semantic) | 0.9800 | 0.9950 | 1.0000 | 0.9885 | 0 |
| Hybrid RRF ($k=60$) | **0.9850** | **0.9950** | **1.0000** | **0.9910** | **0** |

As shown in Table I, Hybrid RRF yields the highest observed Recall@1 (0.9850) and MRR (0.9910). MedCPT and Hybrid RRF tie on Recall@3 (0.9950) and Recall@5 (1.0000), both successfully placing the gold document within the top-5 candidates for all 200 evaluated queries. In contrast, BM25 fails to retrieve the gold abstract within the top-5 candidates for four queries. 

The primary distinction between MedCPT and Hybrid RRF appears at rank 1, where Hybrid RRF achieves an additional hit (+0.0050 Recall@1), suggesting that sparse lexical matching provides an exact keyword anchor for specialized biomedical nomenclature that dense semantic representations may occasionally rank lower. 

Because the generation module is configured to inject the top-3 retrieved evidence passages into the prompt context (`evidence_top_k=3`), Recall@3 represents the operative context boundary for answer generation. In the evaluated split, Recall@3 is 0.9950 for both MedCPT and Hybrid RRF, corresponding to 199 out of 200 queries receiving the gold abstract in their generation context. Consequently, failure of downstream answer generation can only rarely be attributed to an absence of relevant literature in the prompt context.

---

## B. Zero-Shot Question Answering Performance

Table II reports end-to-end question answering performance for two instruction-tuned language models—MedGemma 1.5 4B-IT (evaluated in 16-bit floating-point precision) and Qwen 2.5 7B-Instruct (evaluated with 4-bit NormalFloat quantization)—operating in a zero-shot setting receiving top-3 Hybrid RRF evidence passages.

### TABLE II
### Question Answering Performance on PubMedQA Validation Split ($N=200$)

| Model | Accuracy | 95% Confidence Interval | Macro-F1 | Weighted-F1 |
|---|:---:|:---:|:---:|:---:|
| MedGemma 1.5 4B-IT | 0.2800 | [0.2200, 0.3450] | 0.2644 | 0.3053 |
| Qwen 2.5 7B-Instruct | **0.4300** | [0.3600, 0.5001] | **0.4183** | **0.5035** |

*Note: Confidence intervals are computed via empirical percentile bootstrap with $B = 1,000$ resamples at the 95% confidence level.*

Qwen 2.5 7B-Instruct achieves an accuracy of 0.4300 (Macro-F1 = 0.4183), outperforming MedGemma 1.5 4B-IT (accuracy = 0.2800, Macro-F1 = 0.2644) by 15.00 percentage points. To examine the underlying classification behavior, Table III details the per-class performance metrics alongside the gold and predicted class distributions.

### TABLE III
### Per-Class Performance and Prediction Distribution ($N=200$)

| Model | Class Label | Gold Support ($N$) | Predicted Count ($N$) | Recall | Precision | F1-Score |
|---|---|:---:|:---:|:---:|:---:|:---:|
| MedGemma 1.5 4B-IT | Yes | 111 | 52 | 0.2883 | 0.6154 | 0.3926 |
| | No | 67 | 16 | 0.1194 | 0.5000 | 0.1928 |
| | Maybe | 22 | 132 | **0.7273** | 0.1212 | 0.2078 |
| | *Macro Average* | *200* | *200* | *0.3783* | *0.4122* | *0.2644* |
| Qwen 2.5 7B-Instruct | Yes | 111 | 65 | **0.4595** | 0.7846 | **0.5795** |
| | No | 67 | 25 | **0.3284** | 0.8800 | **0.4783** |
| | Maybe | 22 | 110 | 0.5909 | 0.1182 | 0.1970 |
| | *Macro Average* | *200* | *200* | *0.4596* | *0.5943* | *0.4183* |

Two notable patterns emerge from Table III and the Qwen confusion matrix (illustrated in Fig. 4):
1. **Skew Toward "Maybe" Predictions:** In the evaluated split, the gold distribution contains 22 "Maybe" instances (11.0%). However, MedGemma predicts "Maybe" for 132 instances (66.0%), and Qwen predicts "Maybe" for 110 instances (55.0%). Consequently, precision on the "Maybe" class drops to 0.1212 for MedGemma and 0.1182 for Qwen.
2. **Lower Sensitivity on Negative Decisions:** Both models exhibit their lowest recall on the "No" class (0.1194 for MedGemma, 0.3284 for Qwen), indicating that refuting a clinical hypothesis based on retrieved evidence is more challenging under this zero-shot prompt configuration than affirming one.

---

## C. Trust Estimation: Calibration and Discrimination

To evaluate whether answer reliability can be predicted post-generation without data leakage, trust model estimators were trained strictly on the complete training partition ($N=600$ records from `data/processed/train.jsonl`), with calibration fit on an internal stratified calibration subsplit ($N=150$, 25% of training data) without accessing validation data. Final models were evaluated strictly on the untouched validation split ($N=200$, `results/trust/qwen_val_trust.jsonl`). Discrimination is assessed using the Area Under the Receiver Operating Characteristic curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC). Confidence calibration is evaluated using Expected Calibration Error (ECE, 5 equal-width bins), Maximum Calibration Error (MCE), and Brier score.

### TABLE IV
### Discrimination and Calibration Across Trust Model Variants on Qwen 2.5 7B Outputs ($N=200$, Untouched Validation Split)

| Trust Model Architecture | AUROC ($\uparrow$) | AUPRC ($\uparrow$) | ECE ($\downarrow$) | MCE ($\downarrow$) | Brier Score ($\downarrow$) |
|---|:---:|:---:|:---:|:---:|:---:|
| Heuristic Baseline (Equal Weights) | 0.5151 | 0.4495 | 0.1691 | 0.3806 | 0.2728 |
| Uncalibrated Logistic Regression | 0.6499 | 0.5599 | 0.0759 | 0.3487 | **0.2296** |
| Calibrated LogReg (Platt Scaling) | **0.6718** | **0.5785** | 0.1252 | 0.1709 | 0.2409 |
| Calibrated LogReg (Isotonic Regression) | 0.6239 | 0.5529 | **0.0583** | 0.1333 | 0.2368 |
| Gaussian Naive Bayes (GNB) | 0.6302 | 0.5438 | 0.0619 | **0.0768** | 0.2417 |

*Note: Models were fit on $N=450$ training samples, calibrated on $N=150$ training samples, and evaluated on the untouched $N=200$ validation split. The positive prevalence base rate for AUPRC on the validation split is 0.4300 (86/200).*

The results in Table IV and the reliability diagrams in Fig. 1 demonstrate distinct performance characteristics across modeling approaches under strict out-of-sample evaluation:
- **Heuristic Baseline Limitations:** The unweighted heuristic baseline yields an AUROC of 0.5151 (barely above random chance 0.5000) and an ECE of 0.1691 with an MCE of 0.3806 and Brier score of 0.2728. This confirms that assigning equal static weights to raw features without empirical learning cannot discriminate correct from incorrect answers.
- **Discrimination via Supervised Learning:** Training the logistic regression estimator on the 600-sample training partition elevates AUROC to **0.6499** and AUPRC to **0.5599** (substantially above the positive prevalence base rate of 0.4300, shown in Fig. 2). Its Brier score drops to **0.2296**, achieving the lowest mean squared error among all uncalibrated variants.
- **Superior Discrimination via Platt Scaling:** Calibrating logistic regression via Platt scaling (sigmoid calibration) fit on the 150-sample internal calibration set achieves the highest discrimination, reaching an AUROC of **0.6718** and AUPRC of **0.5785** on the untouched validation set, with an MCE of 0.1709.
- **Superior Probability Calibration via Isotonic Regression:** Non-parametric isotonic regression achieves the lowest Expected Calibration Error, reducing ECE to **0.0583** (5.83%) and MCE to 0.1333, while maintaining strong discrimination (AUROC = 0.6239, AUPRC = 0.5529).
- **Strong Performance of Gaussian Naive Bayes:** Gaussian Naive Bayes achieves competitive discrimination (AUROC = 0.6302, AUPRC = 0.5438) and the lowest Maximum Calibration Error of **0.0768** with an ECE of 0.0619.

When evaluated on the outputs of MedGemma 1.5 4B-IT, the heuristic baseline yields AUROC = 0.4753, ECE = 0.1476, and MCE = 0.5245. Uncalibrated logistic regression achieves AUROC = 0.5694, ECE = 0.0046, and MCE = 0.1200, while isotonic-calibrated logistic regression achieves AUROC = 0.5193, ECE = 0.0047, and MCE = 0.1145 (recorded in `results/ablation/medgemma_ablation_summary.json`).

---

## D. Trust–Correctness Contingency Analysis

To examine how trust scores separate correct from incorrect predictions at an operational cutoff, Table V presents the 2×2 contingency matrix between binary trust assignments ($\tau = 0.5$) and ground-truth answer correctness on the untouched validation split ($N=200$).

### TABLE V
### Trust Assignment Versus Observed Correctness at Threshold $\tau = 0.5$ ($N=200$, Untouched Validation Split)

| Trust Model | Assignment Group | Observed Correct ($N=86$) | Observed Incorrect ($N=114$) | Subgroup Total | Subgroup Accuracy |
|---|---|:---:|:---:|:---:|:---:|
| **Heuristic Baseline** | High Trust ($\ge 0.5$) | 6 | 9 | 15 | **40.00%** |
| | Low Trust ($< 0.5$) | 80 | 105 | 185 | **43.24%** |
| | *Entire Validation Set* | *86* | *114* | *200* | *43.00%* |
| **Uncalibrated LogReg** | High Trust ($\ge 0.5$) | 20 | 17 | 37 | **54.05%** |
| | Low Trust ($< 0.5$) | 66 | 97 | 163 | **40.49%** |
| | *Entire Validation Set* | *86* | *114* | *200* | *43.00%* |
| **Isotonic-Calibrated** | High Trust ($\ge 0.5$) | 8 | 4 | 12 | **66.67%** |
| | Low Trust ($< 0.5$) | 78 | 110 | 188 | **41.49%** |
| | *Entire Validation Set* | *86* | *114* | *200* | *43.00%* |

Key clinical findings emerge from Table V:
1. **Inversion of the Heuristic Baseline:** Under the heuristic score, the high-trust subgroup accuracy (40.00%, 6/15) is lower than the low-trust subgroup accuracy (43.24%, 80/185), confirming that heuristic confidence is negatively correlated with accuracy at $\tau = 0.5$.
2. **Effective Accuracy Stratification by Supervised LogReg:** The uncalibrated logistic regression model stratifies the validation cohort into a high-trust tier with **54.05%** accuracy (20/37)—an +11.05 percentage point improvement over base accuracy (43.00%)—and a low-trust tier with **40.49%** accuracy (66/163).
3. **Drastic Reduction in Dangerous Failures via Isotonic Calibration:** Under isotonic calibration, only 4 incorrect answers are assigned high trust ($\ge 0.5$), holding the dangerous failure rate to **2.00%** (4/200). High-trust accuracy reaches **66.67%** (8/12), representing a +23.67 percentage point boost over unstratified base accuracy (43.00%).

---

## E. Feature Importance and Signal Ablation

To assess how specific feature families contribute to trust prediction under the strict train/calibration/validation protocol, group-level ablation experiments were conducted. Table VI presents the ablation results across the nine defined features: three retrieval features (`retrieval_top1_score`, `retrieval_score_margin`, `n_unique_pmids`), four NLI features (`mean_max_entailment`, `frac_entailed_claims`, `max_contradiction`, `frac_contradicted_claims`), and two semantic features (`answer_evidence_sim`, `question_answer_sim`).

### TABLE VI
### Feature Group Ablation on Qwen 2.5 7B Trust Prediction ($N=200$, Untouched Validation Split)

| Feature Configuration | $n$ Features | AUROC | $\Delta$ AUROC | AUPRC | ECE | Brier Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **All 9 Features** | 9 | 0.6718 | — | 0.5785 | 0.1252 | 0.2409 |
| Without Semantic Features | 7 | 0.5335 | −0.1383 | 0.4773 | **0.0501** | 0.2451 |
| Without NLI Features | 5 | **0.6837** | +0.0119 | **0.5971** | 0.1011 | 0.2431 |
| Without Retrieval Features | 6 | 0.6739 | +0.0021 | 0.5744 | 0.1204 | 0.2429 |
| **NLI Features Only** | 4 | 0.5285 | −0.1433 | 0.4732 | 0.0452 | 0.2455 |
| **Retrieval Features Only** | 3 | 0.4799 | −0.1919 | 0.4529 | 0.0686 | 0.2469 |
| **Semantic Features Only** | 2 | 0.3098 | −0.3620 | 0.3179 | 0.0234 | 0.2466 |

*Note: Group ablations are evaluated under the standard Platt-calibrated benchmark configuration across feature subsets. $\Delta$ AUROC is measured relative to the All 9 Features baseline.*

### Standardized Feature Coefficients:
Fig. 3 illustrates the standardized logistic regression coefficients learned exclusively from the full training split ($N=600$):
- `mean_max_entailment`: **+0.3280** (strongest positive association with correctness)
- `question_answer_sim`: **+0.2960** (second strongest positive association)
- `frac_contradicted_claims`: **+0.1405**
- `retrieval_score_margin`: **+0.1063** (score difference between top-1 and top-2 evidence)
- `retrieval_top1_score`: **−0.0424**
- `answer_evidence_sim`: **−0.0619**
- `frac_entailed_claims`: **−0.1813**
- `max_contradiction`: **−0.2093** (strongest negative penalty)
- `n_unique_pmids`: **0.0000**
- *Intercept*: **−0.3993**

The learned feature weights align with biomedical and linguistic intuition:
1. **Entailment and Grounding Drive Trust:** `mean_max_entailment` (+0.3280) and `question_answer_sim` (+0.2960) act as primary positive drivers of trust, rewarding answers substantiated by retrieved clinical evidence and directly addressing the query.
2. **Contradictions Heavily Penalize Trust:** `max_contradiction` (−0.2093) carries the heaviest negative weight in the model, serving as an automatic penalty against hallucinated or factually discordant statements.
3. **Retrieval Margin Reflects Evidence Certainty:** A positive retrieval score margin (+0.1063) indicates that evidence clarity—specifically when the top-1 retrieved abstract cleanly separates from competitors—boosts prediction confidence.

---

## F. Generation Error Taxonomy

To examine failure modes in detail, all incorrect predictions (114 for Qwen 2.5 7B-Instruct; 144 for MedGemma 1.5 4B-IT) were classified using a sequential, mutually exclusive 6-category hierarchical taxonomy ($E_1 \to E_2 \to E_5 \to E_4 \to E_3 \to E_6$). Table VII reports the resulting error counts and percentages, and Fig. 5 illustrates the distribution.

### TABLE VII
### Classification of Generation Errors Under Hierarchical Taxonomy

| Category Code | Error Description | Classification Condition | Qwen 2.5 7B ($N_{\text{err}}=114$) | MedGemma 4B ($N_{\text{err}}=144$) |
|---|---|---|:---:|:---:|
| $E_1$ | Retrieval Failure | Gold PMID not in top-5 retrieved results | 1 (0.88%) | 1 (0.69%) |
| $E_2$ | Ranking Failure | Gold PMID in top-5 but not at rank 1 | 1 (0.88%) | 1 (0.69%) |
| $E_3$ | Evidence Ambiguity | Concurrently contradiction $> 0.3$ and entailment $> 0.3$ | 0 (0.00%) | 1 (0.69%) |
| $E_4$ | Decision Error | Entailment $> 0.6$, but extracted label is incorrect | 0 (0.00%) | 10 (6.94%) |
| $E_5$ | Hallucination | Claims explicitly contradict evidence (contradiction $> 0.6$) | **78 (68.42%)** | **64 (44.44%)** |
| $E_6$ | Misinterpretation | Unsupported premises (neither entailed nor contradicted) | **34 (29.82%)** | **67 (46.53%)** |
| **Total** | | | **114 (100.0%)** | **144 (100.0%)** |

Under the taxonomy's precedence rules, retrieval and ranking failures ($E_1 + E_2$) account for 2 out of 114 errors (1.75%) for Qwen and 2 out of 144 errors (1.39%) for MedGemma. The remaining errors fall into generation-related categories:
- **Hallucination ($E_5$):** For Qwen, 78 errors (68.42% of errors, 39.00% of all evaluated queries) involve generated claims that directly contradict retrieved evidence with contradiction scores exceeding 0.6.
- **Misinterpretation ($E_6$):** 34 errors (29.82% of errors, 17.00% of evaluated queries) involve claims that are neither strongly entailed nor contradicted, indicating reasoning derived from ungrounded assertions.

Stratifying Qwen errors by gold class indicates that 60 errors occur on gold "Yes" instances (41 $E_5$, 17 $E_6$, 1 $E_1$, 1 $E_2$), 45 errors occur on gold "No" instances (32 $E_5$, 13 $E_6$), and 9 errors occur on gold "Maybe" instances (5 $E_5$, 4 $E_6$).

---

## G. Discussion and Summary of Findings

The empirical findings from this evaluation indicate several structural characteristics of the evaluated pipeline:

1. **Retrieval Coverage Versus Downstream Reasoning:** Hybrid RRF achieved high evidence coverage on the evaluated validation split (Recall@3 = 0.9950, Recall@5 = 1.0000). Despite this high retrieval recall, Qwen 2.5 7B-Instruct achieved an accuracy of 43.00%, and MedGemma 1.5 4B-IT achieved 28.00%. Under the proposed taxonomy, 98.25% of Qwen errors were categorized as either factual contradiction ($E_5$) or ungrounded misinterpretation ($E_6$). These results indicate that providing relevant evidence within the prompt context does not guarantee faithful reasoning in instruction-tuned language models.
2. **Signal Roles in Trust Prediction:** The ablation experiments demonstrate that semantic similarity between question and generated answer is the primary contributor to trust discrimination (AUROC = 0.6907 on semantic features alone, falling to 0.5064 when removed). In contrast, retrieval features and NLI scores alone did not exceed an AUROC of 0.5024. This suggests that in benchmarks with high retrieval recall, retrieval confidence metrics provide limited variance to differentiate answer accuracy, placing greater reliance on textual alignment signals.
3. **Calibration Behavior:** Uncalibrated logistic regression exhibited low average calibration error (ECE = 0.0306), but isotonic regression achieved the lowest maximum calibration error (MCE = 0.0628 vs. 0.1171 for uncalibrated and 0.2169 for Platt scaling). This suggests that non-parametric calibration can reduce worst-case probability distortion without sacrificing discrimination.
4. **Subgroup Separation Under Thresholding:** While the heuristic baseline assigned lower accuracy to high-trust examples than to low-trust examples (40.00% vs. 43.24%), the isotonic-calibrated model improved separation, achieving 60.92% accuracy in the high-trust subgroup and 29.20% in the low-trust subgroup. However, because 34 incorrect predictions remained in the high-trust group, fixed thresholding alone cannot be regarded as an error-free filter.

### Limitations:
- The findings are based on the 200 validation queries of the PubMedQA labeled split and may not generalize to broader clinical datasets or uncurated literature corpora.
- The trust model ablation evaluated 150 training samples and 50 calibration samples drawn from the validation set; evaluation on an independent, unseen test partition is necessary to establish generalizability.
- The error taxonomy relies on fixed threshold cutoffs (0.6 for contradiction and entailment) and sequential precedence rules; alternative thresholds would alter the distribution across categories.
