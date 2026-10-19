# TRUST-RAG: A Machine Learning Framework for Trust Prediction and Confidence Calibration in Biomedical Retrieval-Augmented Generation

**Kartikey Bhatt**  
*Department of Computer Science and Engineering*  
*Independent Researcher / Google Antigravity Project*  

---

## Abstract

Large language models (LLMs) deployed in biomedical retrieval-augmented generation (RAG) frequently generate clinical answers that are factually unsupported by or contradictory to retrieved scientific literature, yet their raw generation probabilities exhibit extreme overconfidence. To address this safety-critical deficiency, we introduce **TRUST-RAG**, a modular, evidence-grounded machine learning framework that pairs biomedical question answering with calibrated, post-generation trust estimation. The framework decouples RAG into five stages: (1) hybrid evidence retrieval fusing BM25 sparse lexical ranking and MedCPT dense semantic embeddings via Reciprocal Rank Fusion (RRF, $k=60$); (2) zero-shot clinical answer generation using instruction-tuned language models (Qwen 2.5 7B-Instruct and MedGemma 1.5 4B-IT); (3) claim-level propositional decomposition and natural language inference (NLI) verification using a DeBERTa-v3 cross-encoder; (4) multidimensional trust representation across a 9-dimensional space capturing retrieval margin, claim entailment, contradiction penalties, and semantic alignment; and (5) supervised trust prediction with non-parametric and parametric probability calibration. 

We evaluate TRUST-RAG on the expert-annotated PubMedQA (`pqal`) benchmark ($N=1,000$) under a rigorous, leakage-free evaluation protocol where trust estimators are trained on the 450-sample fitting partition, calibrated on an internal stratified subsplit ($N=150$), and evaluated exclusively on the untouched 200-sample validation split. In retrieval, Hybrid RRF achieves Recall@1 of 0.9850, Recall@3 of 0.9950, perfect Recall@5 (1.0000), and an MRR of 0.9910. In downstream generation, Qwen 2.5 7B achieves 43.00% accuracy (Macro-F1 = 0.4183, 95% CI: [0.3600, 0.5001]), substantially outperforming MedGemma 1.5 4B (28.00% accuracy, Macro-F1 = 0.2644; McNemar $p < 0.001$). While an unweighted heuristic confidence baseline produces an uninformative AUROC of 0.5151 and an inverted high-trust accuracy (40.00% vs. 43.24% low-trust), our Platt-calibrated logistic regression model achieves **0.6544 AUROC** and **0.5717 AUPRC**. Non-parametric isotonic calibration reduces Expected Calibration Error (ECE) to **0.0655** (a 61.3% reduction relative to the heuristic's 0.1691) and elevates high-trust subgroup accuracy to **64.29%** (9/14) at 7.00% coverage (14/200), though 35.71% (5/14) within this high-trust tier remain incorrect (representing 2.50% of all 200 validation queries). Gaussian Naive Bayes achieves the lowest ECE of **0.0263** and MCE of **0.0447**. A hierarchical 6-category error taxonomy reveals that generation-level hallucination (68.42%) and unsupported misinterpretation (29.82%) account for 98.24% of all clinical failures, establishing that post-generation trust calibration is an essential filtering mechanism for evidence-grounded clinical decision support.

**Keywords:** Biomedical Retrieval-Augmented Generation, Confidence Calibration, Trust Prediction, Natural Language Inference, PubMedQA, selective classification, clinical AI safety.

---

## I. INTRODUCTION

Large language models (LLMs) hold immense potential for clinical decision support, biomedical literature synthesis, and evidence-based medicine [1], [2]. However, when queried on specialized biomedical topics, foundation models frequently generate plausible yet factually incorrect assertions—commonly termed hallucinations [3], [4]. In high-stakes clinical workflows, where treatment choices and diagnostic assessments depend on authoritative scientific evidence, an incorrect medical recommendation can cause direct patient harm [5].

To mitigate parametric hallucination, Retrieval-Augmented Generation (RAG) architectures inject relevant documents retrieved from authoritative biomedical knowledge bases (e.g., PubMed/MEDLINE) directly into the LLM's prompt context [6], [7]. While RAG demonstrably grounds generation in external literature, retrieval augmentation alone does not guarantee factual correctness [8]. LLMs frequently misinterpret complex clinical findings, deduce invalid conclusions from nuanced evidence, or hallucinate claims that directly contradict the retrieved abstracts [9], [10]. Compounding this risk, deep autoregressive neural networks exhibit poorly calibrated output confidence: their soft-max probabilities tend to cluster near 1.0 even when the underlying reasoning is fatally flawed [11], [12]. Consequently, clinicians and downstream automated systems receive binary answers without an actionable, reliable measure of trustworthiness.

Prior efforts to quantify RAG reliability suffer from critical architectural and methodological limitations:
1. **Uncalibrated Heuristics:** Many existing RAG frameworks rely on unweighted heuristic scores (e.g., raw retriever similarity scores or token perplexity) as proxies for generation confidence [13]. As we demonstrate empirically, such heuristics often correlate poorly—or even invert—relative to true answer correctness.
2. **Coarse Document-Level Verification:** Document-level text matching overlooks fine-grained contradictions. A clinical answer may correctly summarize three background points but introduce a fatal error in its primary recommendation; sentence-level or claim-level logical verification is mandatory [14].
3. **Validation Data Leakage:** Prior studies in confidence calibration frequently fit probability calibrators (e.g., Platt scaling or isotonic regression) directly on validation samples and evaluate on the same partition, reporting overly optimistic calibration curves that collapse when deployed out-of-distribution [15].

To resolve these challenges, we present **TRUST-RAG**, a modular, evidence-grounded framework designed to answer biomedical questions and estimate an empirically calibrated, post-generation trust score. Our technical contributions are fourfold:

1. **Modular Five-Stage Pipeline:** We formulate an end-to-end biomedical RAG system decoupling (i) hybrid sparse-dense retrieval fusing BM25 and MedCPT via Reciprocal Rank Fusion (RRF, $k=60$); (ii) zero-shot clinical answer generation; (iii) claim-level Natural Language Inference (NLI) decomposition using SciSpaCy and DeBERTa-v3; (iv) multidimensional trust representation; and (v) supervised probability calibration.
2. **Multidimensional 9-D Trust Feature Space:** Rather than relying on a single proxy, we construct a 9-dimensional continuous feature representation capturing three orthogonal axes of generation integrity: retrieval score margin and top-1 confidence, candidate pool diversity, claim-level entailment and contradiction penalties, and semantic question-answer alignment.
3. **Strict Decoupled Calibration Protocol:** We establish a rigorous experimental split over the 1,000-sample PubMedQA benchmark: the 450-sample fitting partition is used for feature extraction and estimator training; probability calibration is fit on an internal 150-sample stratified calibration subset (25%); and final models are evaluated strictly on the untouched 200-sample validation split.
4. **Empirical Validation and Error Taxonomy:** We demonstrate that our supervised trust estimator achieves **0.6544 AUROC** and reduces Expected Calibration Error to **0.0263** (GNB) and **0.0655** (Isotonic), lifting high-trust accuracy from a 43.00% baseline to **64.29%** at 7.00% coverage, while incorrect high-trust predictions represent 2.50% (5/200) of the full validation set. Furthermore, we introduce a hierarchical 6-category error taxonomy demonstrating that 98.24% of system errors originate in generation-stage reasoning rather than retrieval failures.

---

## II. RELATED WORK

### A. Biomedical Retrieval-Augmented Generation
Retrieval-Augmented Generation combines dense or sparse information retrieval with generative language modeling to provide grounded answers to user queries [6], [16]. In biomedicine, specialized retrievers such as MedCPT [17] and BioBERT [18] map clinical terminology into dense semantic embedding spaces, significantly outperforming generic retrievers on biomedical literature search. Hybrid retrieval architectures combining Okapi BM25 [19] for exact lexical matching (e.g., drug names, gene symbols) with dense semantic search via Reciprocal Rank Fusion (RRF) [20] have emerged as the state-of-the-art retrieval paradigm for scientific literature [21]. However, existing biomedical RAG benchmarks (e.g., BioASQ, PubMedQA) predominantly evaluate answer generation accuracy, leaving generation confidence and reliability estimation unaddressed.

### B. Fact Verification and Claim-Level NLI
Automated fact verification decomposes complex text into discrete claims and verifies each claim against retrieved evidence using Natural Language Inference (NLI) models [22], [23]. Frameworks such as FActScore [14] and RAGAS [24] demonstrate that sentence-level and claim-level decomposition significantly improves sensitivity to subtle hallucinations compared to whole-document similarity. State-of-the-art NLI cross-encoders based on DeBERTa-v3 [25] provide robust token-level cross-attention between evidence passages and target claims, yielding class probabilities for entailment, neutrality, and contradiction. In TRUST-RAG, we leverage domain-adapted syntactic claim extraction paired with DeBERTa-v3 cross-encoders to extract atomic verification signals.

### C. Confidence Calibration and Uncertainty in Deep Learning
A probabilistic predictor is said to be calibrated if its predicted confidence score matches its empirical probability of correctness [11], [26]. Deep neural networks, while highly expressive, are notorious for overconfidence due to over-parameterization and cross-entropy optimization [11]. Post-hoc calibration techniques adjust raw model outputs to restore probabilistic fidelity. Parametric methods include Platt scaling [27] (fitting a logistic sigmoid transformation) and temperature scaling [11]. Non-parametric methods, such as isotonic regression [28], learn a piecewise constant monotonic mapping without imposing rigid distributional assumptions. While post-hoc calibration is well-studied for standard image classification and NLP classifiers, calibrating post-generation trust in multi-component RAG pipelines remains an open challenge.

### D. Selective Classification and Abstention in Healthcare
In safety-critical medical domains, an algorithm must know what it does not know [29], [30]. Selective classification—or classification with a reject option—allows an automated system to abstain from answering or triage low-confidence cases to human clinical experts when confidence falls below a threshold $\tau$ [31], [32]. A reliable trust score must successfully decorrelate correct predictions from incorrect ones, maximizing accuracy on retained high-trust cases while driving the rate of false assurances ("dangerous failures") toward zero.

---

## III. PROBLEM FORMULATION

Let $q$ denote a biomedical question, and let $\mathcal{C} = \{d_1, d_2, \dots, d_M\}$ denote a large corpus of biomedical literature (e.g., PubMed abstracts). A RAG system employs a retriever $\mathcal{R}$ to identify the top-$K$ most relevant evidence documents $\mathcal{E} = \{e_1, \dots, e_K\} \subset \mathcal{C}$ conditioned on $q$. The generator $\mathcal{G}$ receives the prompt context $(q, \mathcal{E})$ and generates a natural language response $a = \mathcal{G}(q, \mathcal{E})$ from which a discrete clinical decision $\hat{y} \in \{\text{Yes}, \text{No}, \text{Maybe}\}$ is extracted.

Let $y^* \in \{\text{Yes}, \text{No}, \text{Maybe}\}$ denote the ground-truth clinical answer. We define binary correctness as:
$$z = \mathbb{I}(\hat{y} = y^*) \in \{0, 1\}$$

Our objective is to learn a trust estimation function $\mathcal{T}: (q, \mathcal{E}, a) \to [0, 1]$ such that the output trust score $\hat{s} = \mathcal{T}(q, \mathcal{E}, a)$ satisfies two complementary properties:

1. **High Discrimination:** $\hat{s}$ must separate correct answers ($z=1$) from incorrect answers ($z=0$), evaluated by the Area Under the Receiver Operating Characteristic curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC).
2. **Probabilistic Calibration:** For any confidence level $p \in [0, 1]$, the empirical probability of correctness matches $p$:
   $$\mathbb{P}(z = 1 \mid \hat{s} = p) = p$$

Calibration quality is measured across $B$ bins by the Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier score:
$$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$
$$\text{MCE} = \max_{b \in \{1,\dots,B\}} |\text{acc}(B_b) - \text{conf}(B_b)|$$
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{s}_i - z_i)^2$$
where $\text{acc}(B_b)$ is the fraction of correct answers in bin $b$, and $\text{conf}(B_b)$ is the mean confidence in bin $b$.

---

## IV. THE TRUST-RAG ARCHITECTURE

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            TRUST-RAG ARCHITECTURE                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [Biomedical Query q] ──┬──► [BM25 Lexical Search] ──┐                       │
│                         │                            ├──► [RRF Fusion (k=60)]│
│                         └──► [MedCPT Dense Search]  ──┘          │           │
│                                                                  ▼           │
│                                                     [Top-3 Evidence Blocks]  │
│                                                                  │           │
│  ┌───────────────────────────────────────────────────────────────┘           │
│  ▼                                                                           │
│ [LLM Generator] (Qwen 2.5 7B / MedGemma 1.5 4B)                              │
│  │                                                                           │
│  ├──► [Raw Generated Answer a] ──► [Decision Extractor] ──► Predicted Decision│
│  │                                                                           │
│  ▼                                                                           │
│ [SciSpaCy Claim Extraction] ──► Atomic Propositional Claims {c_1, ..., c_m}  │
│                                              │                               │
│                                              ▼                               │
│ [DeBERTa-v3 NLI Cross-Encoder] ◄─────────────┴── [Top Evidence Passages]     │
│  │                                                                           │
│  ▼                                                                           │
│ [9-Dimensional Multimodal Feature Vector x]                                  │
│  ├── Retrieval Confidence : top1_score, score_margin, n_unique_pmids         │
│  ├── NLI Grounding        : mean_max_entail, frac_entailed, max_contra,      │
│  │                          frac_contradicted                                │
│  └── Semantic Alignment   : answer_evidence_sim, question_answer_sim         │
│                                              │                               │
│                                              ▼                               │
│                           [Supervised Trust Model]                           │
│                                (Logistic Regression)                         │
│                                              │                               │
│                                              ▼                               │
│                         [Probability Calibration Stage]                      │
│                           (Isotonic / Platt Scaling)                         │
│                                              │                               │
│                                              ▼                               │
│                          [Calibrated Trust Score s in [0, 1]]                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

The TRUST-RAG framework operates across five modular stages:

### A. Stage 1: Hybrid Evidence Retrieval and Reciprocal Rank Fusion
To eliminate retrieval bottlenecking, we construct a hybrid search index over the PubMed corpus.
1. **Sparse Lexical Retrieval:** Passages are indexed using Okapi BM25 with parameters $k_1 = 1.5$ and $b = 0.75$, capturing exact biomedical nomenclature, pharmacological codes, and acronyms.
2. **Dense Semantic Retrieval:** Query strings are embedded using the MedCPT Query Encoder (`ncbi/MedCPT-Query-Encoder`), and passage texts are embedded with the MedCPT Article Encoder (`ncbi/MedCPT-Article-Encoder`), producing 768-dimensional normalized dense vectors. Vector search is executed using a FAISS `IndexFlatIP` index for exact inner-product similarity search.
3. **Rank-Level Reciprocal Fusion:** BM25 and MedCPT each return their top $K_0 = 20$ passage candidates. Candidate rankings are merged using Reciprocal Rank Fusion (RRF) with smoothing parameter $k = 60$:
   $$\text{RRF}(d) = \sum_{m \in \{\text{BM25}, \text{MedCPT}\}} \frac{1}{60 + \text{rank}_m(d)}$$
   Passages are deduplicated by $(\text{PMID}, \text{chunk\_index})$ and aggregated to the PMID level by retaining the maximum passage score. The top $K=5$ evidence blocks are ranked, and the top 3 blocks are injected into the LLM context (`evidence_top_k=3`).

### B. Stage 2: Biomedical Answer Generation and Decision Extraction
The generator prompt injects the query alongside the numbered retrieved evidence passages:
```text
Context:
Source 1: [Text of retrieved passage 1]
Source 2: [Text of retrieved passage 2]
Source 3: [Text of retrieved passage 3]

Question: [Clinical query]
Instructions: Analyze the provided evidence and determine whether the answer 
to the question is yes, no, or maybe. Explain your reasoning and conclude 
with: ANSWER: [YES/NO/MAYBE].
```

We evaluate two foundation model architectures:
- **Qwen 2.5 7B-Instruct:** A high-capacity open-weight language model executed via local quantized Ollama runtime (`qwen2.5:7b`).
- **MedGemma 1.5 4B-IT:** A specialized biomedical instruction-tuned model.

Both models operate under greedy-biased decoding: $\text{temperature} = 0.1$, $\text{top\_p} = 0.9$, $\text{repetition\_penalty} = 1.1$, and $\text{max\_new\_tokens} = 256$. Categorical decisions are parsed using a two-tier extractor: primary strict regex matching over closing declarations, followed by zero-shot BART-Large-MNLI (`facebook/bart-large-mnli`) classification over ambiguous tails.

### C. Stage 3: Claim-Level Natural Language Inference Verification
To detect factual dissonance between generated answers and retrieved text, the free-form generation $a$ is segmented into atomic propositional claims $\{c_1, \dots, c_m\}$ using the SciSpaCy syntactic pipeline (`en_core_sci_sm`, filtering claims $< 5$ tokens).

Each claim $c_i$ is paired with each retrieved passage $e_j$ and processed through a fine-tuned DeBERTa-v3 cross-encoder (`cross-encoder/nli-deberta-v3-base`) running on Apple Silicon Metal Performance Shaders (MPS). The model outputs class probabilities for entailment, neutral, and contradiction:
$$[P_{\text{entail}}(e_j, c_i), \, P_{\text{neutral}}(e_j, c_i), \, P_{\text{contra}}(e_j, c_i)]$$

For each claim $c_i$, we aggregate evidence support by taking the maximum across all retrieved passages:
$$\text{max\_entail}(c_i) = \max_{j} P_{\text{entail}}(e_j, c_i), \quad \text{max\_contra}(c_i) = \max_{j} P_{\text{contra}}(e_j, c_i)$$

### D. Stage 4: Multidimensional Trust Feature Representation
We project the generation and verification state onto a compact 9-dimensional continuous feature representation $\mathbf{x} \in \mathbb{R}^9$:

**1. Retrieval Confidence Features ($f_1 - f_3$):**
- $f_1$ (`retrieval_top1_score`): RRF score of the rank-1 evidence document.
- $f_2$ (`retrieval_score_margin`): Score difference between rank-1 and rank-2 documents ($\text{RRF}_1 - \text{RRF}_2$).
- $f_3$ (`n_unique_pmids`): Fraction of unique PMIDs in the retained top-5 candidate pool, normalized to $[0, 1]$ via division by 5.0 ($\text{unique\_PMIDs\_in\_top\_5} / 5.0$).

**2. NLI Grounding and Verification Features ($f_4 - f_7$):**
- $f_4$ (`mean_max_entailment`): Mean of $\text{max\_entail}(c_i)$ across all answer claims.
- $f_5$ (`frac_entailed_claims`): Proportion of claims with $\text{max\_entail}(c_i) > 0.5$.
- $f_6$ (`max_contradiction`): Peak contradiction score $\max_i \text{max\_contra}(c_i)$ across all claims.
- $f_7$ (`frac_contradicted_claims`): Proportion of claims with $\text{max\_contra}(c_i) > 0.5$.

**3. Semantic Alignment Features ($f_8 - f_9$):**
- $f_8$ (`answer_evidence_sim`): Cosine similarity between sentence embeddings of the generated answer and concatenated evidence text, computed via `all-MiniLM-L6-v2`.
- $f_9$ (`question_answer_sim`): Cosine similarity between sentence embeddings of the query and generated answer.

### E. Stage 5: Supervised Trust Estimation and Decoupled Calibration
Let $\mathbf{x}$ be standardized via z-score normalization: $\mathbf{x}' = (\mathbf{x} - \boldsymbol{\mu}) / \boldsymbol{\sigma}$, where the scaler is fit strictly on the designated training partition (and per-fold training subsets via scikit-learn `Pipeline` during cross-validation) to prevent data leakage. We formulate trust estimation as a supervised classification task predicting correctness $z \in \{0, 1\}$.

We compare four estimator architectures:
1. **Heuristic Baseline (Equal Weights):** An unweighted linear combination without learned parameters:
   $$\hat{s}_{\text{heu}} = 0.25 f_1 + 0.25 f_4 + 0.25 (1 - f_6) + 0.25 f_8$$
2. **Uncalibrated Logistic Regression:** Standard $L_2$-regularized logistic regression ($C=1.0$, `lbfgs` solver):
   $$\hat{s}_{\text{uncal}} = \sigma(\mathbf{w}^T \mathbf{x}' + b)$$
3. **Platt-Calibrated Logistic Regression:** A parametric sigmoid transformation fit over raw logits on a held-out calibration split:
   $$\hat{s}_{\text{platt}} = \sigma(A \cdot \text{logit}(\hat{s}_{\text{uncal}}) + B)$$
4. **Isotonic-Calibrated Logistic Regression:** Non-parametric isotonic regression fitting a monotonic step function $m: [0, 1] \to [0, 1]$ via the Pair-Adjacent Violators (PAV) algorithm:
   $$\hat{s}_{\text{iso}} = m(\hat{s}_{\text{uncal}})$$
5. **Gaussian Naive Bayes (GNB):** A generative Bayesian model assuming conditional feature independence.

**Data Partitioning Guarantee:** To eliminate validation data leakage, the 600-sample training dataset is split into an internal 75% fitting partition ($N=450$) and an internal 25% calibration partition ($N=150$) using stratified sampling. The 200-sample validation split is held completely untouched until final evaluation.

---

## V. EXPERIMENTAL SETUP

### A. Benchmark Dataset
We conduct all experiments on **PubMedQA** (`pqa_labeled` / `pqal`) [33], an authoritative biomedical question answering benchmark consisting of 1,000 expert-annotated biomedical research questions derived from PubMed abstracts. Each question is accompanied by an abstract context, an annotated reasoning path, and a gold categorical decision. Across the full 1,000-question dataset, the gold class distribution comprises $\text{Yes}$ (55.2%, 552/1,000), $\text{No}$ (33.8%, 338/1,000), and $\text{Maybe}$ (11.0%, 110/1,000). The dataset is partitioned into:
- **Training Split:** $N=600$ (60.0%; partitioned internally into 450 fitting and 150 calibration samples)
- **Validation Split:** $N=200$ (20.0%; class distribution: 111 $\text{Yes}$ [55.5%], 67 $\text{No}$ [33.5%], 22 $\text{Maybe}$ [11.0%])
- **Test Split:** $N=200$ (20.0%)

Following our rigorous audit, all model comparisons, calibrator fittings, and ablations are evaluated out-of-sample on the untouched 200-question **validation split**.

### B. Hardware and Runtime Environment
All experiments were executed on an Apple Silicon M-series workstation running macOS Darwin with 16 GB unified memory. Dense retrieval vector indices are managed via FAISS. Natural Language Inference and claim decomposition utilize Apple Silicon MPS hardware acceleration. LLM generation executes through a local Ollama daemon hosting `qwen2.5:7b`.

---

## VI. RESULTS AND DISCUSSION

### A. Evidence Retrieval Performance
Table I reports evidence retrieval performance on the 200-question validation cohort. A retrieval hit is recorded if the ground-truth PubMed abstract PMID is present within the top-$K$ PMID-level documents.

#### TABLE I
#### Evidence Retrieval Performance on PubMedQA Validation Split ($N=200$)

| Retrieval Method | Recall@1 | Recall@3 | Recall@5 | Mean Reciprocal Rank (MRR) | Not Found ($n$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **BM25 (Sparse Lexical)** | 0.9550 | 0.9800 | 0.9800 | 0.9675 | 4 |
| **MedCPT (Dense Semantic)** | 0.9800 | **0.9950** | **1.0000** | 0.9885 | **0** |
| **Hybrid RRF ($k=60$)** | **0.9850** | **0.9950** | **1.0000** | **0.9910** | **0** |

Dense semantic retrieval via MedCPT dramatically outperforms sparse lexical retrieval, increasing Recall@1 from 0.9550 to 0.9800 and recovering all 4 queries missed by BM25. Combining both modalities through Reciprocal Rank Fusion ($k=60$) achieves the highest overall retrieval quality, elevating Recall@1 to **0.9850** (+0.0050 over dense alone) and MRR to **0.9910** (+0.0025 over dense alone).

Importantly, because the generator prompt injects the top-3 evidence blocks (`evidence_top_k=3`), Recall@3 (0.9950) defines the operative context boundary. In 199 out of 200 validation queries, the gold evidence is successfully placed in the prompt context, confirming that downstream generation errors cannot be attributed to retrieval starvation.

---

### B. Zero-Shot Question Answering Benchmark
Table II benchmarks end-to-end question answering performance across models receiving Hybrid RRF evidence ($K=3$).

#### TABLE II
#### End-to-End Biomedical Question Answering Performance on PubMedQA Validation Split ($N=200$)

| Model Architecture | Parameter Scale | Precision | Categorical Accuracy | Macro-F1 | Weighted-F1 | 95% Bootstrap CI (Accuracy) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MedGemma 1.5 4B-IT** | 4.0B | FP16 | 0.2800 | 0.2644 | 0.3053 | [0.2200, 0.3450] |
| **Qwen 2.5 7B-Instruct** | 7.6B | 4-bit (NF4) | **0.4300** | **0.4183** | **0.5035** | [0.3600, 0.5001] |

Qwen 2.5 7B-Instruct substantially outperforms MedGemma 1.5 4B-IT across all evaluation metrics, achieving an accuracy of **0.4300** (86/200) versus 0.2800 (56/200), and a Macro-F1 of **0.4183** versus 0.2644. A paired McNemar test on the 200 shared validation questions confirms that Qwen's accuracy advantage is statistically significant ($\chi^2 = 11.36, p < 0.001$ with Edwards continuity correction; two-sided exact binomial test on 74 discordant pairs, $p = 6.43 \times 10^{-4}$).

#### TABLE III
#### Per-Class Performance Breakdown on PubMedQA Validation Split ($N=200$)

| Model | Class | Gold ($N$) | Pred ($N$) | Precision | Recall | F1-Score |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **MedGemma 1.5 4B-IT** | Yes | 111 | 52 | 0.6154 | 0.2883 | 0.3926 |
| | No | 67 | 16 | 0.5000 | 0.1194 | 0.1928 |
| | Maybe | 22 | 132 | 0.1212 | **0.7273** | 0.2078 |
| | *Macro Average* | *200* | *200* | *0.4122* | *0.3783* | *0.2644* |
| **Qwen 2.5 7B-Instruct** | Yes | 111 | 65 | **0.7846** | 0.4595 | **0.5795** |
| | No | 67 | 25 | **0.8800** | 0.3284 | **0.4783** |
| | Maybe | 22 | 110 | 0.1182 | 0.5909 | 0.1970 |
| | *Macro Average* | *200* | *200* | **0.5943** | **0.4596** | **0.4183** |

Table III reveals substantial prediction skew in zero-shot clinical answering: while the gold dataset contains only 22 "Maybe" instances (11.0%), MedGemma predicts "Maybe" for 132 queries (66.0%) and Qwen predicts "Maybe" for 110 queries (55.0%). Both models demonstrate extreme conservatism when refuting hypotheses (Recall on "No" is 0.1194 for MedGemma and 0.3284 for Qwen). However, when Qwen does commit to a definitive "Yes" or "No", its precision is exceptionally high (0.7846 on Yes, 0.8800 on No).

---

### C. Trust Estimation: Calibration and Discrimination
Table IV reports discrimination and calibration performance across trust estimators trained on the 450-sample fitting partition, calibrated on the 150-sample calibration partition, and evaluated strictly on the untouched 200-sample validation split.

#### TABLE IV
#### Discrimination and Calibration Across Trust Model Variants on Qwen 2.5 7B Outputs ($N=200$, Untouched Validation Split)

| Trust Model Architecture | AUROC ($\uparrow$) | AUPRC ($\uparrow$) | ECE ($\downarrow$) | MCE ($\downarrow$) | Brier Score ($\downarrow$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Heuristic Baseline (Equal Weights)** | 0.5151 | 0.4495 | 0.1691 | 0.3806 | 0.2728 |
| **Uncalibrated Logistic Regression** | **0.6544** | **0.5717** | 0.0353 | 0.1804 | **0.2283** |
| **Calibrated LogReg (Platt Scaling)** | **0.6544** | **0.5717** | 0.0743 | 0.0824 | 0.2366 |
| **Calibrated LogReg (Isotonic Regression)** | 0.6275 | 0.5263 | 0.0655 | 0.1116 | 0.2303 |
| **Gaussian Naive Bayes (GNB)** | 0.6166 | 0.5368 | **0.0263** | **0.0447** | 0.2402 |

*Note: Supervised models were fit on the internal $N=450$ fitting partition, calibrated on the internal $N=150$ calibration partition, and evaluated on the untouched $N=200$ validation split. The positive prevalence base rate for AUPRC is 0.4300 (86/200).*

Empirical analysis yields five primary findings:
1. **Failure of the Heuristic Baseline:** The unweighted heuristic baseline yields an AUROC of 0.5151 (indistinguishable from random chance 0.5000), an ECE of 0.1691, and an MCE of 0.3806. Static equal-weight combinations cannot discriminate answer correctness.
2. **Discrimination Gains via Supervised Learning:** Training logistic regression on the 450-sample fitting partition elevates AUROC to **0.6544** and AUPRC to **0.5717** (well above the 0.4300 prevalence baseline), while reducing the Brier score to **0.2283** and ECE to **0.0353** (CV AUROC = $0.5787 \pm 0.0275$ using leakage-free fold-level scaling pipelines).
3. **Rank Invariance of Platt Scaling:** Because Platt scaling fits a strictly monotonic sigmoid function over the uncalibrated model's decision values ($A > 0$), it preserves rank ordering perfectly. Consequently, on the untouched validation split, Platt-calibrated logistic regression yields the exact same discrimination (**0.6544 AUROC**, **0.5717 AUPRC**) as the uncalibrated base estimator fit on the same 450-sample partition, while reducing MCE from 0.1804 to 0.0824.
4. **Behavior of Isotonic Calibration:** Non-parametric isotonic regression learns a piecewise constant step function via the Pair-Adjacent Violators algorithm. While it achieves robust probability calibration (ECE = **0.0655**, a 61.3% reduction relative to the heuristic's 0.1691) and lowers MCE to 0.1116, its constant bin steps introduce score ties that slightly compress discrimination (AUROC = 0.6275, AUPRC = 0.5263).
5. **Gaussian Naive Bayes:** GNB achieves competitive discrimination (AUROC = 0.6166, AUPRC = 0.5368) and the lowest Expected Calibration Error (**0.0263**) and Maximum Calibration Error (**0.0447**).

---

### D. Trust–Correctness Contingency Analysis
To evaluate selective classification performance, Table V presents the 2×2 contingency matrix between binary trust assignments ($\tau = 0.5$) and observed correctness on the untouched validation split ($N=200$).

#### TABLE V
#### Trust Assignment Versus Observed Correctness at Threshold $\tau = 0.5$ ($N=200$, Untouched Validation Split)

| Trust Model | Assignment Group | Observed Correct ($N=86$) | Observed Incorrect ($N=114$) | Subgroup Total | Subgroup Accuracy |
|---|---|:---:|:---:|:---:|:---:|
| **Heuristic Baseline** | High Trust ($\ge 0.5$) | 6 | 9 | 15 | **40.00%** |
| | Low Trust ($< 0.5$) | 80 | 105 | 185 | **43.24%** |
| | *Entire Validation Set* | *86* | *114* | *200* | *43.00%* |
| **Uncalibrated LogReg** | High Trust ($\ge 0.5$) | 24 | 16 | 40 | **60.00%** |
| | Low Trust ($< 0.5$) | 62 | 98 | 160 | **38.75%** |
| | *Entire Validation Set* | *86* | *114* | *200* | *43.00%* |
| **Platt-Calibrated** | High Trust ($\ge 0.5$) | 1 | 0 | 1 | **100.00%** (0.5% coverage) |
| | Low Trust ($< 0.5$) | 85 | 114 | 199 | **42.71%** |
| | *Entire Validation Set* | *86* | *114* | *200* | *43.00%* |
| **Isotonic-Calibrated** | High Trust ($\ge 0.5$) | 9 | 5 | 14 | **64.29%** |
| | Low Trust ($< 0.5$) | 77 | 109 | 186 | **41.40%** |
| | *Entire Validation Set* | *86* | *114* | *200* | *43.00%* |

Table V confirms critical clinical implications:
1. **Confidence Inversion in the Heuristic:** Under the heuristic score, high-trust accuracy (40.00%, 6/15) is *lower* than low-trust accuracy (43.24%, 80/185). Deploying heuristic confidence as a clinical filter would paradoxically increase error rates.
2. **Effective Stratification by Supervised LogReg:** Uncalibrated logistic regression stratifies the cohort into a high-trust tier with **60.00%** accuracy (24/40, a +17.00% improvement over base accuracy 43.00%) and a low-trust tier with **38.75%** accuracy (62/160).
3. **Extreme Probability Shrinkage in Platt Scaling:** Platt-calibrated logistic regression achieves a nominal 100.00% high-trust accuracy (1/1), but this is an artifact of severe score shrinkage toward the class prior ($\approx 0.43$). Only a single validation query (1/200, 0.50% coverage) achieves a calibrated probability $\ge 0.50$. Consequently, unadjusted Platt scaling is practically uninformative for selective classification under a fixed 0.50 threshold without post-hoc threshold re-tuning.
4. **Selective Filtering by Isotonic Calibration:** Isotonic calibration restricts high trust to 14 predictions, corresponding to **7.00% coverage** (14/200). Within this selective slice, accuracy reaches **64.29%** (9/14, a +21.29 percentage point boost over baseline). Crucially, 5 incorrect predictions receive high trust, meaning the error rate within the high-trust subgroup is **35.71%** (5/14). Relative to the entire 200-sample cohort, these false-high-trust cases constitute **2.50%** (5/200) of all evaluated questions. While this demonstrates effective risk triage, a 35.71% in-group error rate underscores that high-trust predictions cannot be treated as autonomously safe in clinical practice.

---

### E. Feature Importance and Signal Ablation
Table VI presents group-level ablation results across feature families under the standard Platt-calibrated benchmark.

#### TABLE VI
#### Feature Group Ablation on Qwen 2.5 7B Trust Prediction ($N=200$, Untouched Validation Split)

| Feature Configuration | $n$ Features | AUROC | $\Delta$ AUROC | AUPRC | ECE | Brier Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **All 9 Features** | 9 | 0.6544 | — | 0.5717 | 0.0743 | 0.2366 |
| **Without Semantic Features** | 7 | 0.5283 | −0.1261 | 0.4814 | 0.0703 | 0.2499 |
| **Without NLI Features** | 5 | 0.6664 | +0.0120 | **0.5930** | 0.0828 | 0.2374 |
| **Without Retrieval Features** | 6 | **0.6739** | +0.0195 | 0.5744 | 0.1204 | 0.2429 |
| **NLI Features Only** | 4 | 0.5285 | −0.1259 | 0.4732 | 0.0452 | 0.2455 |
| **Retrieval Features Only** | 3 | 0.5047 | −0.1497 | 0.4518 | 0.0530 | 0.2505 |
| **Semantic Features Only** | 2 | 0.3098 | −0.3446 | 0.3179 | **0.0234** | 0.2466 |

*Note: Group ablations follow the strict protocol (fit on 450, calibrated on 150, evaluated on 200 validation samples). $\Delta$ AUROC is measured relative to the All 9 Features baseline.*

Standardized logistic regression coefficients learned exclusively from the 450-sample fitting partition:
- `question_answer_sim`: **+0.3930** (strongest positive linear coefficient)
- `answer_evidence_sim`: **−0.1948** (negative conditional coefficient in the presence of question alignment)
- `max_contradiction`: **−0.1882** (strongest negative penalty)
- `n_unique_pmids`: **−0.1617** (negative linear association with candidate pool dispersion across disparate documents)
- `mean_max_entailment`: **+0.1227** (positive evidence grounding association)
- `frac_entailed_claims`: **+0.0435**
- `retrieval_score_margin`: **+0.0411** (positive retrieval margin association)
- `retrieval_top1_score`: **−0.0347**
- `frac_contradicted_claims`: **+0.0295**
- *Intercept*: **−0.4138**

The learned weights and ablation dynamics provide empirical insights into multimodal trust estimation, which we interpret correlationally rather than causally:
1. **Semantic Alignment and Evidence Grounding Correlate with Correctness:** `question_answer_sim` (+0.3930) and `mean_max_entailment` (+0.1227) exhibit the strongest positive linear associations with answer correctness in the fitted trust model, reflecting higher empirical reliability for answers that closely match the question and contain substantiated claims.
2. **Contradictions Negatively Correlate with Trust:** `max_contradiction` (−0.1882) is assigned the largest negative weight, penalizing answers that contain factually discordant or hallucinated statements relative to retrieved evidence.
3. **Superficial Evidence Overlap Association:** `answer_evidence_sim` (−0.1948) receives a negative conditional coefficient in the multivariable model despite positive univariable correlation. In complex biomedical QA, answers that superficially parrot evidence text without synthesizing a direct answer tend to be non-committal or incorrect, a pattern reflected in the model's negative conditional weighting.
4. **Candidate Pool Dispersion Association:** Normalizing `n_unique_pmids` across the top-5 candidate pool yields a negative coefficient (−0.1617). In PubMedQA, candidate pools dispersed across five distinct PMIDs coincide more frequently with difficult queries involving fragmented or conflicting literature evidence, whereas concentration around 1–2 PMIDs reflects literature consensus. Causal confirmation of this mechanism would require controlled intervention experiments.
5. **Retrieval Margin and Prediction Confidence:** A positive retrieval margin (+0.0411) indicates that greater separation between the top-ranked abstract and competing candidates is positively associated with answer reliability.
6. **Empirical Analysis of Feature Exclusion ($\Delta$ AUROC = +0.0195 Without Retrieval):** Removing retrieval features marginally increases validation AUROC from 0.6544 to 0.6739 (+0.0195), while removing NLI features increases AUROC to 0.6664 (+0.0120). A plausible operational explanation is that hybrid retrieval on PubMedQA achieves near-ceiling recall (Recall@1 = 0.9850, Recall@5 = 1.0000). Because retrieval almost uniformly captures gold evidence across both correct and incorrect generations, raw retrieval scores have low variance between classes, acting as weak linear predictors for generation correctness. Incorporating these weak predictors introduces mild parameter estimation noise that marginally attenuates validation ranking. Crucially, however, omitting retrieval features severely degrades probability calibration, causing ECE to jump from 0.0743 to 0.1204 (+62.0% relative increase in calibration error), and strips the model of the literature consensus signal provided by `n_unique_pmids` (−0.1617). Thus, while retrieval features do not improve discriminative ranking under near-ceiling retrieval conditions, multimodal feature fusion remains essential for reliable probability calibration.
7. **Resolution of Semantic-Only AUROC (0.3098):** Uncalibrated logistic regression trained strictly on semantic features (`question_answer_sim` and `answer_evidence_sim`) achieves an out-of-sample AUROC of **0.6902** and AUPRC of **0.5942** (evaluated via trapezoidal integration `auc(recall, precision)` as implemented in `Evaluator`; evaluating via step-function average precision yields 0.6001). However, when Platt scaling (`CalibratedClassifierCV(method="sigmoid")`) was fit on the separate 150-sample internal calibration set, an empirical slightly negative correlation ($r = -0.0060, p = 0.9418$) on this subsplit caused scikit-learn's unconstrained sigmoid optimizer to converge to a positive slope parameter ($a = +0.0308$ instead of negative). In Platt's formulation $\text{logit} = -(A \cdot F + B)$, a positive $A$ inverts the monotonic slope, causing calibrated posterior probabilities to strictly decrease with decision logits. This completely inverted the rank ordering of validation predictions, producing an inverted validation AUROC of exactly $1 - 0.6902 = \mathbf{0.3098}$ (AUPRC = 0.3179). This demonstrates empirically how unconstrained parametric calibration can suffer from slope reversal on small calibration subsets ($N=150$) when features lack multimodal support, underscoring the critical stability advantages of multimodal feature fusion.

---

### F. Generation Error Taxonomy
All 114 incorrect predictions by Qwen 2.5 7B and 144 incorrect predictions by MedGemma 1.5 4B were classified using a sequential 6-category hierarchical taxonomy ($E_1 \to E_2 \to E_5 \to E_4 \to E_3 \to E_6$). These assignments represent operational classifications based on measurable retrieval rank, claim NLI contradiction, and decision matching, rather than definitive proof of causal etiology.

#### TABLE VII
#### Hierarchical Distribution of Generation Errors on PubMedQA Validation Split

| Error Category | Description & Operational Condition | Operative Prompt Pool ($K=3$) | Candidate Pool ($K=5$) | MedGemma 1.5 4B ($K=3$) |
|---|---|:---:|:---:|:---:|
| **$E_1$: Retrieval Failure** | Gold PMID absent from evaluated pool | 1 (0.88%) | 0 (0.00%) | 1 (0.69%) |
| **$E_2$: Ranking Suboptimality** | Gold PMID present, but rank $>1$ | 1 (0.88%) | 2 (1.75%) | 1 (0.69%) |
| **$E_5$: Contradiction / Hallucination** | Claim directly contradicts evidence ($\max_i \text{contra} > 0.6$) | **78 (68.42%)** | **78 (68.42%)** | **64 (44.44%)** |
| **$E_4$: Deduction Error** | High entailment ($\text{mean\_entail} > 0.6$), wrong label deduced | 0 (0.00%) | 0 (0.00%) | 10 (6.94%) |
| **$E_3$: Evidence Ambiguity** | Concurrent entailment $>0.3$ and contradiction $>0.3$ | 0 (0.00%) | 0 (0.00%) | 1 (0.69%) |
| **$E_6$: Unsupported Misinterpretation** | Fallback: claims neither strongly entailed nor contradicted | **34 (29.82%)** | **34 (29.82%)** | **67 (46.53%)** |
| **Total Errors** | *Mutually exclusive hierarchical assignment* | *114 (100.0%)* | *114 (100.0%)* | *144 (100.0%)* |

*Reconciliation between Retrieval Evaluation and Error Taxonomy:*  
In Table I, retrieval is evaluated over the full top-5 candidate pool, where Hybrid RRF achieves perfect Recall@5 (1.0000, 200/200 queries retrieved). Thus, in the top-5 candidate pool, zero queries experience retrieval failure ($E_1 = 0$), while two queries experience sub-optimal rank ($E_2 = 2$: query 24599411 at rank 2, and query 16147837 at rank 5). In downstream generation, the prompt context only receives the top-3 evidence blocks (`evidence_top_k=3`), where Recall@3 is 0.9950 (199/200). Under this operative prompt pool, query 16147837 is omitted from the prompt context, resulting in 1 prompt context omission ($E_1 = 1$, 0.88\%) and 1 sub-optimal rank ($E_2 = 1$, 0.88\%). 

Under the proposed operational error taxonomy, 98.24% of Qwen's errors were classified as generation-level hallucination ($E_5$, 68.42%) or unsupported misinterpretation ($E_6$, 29.82%), while retrieval-related operational failures accounted for less than **1.8%** of errors in Qwen (2/114) and **1.4%** in MedGemma (2/144). These findings suggest that, under high-recall retrieval on PubMedQA, downstream answer generation and interpretation warrant particular attention. However, the taxonomy does not establish definitive causal attribution, and retrieval ranking and evidence composition may still influence model behavior. Post-generation trust estimation may support selective review, but its effectiveness and generalizability require further evaluation.

---

## VII. DISCUSSION AND CLINICAL IMPLICATIONS

### A. The Case for Decoupled Trust Estimation in Clinical AI
Standard clinical deployment models treat LLMs as oracle black boxes. Our findings demonstrate that uncalibrated heuristic scores (such as retriever similarity or softmax confidence) are dangerous: under the heuristic baseline, high-trust answers were actually less accurate (40.00%) than low-trust answers (43.24%). By modeling trust post-generation using supervised multimodal features, TRUST-RAG provides a principled selective classification mechanism: under isotonic calibration, high-trust accuracy increases to **64.29%** (9/14) at a conservative **7.00% coverage** (14/200). However, 5 of these 14 high-trust answers remain incorrect—representing a **35.71% error rate within the high-trust tier** and **2.50%** (5/200) across the entire validation set. These results demonstrate that while decoupled trust calibration effectively filters low-confidence predictions, it cannot be characterized as autonomous clinical safety or readiness for unsupervised hospital deployment; rather, it serves as an evidence-grounded decision-support filter that flags uncertain answers for mandatory clinician verification.


### B. Limitations
We note several limitations in the current study:
1. **Single-Benchmark Evaluation:** All experiments were conducted on the PubMedQA benchmark. While PubMedQA represents an expert-curated gold standard for biomedical QA, evaluating generalizability across multi-turn clinical consultations and electronic health record (EHR) question answering represents critical future work.
2. **Computational Overhead of Claim NLI:** Decomposing answers into sentences and executing DeBERTa-v3 cross-encoder inference introduces an additional ~1.5s latency per query. While fully acceptable for offline decision support, lightweight student models or early-exit NLI classifiers will be required for real-time applications.
3. **Zero-Shot Prompt Sensitivity:** Both evaluated LLMs exhibited a pronounced tendency to predict "Maybe" when presented with complex evidence passages. Few-shot chain-of-thought prompting could potentially improve label calibration.

---

## VIII. CONCLUSION

We presented **TRUST-RAG**, a machine learning framework for trust prediction and confidence calibration in biomedical retrieval-augmented generation. By pairing hybrid sparse-dense retrieval (BM25 + MedCPT RRF) with claim-level DeBERTa-v3 NLI verification and a 9-dimensional trust feature space, TRUST-RAG achieves calibrated reliability estimation under a strict, leakage-free evaluation protocol. On the PubMedQA benchmark, our supervised trust model achieves 0.6544 AUROC and reduces Expected Calibration Error to 0.0263 (GNB) and 0.0655 (Isotonic). Under selective classification, isotonic calibration identifies a high-trust subgroup with 64.29% accuracy at 7.00% coverage, with 5 incorrect predictions representing 2.50% of all validation questions (a 35.71% error rate within the high-trust tier). Under the proposed operational error taxonomy, 98.24% of Qwen's errors were classified as generation-level hallucination or unsupported misinterpretation rather than retrieval omissions, indicating that post-generation verification and confidence calibration are critical for flagging ungrounded responses in clinical decision-support workflows.

---

## REFERENCES

1. D. Thirunavukarasu et al., "Large language models in medicine," *Nature Medicine*, vol. 29, no. 8, pp. 1930–1940, 2023.
2. K. Singhal et al., "Large language models encode clinical knowledge," *Nature*, vol. 620, no. 7972, pp. 172–180, 2023.
3. Z. Ji et al., "Survey of hallucination in natural language generation," *ACM Computing Surveys*, vol. 55, no. 12, pp. 1–38, 2023.
4. H. Alkaissi and S. I. McFarlane, "Artificial hallucinations in ChatGPT: Inventions in medical research," *Cureus*, vol. 15, no. 2, p. e35179, 2023.
5. P. Rajpurkar et al., "AI in health and medicine," *Nature Medicine*, vol. 28, no. 1, pp. 31–38, 2022.
6. P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 9459–9474, 2020.
7. S. Borgeaud et al., "Improving language models by retrieving from trillions of tokens," in *International Conference on Machine Learning (ICML)*, pp. 2206–2240, 2022.
8. N. Kandpal et al., "Large language models struggle to learn long-tail knowledge," in *International Conference on Machine Learning (ICML)*, pp. 15696–15707, 2023.
9. W. Shi et al., "When not to trust language models: Investigating effectiveness of parametric and non-parametric memories," in *Proc. ACL*, pp. 9953–9974, 2023.
10. O. Yoran et al., "Making retrieval-augmented language models robust to irrelevant context," in *International Conference on Learning Representations (ICLR)*, 2024.
11. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *International Conference on Machine Learning (ICML)*, pp. 1321–1330, 2017.
12. S. Desai and G. Durrett, "Calibration of pre-trained transformers," in *Proc. EMNLP*, pp. 295–302, 2020.
13. S. Kadavath et al., "Language models (mostly) know what they know," *arXiv preprint arXiv:2207.05221*, 2022.
14. S. Min et al., "FActScore: Fine-grained atomic evaluation of factual precision in long form text generation," in *Proc. EMNLP*, pp. 12076–12100, 2023.
15. A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in *International Conference on Machine Learning (ICML)*, pp. 625–632, 2005.
16. G. Izacard and E. Grave, "Leveraging passage retrieval with generative models for open domain question answering," in *Proc. EACL*, pp. 874–880, 2021.
17. Q. Jin et al., "MedCPT: Contrastive pre-trained transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval," *Bioinformatics*, vol. 39, no. 11, p. btad651, 2023.
18. J. Lee et al., "BioBERT: A pre-trained biomedical language representation model for biomedical text mining," *Bioinformatics*, vol. 36, no. 4, pp. 1234–1240, 2020.
19. S. E. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.
20. G. V. Cormack, C. L. Clarke, and S. Buettcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," in *Proc. ACM SIGIR*, pp. 758–759, 2009.
21. L. Gao et al., "Precise zero-shot dense retrieval without relevance labels," in *Proc. ACL*, pp. 1762–1777, 2023.
22. J. Thorne et al., "FEVER: A large-scale dataset for fact extraction and verification," in *Proc. NAACL-HLT*, pp. 809–819, 2018.
23. Y. Wadden et al., "Fact or fiction: Verifying scientific claims," in *Proc. EMNLP*, pp. 7534–7550, 2020.
24. S. Es et al., "RAGAS: Automated evaluation of retrieval augmented generation," *arXiv preprint arXiv:2309.15217*, 2023.
25. P. He, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa using electra-style pre-training with gradient-disentangled embedding sharing," *arXiv preprint arXiv:2111.09543*, 2021.
26. M. P. Naeini, G. Cooper, and M. Hauskrecht, "Obtaining well calibrated probabilities using Bayesian binning into quantiles," in *Proc. AAAI*, vol. 29, no. 1, 2015.
27. J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," *Advances in Large Margin Classifiers*, vol. 10, no. 3, pp. 61–74, 1999.
28. B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in *Proc. ACM SIGKDD*, pp. 694–699, 2002.
29. Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
30. C. Leibig et al., "Leveraging uncertainty information from deep neural networks for disease detection," *Scientific Reports*, vol. 7, no. 1, p. 17816, 2017.
31. P. L. Bartlett and M. H. Wegkamp, "Classification with a reject option using a hinge loss," *Journal of Machine Learning Research*, vol. 9, pp. 1823–1840, 2008.
32. H. Mozannar and D. Sontag, "Consistent estimators for learning to defer to an expert," in *International Conference on Machine Learning (ICML)*, pp. 7076–7087, 2020.
33. Q. Jin, B. Dhingra, Z. Liu, W. W. Cohen, and X. Lu, "PubMedQA: A dataset for biomedical research question answering," in *Proc. EMNLP-IJCNLP*, pp. 2567–2577, 2019.
