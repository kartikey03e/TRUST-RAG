"""
TRUST-RAG V2 — Full Pipeline Runner
=====================================
Orchestrates the complete TRUST-RAG pipeline end-to-end:
  Data → Index → Retrieve → Generate → Extract → Verify → Trust → Evaluate

Designed for Google Colab with sequential model loading on T4.

Usage
-----
    python experiments/run_full_pipeline.py --model medgemma
    python experiments/run_full_pipeline.py --model qwen --split val
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_default_config, load_config, resolve_path, set_seed
from src.utils import get_logger, load_jsonl, save_json, save_jsonl, timer

logger = get_logger("pipeline")


def run_pipeline(
    config: dict | None = None,
    model_key: str = "medgemma",
    split: str = "val",
    skip_indexing: bool = False,
    skip_retrieval: bool = False,
    skip_generation: bool = False,
) -> dict:
    """Run the full TRUST-RAG pipeline.

    Parameters
    ----------
    config : dict | None
        Configuration. Uses default if None.
    model_key : str
        "medgemma" or "qwen".
    split : str
        Dataset split to evaluate on: "val" or "test".
    skip_indexing : bool
        Skip index building (use existing indices).
    skip_retrieval : bool
        Skip retrieval (use cached retrieval results).
    skip_generation : bool
        Skip generation (use cached generation results).

    Returns
    -------
    dict
        Complete pipeline results.
    """
    if config is None:
        config = get_default_config()

    seed = config.get("seed", 42)
    set_seed(seed)

    results_dir = resolve_path(config["results"]["base_dir"])
    data_dir = resolve_path(config["data"]["processed_dir"])

    pipeline_start = time.time()
    all_results = {"model": model_key, "split": split, "seed": seed}

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 1: Load data
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE 1: Loading data")
    logger.info("=" * 60)

    split_path = data_dir / f"{split}.jsonl"
    if not split_path.exists():
        logger.info("Splits not found — running data loader first")
        from src.data_loader import run as run_data_loader
        run_data_loader(config)

    records = load_jsonl(split_path)
    logger.info(f"Loaded {len(records)} records from {split} split")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 2: Indexing
    # ═══════════════════════════════════════════════════════════════════
    if not skip_indexing:
        faiss_path = resolve_path(config["retrieval"]["faiss_index_path"])
        if not faiss_path.exists():
            logger.info("=" * 60)
            logger.info("STAGE 2: Building indices")
            logger.info("=" * 60)
            from src.indexer import run as run_indexer
            index_result = run_indexer(config)
            all_results["indexing"] = index_result
        else:
            logger.info("STAGE 2: Indices already exist — skipping")
    else:
        logger.info("STAGE 2: Indexing skipped (--skip-indexing)")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 3: Retrieval
    # ═══════════════════════════════════════════════════════════════════
    retrieval_path = results_dir / "retrieval" / f"{split}_retrieval.jsonl"

    if not skip_retrieval:
        logger.info("=" * 60)
        logger.info("STAGE 3: Retrieval")
        logger.info("=" * 60)

        from src.retrieval_eval import evaluate_retrieval
        from src.retriever import HybridRetriever

        retriever = HybridRetriever.from_config(config)

        with timer("Retrieval + evaluation", logger):
            retrieval_results = evaluate_retrieval(
                retriever, records,
                ks=[1, 3, 5],
                top_k=config["retrieval"]["final_top_k"],
            )

        all_results["retrieval"] = retrieval_results["aggregate"]
        save_json(
            retrieval_results["aggregate"],
            results_dir / "retrieval" / f"{split}_retrieval_metrics.json",
        )

        # Cache retrieval results + evidence for generation
        evidence_cache = {}
        for rec in records:
            pmid = str(rec["pmid"])
            question = rec["question"]
            blocks = retriever.retrieve(
                question,
                top_k=config["retrieval"]["evidence_top_k"],
            )
            evidence_cache[pmid] = blocks

        # Save retrieval details
        retrieval_records = []
        for pmid, blocks in evidence_cache.items():
            retrieval_records.append({
                "question_pmid": pmid,
                "evidence_pmids": [b.pmid for b in blocks],
                "evidence_scores": [round(b.score, 6) for b in blocks],
                "evidence_texts": [b.text for b in blocks],
                "n_chunks": [b.n_chunks for b in blocks],
            })
        save_jsonl(retrieval_records, retrieval_path)

        retriever.unload_query_encoder()
        del retriever
        from src.utils import clear_gpu_memory
        clear_gpu_memory()

    else:
        logger.info("STAGE 3: Loading cached retrieval results")
        cached = load_jsonl(retrieval_path)
        # Rebuild evidence_cache from cached results
        from src.retriever import EvidenceBlock
        evidence_cache = {}
        for item in cached:
            pmid = item["question_pmid"]
            blocks = []
            for i in range(len(item["evidence_pmids"])):
                blocks.append(EvidenceBlock(
                    pmid=item["evidence_pmids"][i],
                    text=item["evidence_texts"][i],
                    score=item["evidence_scores"][i],
                    n_chunks=item["n_chunks"][i],
                ))
            evidence_cache[pmid] = blocks

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 4: Generation
    # ═══════════════════════════════════════════════════════════════════
    gen_output_path = results_dir / "generation" / f"{model_key}_{split}_outputs.jsonl"

    if not skip_generation:
        logger.info("=" * 60)
        logger.info(f"STAGE 4: Generation ({model_key})")
        logger.info("=" * 60)

        from src.generator import BiomedicalGenerator

        generator = BiomedicalGenerator.from_config(config, model_key=model_key)
        generator.load_model()

        with timer("Generation", logger):
            gen_results = generator.generate_batch(
                records=records,
                evidence_map=evidence_cache,
                output_path=gen_output_path,
                resume=True,
            )

        generator.unload_model()
        del generator
        from src.utils import clear_gpu_memory
        clear_gpu_memory()

    gen_outputs = load_jsonl(gen_output_path)
    logger.info(f"Loaded {len(gen_outputs)} generation outputs")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 5: Decision Extraction
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE 5: Decision Extraction")
    logger.info("=" * 60)

    from src.decision_extractor import DecisionExtractor

    extractor = DecisionExtractor.from_config(config)
    raw_outputs = [g["raw_output"] for g in gen_outputs]

    with timer("Decision extraction", logger):
        decisions = extractor.extract_batch(raw_outputs)

    extractor.unload_classifier()

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 6: Claim Extraction + NLI Verification
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE 6: Claim Extraction + NLI Verification")
    logger.info("=" * 60)

    from src.claim_extractor import ClaimExtractor
    from src.nli_verifier import NLIVerifier

    claim_extractor = ClaimExtractor.from_config(config)
    nli_verifier = NLIVerifier.from_config(config)

    verification_results = []
    with timer("Claim extraction + NLI", logger):
        for i, gen_out in enumerate(gen_outputs):
            # Extract claims
            claims = claim_extractor.extract(gen_out["raw_output"])

            # Get evidence texts
            evidence_texts = gen_out.get("evidence_texts", [])

            # Verify claims against evidence
            answer_verif = nli_verifier.verify_answer(
                claims=claims,
                evidence_passages=evidence_texts,
                question_pmid=gen_out.get("question_pmid", ""),
            )
            verification_results.append(answer_verif.to_dict())

            if (i + 1) % 50 == 0:
                logger.info(f"  Verified {i + 1}/{len(gen_outputs)} answers")

    nli_verifier.unload_model()

    save_jsonl(
        verification_results,
        results_dir / "verification" / f"{model_key}_{split}_nli.jsonl",
    )

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 7: Trust Features + Trust Model
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE 7: Trust Features + Model")
    logger.info("=" * 60)

    import numpy as np

    from src.trust_features import TrustFeatureBuilder

    feature_builder = TrustFeatureBuilder()

    # Build retrieval result dicts for feature builder
    trust_feature_list = []
    for gen_out, verif, dec in zip(gen_outputs, verification_results, decisions):
        pmid = gen_out.get("question_pmid", "")
        ret_scores = gen_out.get("retrieval_scores", [0.0])

        retrieval_dict = {
            "top1_score": ret_scores[0] if ret_scores else 0.0,
            "score_margin": (ret_scores[0] - ret_scores[1]) if len(ret_scores) > 1 else 0.0,
            "retrieved_pmids": gen_out.get("evidence_pmids", []),
        }

        nli_dict = {
            "mean_max_entailment": verif.get("mean_max_entailment", 0.0),
            "frac_entailed": verif.get("frac_entailed", 0.0),
            "max_contradiction": verif.get("max_contradiction", 0.0),
            "frac_contradicted": verif.get("frac_contradicted", 0.0),
        }

        evidence_text = " ".join(gen_out.get("evidence_texts", []))

        tf = feature_builder.build(
            retrieval_result=retrieval_dict,
            nli_result=nli_dict,
            generated_answer=gen_out["raw_output"],
            evidence_text=evidence_text,
            question=gen_out["question"],
            question_pmid=pmid,
        )
        trust_feature_list.append(tf)

    # Feature matrix
    X = feature_builder.get_feature_matrix(trust_feature_list)

    # Get gold labels and correctness
    pmid_to_gold = {str(r["pmid"]): r["final_decision"] for r in records}
    gold_labels = [pmid_to_gold.get(gen_out["question_pmid"], "") for gen_out in gen_outputs]
    predicted_decisions = [d["decision"] for d in decisions]
    correct = np.array([1 if p == g else 0 for p, g in zip(predicted_decisions, gold_labels)])

    # Trust model: use heuristic for single-split evaluation
    # (Learned model requires separate train split)
    from src.trust_model import TrustModel

    trust_model = TrustModel(model_type="heuristic")
    trust_model._is_fitted = True
    trust_scores = trust_model.predict_trust(X)
    trust_predictions = trust_model.predict_batch(X)

    # Save trust results
    trust_records = []
    for i, (tf, tp) in enumerate(zip(trust_feature_list, trust_predictions)):
        trust_records.append({
            **tf.to_dict(),
            **tp,
            "predicted_decision": predicted_decisions[i],
            "gold_decision": gold_labels[i],
            "correct": bool(correct[i]),
        })
    save_jsonl(trust_records, results_dir / "trust" / f"{model_key}_{split}_trust.jsonl")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 8: Evaluation
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE 8: Evaluation")
    logger.info("=" * 60)

    from src.evaluation import Evaluator

    evaluator = Evaluator.from_config(config)
    eval_results = evaluator.evaluate_all(predicted_decisions, gold_labels, trust_scores)
    all_results["evaluation"] = eval_results

    save_json(
        eval_results,
        results_dir / "final" / f"{model_key}_{split}_evaluation.json",
    )

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 9: Error Analysis
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STAGE 9: Error Analysis")
    logger.info("=" * 60)

    from src.error_analysis import ErrorAnalyzer

    # Load retrieval details for error classification
    retrieval_cache = load_jsonl(retrieval_path)
    pmid_to_retrieval = {r["question_pmid"]: r for r in retrieval_cache}

    error_inputs = []
    for i, gen_out in enumerate(gen_outputs):
        pmid = gen_out.get("question_pmid", "")
        ret_detail = pmid_to_retrieval.get(pmid, {})
        retrieved_pmids = ret_detail.get("evidence_pmids", [])

        error_inputs.append({
            "question_pmid": pmid,
            "question": gen_out["question"],
            "correct": bool(correct[i]),
            "predicted_decision": predicted_decisions[i],
            "gold_decision": gold_labels[i],
            "gold_pmid_in_top5": pmid in retrieved_pmids,
            "gold_pmid_rank": (retrieved_pmids.index(pmid) + 1) if pmid in retrieved_pmids else 0,
            "mean_max_entailment": verification_results[i].get("mean_max_entailment", 0.0),
            "max_contradiction": verification_results[i].get("max_contradiction", 0.0),
        })

    analyzer = ErrorAnalyzer()
    error_results = analyzer.classify_errors(error_inputs)
    all_results["error_analysis"] = {
        k: v for k, v in error_results.items() if k != "classifications"
    }

    save_json(
        error_results,
        results_dir / "final" / f"{model_key}_{split}_errors.json",
    )

    # ═══════════════════════════════════════════════════════════════════
    # DONE
    # ═══════════════════════════════════════════════════════════════════
    total_time = time.time() - pipeline_start
    all_results["total_time_seconds"] = round(total_time, 1)
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE — {total_time:.0f}s total")
    logger.info("=" * 60)

    save_json(all_results, results_dir / "final" / f"{model_key}_{split}_full_results.json")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRUST-RAG Full Pipeline")
    parser.add_argument("--model", type=str, default="medgemma", choices=["medgemma", "qwen"])
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-indexing", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    run_pipeline(
        config=cfg,
        model_key=args.model,
        split=args.split,
        skip_indexing=args.skip_indexing,
        skip_retrieval=args.skip_retrieval,
        skip_generation=args.skip_generation,
    )
