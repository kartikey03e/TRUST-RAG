"""
TRUST-RAG: Generate Trust Features for Training Split
======================================================
Runs Stages 3 through 7 on the 600-sample training split:
  1. Evidence Retrieval (Hybrid RRF BM25 + MedCPT) -> results/retrieval/train_retrieval.jsonl
  2. Generation (Qwen 2.5 7B via Ollama) -> results/generation/qwen_train_outputs.jsonl
  3. Decision Extraction (Regex + BART-MNLI)
  4. Claim-level NLI Verification (SciSpaCy + DeBERTa-v3 on MPS) -> results/verification/qwen_train_nli.jsonl
  5. 9-D Trust Feature Construction (MiniLM) -> results/trust/qwen_train_trust.jsonl

Includes resume capability at every stage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.claim_extractor import ClaimExtractor
from src.config import get_default_config, resolve_path, set_seed
from src.decision_extractor import DecisionExtractor
from src.generator import BiomedicalGenerator
from src.nli_verifier import NLIVerifier
from src.retriever import EvidenceBlock, HybridRetriever
from src.trust_features import TrustFeatureBuilder
from src.utils import append_jsonl, get_logger, load_jsonl, save_jsonl, timer

logger = get_logger("train_pipeline")


def run_train_feature_pipeline(
    max_samples: int | None = None,
    batch_size_nli: int = 32,
) -> Path:
    config = get_default_config()
    set_seed(config.get("seed", 42))

    results_dir = resolve_path(config["results"]["base_dir"])
    data_dir = resolve_path(config["data"]["processed_dir"])

    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found at {train_path}")

    records = load_jsonl(train_path)
    if max_samples:
        records = records[:max_samples]
    total_records = len(records)
    logger.info(f"Loaded {total_records} training records")

    # ─────────────────────────────────────────────────────────────
    # STAGE 1: Retrieval
    # ─────────────────────────────────────────────────────────────
    retrieval_dir = results_dir / "retrieval"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = retrieval_dir / "train_retrieval.jsonl"

    cached_retrievals = {}
    if retrieval_path.exists():
        for r in load_jsonl(retrieval_path):
            cached_retrievals[r["question_pmid"]] = r
        logger.info(f"Loaded {len(cached_retrievals)} cached retrievals")

    uncached_ret = [r for r in records if str(r["pmid"]) not in cached_retrievals]
    if uncached_ret:
        logger.info(f"Running retrieval for {len(uncached_ret)} queries...")
        retriever = HybridRetriever.from_config(config)
        with timer("Training Retrieval", logger):
            for i, rec in enumerate(uncached_ret, 1):
                pmid = str(rec["pmid"])
                blocks = retriever.retrieve(
                    rec["question"],
                    top_k=config["retrieval"]["evidence_top_k"],
                )
                entry = {
                    "question_pmid": pmid,
                    "evidence_pmids": [b.pmid for b in blocks],
                    "evidence_scores": [round(b.score, 6) for b in blocks],
                    "evidence_texts": [b.text for b in blocks],
                    "n_chunks": [b.n_chunks for b in blocks],
                }
                append_jsonl(entry, retrieval_path)
                cached_retrievals[pmid] = entry
                if i % 50 == 0 or i == len(uncached_ret):
                    logger.info(f"  Retrieval progress: {i}/{len(uncached_ret)}")
        retriever.unload_query_encoder()
        del retriever

    # Reconstruct evidence blocks map
    evidence_cache = {}
    for pmid, item in cached_retrievals.items():
        blocks = []
        for j in range(len(item["evidence_pmids"])):
            blocks.append(EvidenceBlock(
                pmid=item["evidence_pmids"][j],
                text=item["evidence_texts"][j],
                score=item["evidence_scores"][j],
                n_chunks=item["n_chunks"][j],
            ))
        evidence_cache[pmid] = blocks

    # ─────────────────────────────────────────────────────────────
    # STAGE 2: Generation (Qwen 2.5 7B via Ollama)
    # ─────────────────────────────────────────────────────────────
    gen_dir = results_dir / "generation"
    gen_dir.mkdir(parents=True, exist_ok=True)
    gen_output_path = gen_dir / "qwen_train_outputs.jsonl"

    cached_gens = {}
    if gen_output_path.exists():
        for g in load_jsonl(gen_output_path):
            cached_gens[g["question_pmid"]] = g
        logger.info(f"Loaded {len(cached_gens)} cached generation outputs")

    uncached_gen = [r for r in records if str(r["pmid"]) not in cached_gens]
    if uncached_gen:
        logger.info(f"Running Qwen generation for {len(uncached_gen)} queries...")
        generator = BiomedicalGenerator.from_config(config, model_key="qwen")
        generator.load_model()
        with timer("Training Generation", logger):
            for i, rec in enumerate(uncached_gen, 1):
                pmid = str(rec["pmid"])
                blocks = evidence_cache.get(pmid, [])
                res = generator.generate(rec["question"], blocks, question_pmid=pmid)
                append_jsonl(res.to_dict(), gen_output_path)
                cached_gens[pmid] = res.to_dict()
                if i % 10 == 0 or i == len(uncached_gen):
                    logger.info(f"  Generation progress: {i}/{len(uncached_gen)} (last time: {res.generation_time:.2f}s)")
        generator.unload_model()
        del generator

    # ─────────────────────────────────────────────────────────────
    # STAGE 3: Decision Extraction
    # ─────────────────────────────────────────────────────────────
    logger.info("Extracting categorical decisions...")
    decision_extractor = DecisionExtractor.from_config(config)
    pmid_to_pred_decision = {}
    for pmid, g in cached_gens.items():
        d = decision_extractor.extract(g["raw_output"])
        pmid_to_pred_decision[pmid] = d["decision"]
    decision_extractor.unload_classifier()
    del decision_extractor

    # ─────────────────────────────────────────────────────────────
    # STAGE 4: Claim Extraction & NLI Verification
    # ─────────────────────────────────────────────────────────────
    nli_dir = results_dir / "verification"
    nli_dir.mkdir(parents=True, exist_ok=True)
    nli_output_path = nli_dir / "qwen_train_nli.jsonl"

    cached_nli = {}
    if nli_output_path.exists():
        for n in load_jsonl(nli_output_path):
            cached_nli[n["question_pmid"]] = n
        logger.info(f"Loaded {len(cached_nli)} cached NLI verification outputs")

    uncached_nli = [r for r in records if str(r["pmid"]) not in cached_nli]
    if uncached_nli:
        logger.info(f"Running Claim Extraction & NLI for {len(uncached_nli)} queries...")
        claim_extractor = ClaimExtractor.from_config(config)
        nli_verifier = NLIVerifier.from_config(config)
        nli_verifier.load_model()

        with timer("Training NLI Verification", logger):
            for i, rec in enumerate(uncached_nli, 1):
                pmid = str(rec["pmid"])
                gen_data = cached_gens[pmid]
                raw_answer = gen_data["raw_output"]
                evidence_texts = gen_data.get("evidence_texts", [])

                claims = claim_extractor.extract(raw_answer)
                nli_res = nli_verifier.verify_answer(
                    claims=claims,
                    evidence_passages=evidence_texts,
                    question_pmid=pmid,
                )
                entry = {
                    "question_pmid": pmid,
                    "mean_max_entailment": nli_res.mean_max_entailment,
                    "frac_entailed_claims": nli_res.frac_entailed,
                    "max_contradiction": nli_res.max_contradiction,
                    "frac_contradicted_claims": nli_res.frac_contradicted,
                    "n_claims": nli_res.n_claims,
                }
                append_jsonl(entry, nli_output_path)
                cached_nli[pmid] = entry
                if i % 25 == 0 or i == len(uncached_nli):
                    logger.info(f"  NLI verification progress: {i}/{len(uncached_nli)}")

        nli_verifier.unload_model()
        del nli_verifier

    # ─────────────────────────────────────────────────────────────
    # STAGE 5: 9-D Trust Feature Construction
    # ─────────────────────────────────────────────────────────────
    trust_dir = results_dir / "trust"
    trust_dir.mkdir(parents=True, exist_ok=True)
    trust_output_path = trust_dir / "qwen_train_trust.jsonl"

    logger.info("Building 9-D trust feature representations...")
    feature_builder = TrustFeatureBuilder()
    pmid_to_gold = {str(r["pmid"]): r["final_decision"] for r in records}

    trust_records = []
    for rec in records:
        pmid = str(rec["pmid"])
        ret_data = cached_retrievals[pmid]
        nli_data = cached_nli[pmid]
        gen_data = cached_gens[pmid]

        retrieval_dict = {
            "top1_score": ret_data["evidence_scores"][0] if ret_data["evidence_scores"] else 0.0,
            "score_margin": (ret_data["evidence_scores"][0] - ret_data["evidence_scores"][1]) if len(ret_data["evidence_scores"]) > 1 else 0.0,
            "candidate_pmids": ret_data.get("candidate_pmids", ret_data.get("evidence_pmids", [])),
            "evidence_pmids": ret_data.get("evidence_pmids", []),
            "retrieved_pmids": ret_data.get("evidence_pmids", []),
        }
        nli_dict = {
            "mean_max_entailment": nli_data["mean_max_entailment"],
            "frac_entailed": nli_data["frac_entailed_claims"],
            "max_contradiction": nli_data["max_contradiction"],
            "frac_contradicted": nli_data["frac_contradicted_claims"],
        }
        evidence_text = " ".join(gen_data.get("evidence_texts", []))

        tf = feature_builder.build(
            retrieval_result=retrieval_dict,
            nli_result=nli_dict,
            generated_answer=gen_data["raw_output"],
            evidence_text=evidence_text,
            question=gen_data["question"],
            question_pmid=pmid,
        )

        pred_dec = pmid_to_pred_decision.get(pmid, "")
        gold_dec = pmid_to_gold.get(pmid, "")
        is_correct = (pred_dec.lower() == gold_dec.lower()) if (pred_dec and gold_dec) else False

        trust_records.append({
            **tf.to_dict(),
            "predicted_decision": pred_dec,
            "gold_decision": gold_dec,
            "correct": is_correct,
        })

    save_jsonl(trust_records, trust_output_path)
    logger.info(f"Saved {len(trust_records)} training trust records to {trust_output_path}")

    return trust_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Training Trust Features")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    run_train_feature_pipeline(max_samples=args.max_samples)
