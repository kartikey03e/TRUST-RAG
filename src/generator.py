"""
TRUST-RAG V2 — LLM Generator
==============================
Unified generation interface for Qwen 2.5 7B and MedGemma 4B IT.
Handles model loading (with quantization), prompt formatting, and
batch inference with checkpoint/resume support.

Supports three backends:
  - "hf"     : HuggingFace transformers (original, for CUDA / Colab)
  - "ollama" : Ollama HTTP API (for local Qwen via Ollama)
  - "mlx"    : Apple MLX (for local MedGemma via mlx-vlm)

Backend is auto-detected based on platform and config.

Usage
-----
    from src.generator import BiomedicalGenerator
    gen = BiomedicalGenerator.from_config(config, model_key="medgemma")
    output = gen.generate(question="...", evidence_blocks=[...])
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import get_default_config, resolve_path
from src.retriever import EvidenceBlock
from src.utils import (
    append_jsonl,
    clear_gpu_memory,
    get_logger,
    load_jsonl,
    print_gpu_memory,
    timer,
)

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Result from a single generation call."""

    question: str
    question_pmid: str
    evidence_pmids: list[str]
    evidence_texts: list[str]
    retrieval_scores: list[float]
    raw_output: str
    model_name: str
    generation_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "question_pmid": self.question_pmid,
            "evidence_pmids": self.evidence_pmids,
            "evidence_texts": self.evidence_texts,
            "retrieval_scores": [round(s, 6) for s in self.retrieval_scores],
            "raw_output": self.raw_output,
            "model_name": self.model_name,
            "generation_time": round(self.generation_time, 3),
        }


class BiomedicalGenerator:
    """Unified LLM generator for biomedical question answering."""

    def __init__(
        self,
        model_name: str,
        prompt_template: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        quantization: str | None = None,
        dtype: str = "float16",
        device: str = "cuda",
        backend: str = "hf",
        local_name: str | None = None,
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.quantization = quantization
        self.dtype = dtype
        self.device = device
        self.backend = backend
        self.local_name = local_name

        self._model = None
        self._tokenizer = None
        # MLX-specific
        self._mlx_model = None
        self._mlx_processor = None

    @classmethod
    def from_config(
        cls,
        config: dict | None = None,
        model_key: str = "medgemma",
    ) -> "BiomedicalGenerator":
        """Construct a generator from config for a specific model."""
        if config is None:
            config = get_default_config()

        gen_cfg = config["generation"]
        model_cfg = gen_cfg["models"][model_key]

        from src.config import get_device

        device = get_device()

        # Determine backend: on CUDA use HuggingFace, otherwise use local backend
        if device == "cuda":
            backend = "hf"
        else:
            backend = model_cfg.get("local_backend", "hf")

        local_name = model_cfg.get("local_name")

        return cls(
            model_name=model_cfg["name"],
            prompt_template=gen_cfg["prompt_template"],
            max_new_tokens=model_cfg["max_new_tokens"],
            temperature=model_cfg["temperature"],
            top_p=model_cfg["top_p"],
            repetition_penalty=model_cfg["repetition_penalty"],
            do_sample=model_cfg["do_sample"],
            quantization=model_cfg.get("quantization"),
            dtype=model_cfg.get("dtype", "float16"),
            device=device,
            backend=backend,
            local_name=local_name,
        )

    # ── Model loading ────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the model using the appropriate backend."""
        if self.backend == "ollama":
            self._load_model_ollama()
        elif self.backend == "mlx":
            self._load_model_mlx()
        else:
            self._load_model_hf()

    def _load_model_hf(self) -> None:
        """Load the model and tokenizer via HuggingFace transformers (original)."""
        if self._model is not None:
            logger.info("Model already loaded")
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {self.model_name}")
        print_gpu_memory("before model load")

        import os
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        # Tokenizer kwargs
        tok_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if token:
            tok_kwargs["token"] = token

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **tok_kwargs,
            )
        except Exception as e:
            if "gated" in str(e).lower() or "401" in str(e) or "restricted" in str(e).lower():
                logger.error(
                    f"Model '{self.model_name}' is a gated Hugging Face repository. "
                    f"Please set your HF_TOKEN (e.g., `os.environ['HF_TOKEN'] = 'your_token'`) "
                    f"or run `huggingface_hub.login()` in Colab."
                )
            raise e

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Model kwargs
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if token:
            model_kwargs["token"] = token

        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        else:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            model_kwargs["torch_dtype"] = dtype_map.get(self.dtype, torch.float16)
            model_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        print_gpu_memory("after model load")
        logger.info(f"Model loaded: {self.model_name}")

    def _load_model_ollama(self) -> None:
        """Verify Ollama is running and the model is available."""
        import json
        import urllib.request

        ollama_model = self.local_name or self.model_name
        logger.info(f"Connecting to Ollama for model: {ollama_model}")

        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            available = [m["name"] for m in data.get("models", [])]
            if ollama_model not in available:
                logger.warning(
                    f"Model '{ollama_model}' not found in Ollama. "
                    f"Available: {available}. Attempting to pull..."
                )
                # Try to use it anyway — Ollama may pull on first request
            else:
                logger.info(f"Ollama model verified: {ollama_model}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at localhost:11434. "
                f"Please start Ollama first: https://ollama.ai. Error: {e}"
            ) from e

        # Mark as loaded (Ollama manages its own model memory)
        self._model = "ollama"

    def _load_model_mlx(self) -> None:
        """Load the model via Apple MLX (mlx-vlm)."""
        if self._mlx_model is not None:
            logger.info("MLX model already loaded")
            return

        import mlx_vlm

        mlx_model_name = self.local_name or self.model_name
        logger.info(f"Loading MLX model: {mlx_model_name}")

        self._mlx_model, self._mlx_processor = mlx_vlm.load(mlx_model_name)

        logger.info(f"MLX model loaded: {mlx_model_name}")

    def unload_model(self) -> None:
        """Free model from memory."""
        if self._model is not None and self._model != "ollama":
            del self._model
            self._model = None
        elif self._model == "ollama":
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._mlx_model is not None:
            del self._mlx_model
            self._mlx_model = None
        if self._mlx_processor is not None:
            del self._mlx_processor
            self._mlx_processor = None
        clear_gpu_memory()
        logger.info(f"Model unloaded: {self.model_name}")

    # ── Prompt formatting ────────────────────────────────────────────────

    def format_prompt(
        self,
        question: str,
        evidence_blocks: list[EvidenceBlock],
    ) -> str:
        """Format the prompt with question and evidence."""
        evidence_parts = []
        for i, block in enumerate(evidence_blocks, 1):
            evidence_parts.append(f"[Source {i} — PMID {block.pmid}]\n{block.text}")

        evidence_str = "\n\n".join(evidence_parts)

        return self.prompt_template.format(
            question=question,
            evidence=evidence_str,
        )

    # ── Generation ───────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        evidence_blocks: list[EvidenceBlock],
        question_pmid: str = "",
    ) -> GenerationResult:
        """Generate an answer for a single question.

        Parameters
        ----------
        question : str
            The biomedical question.
        evidence_blocks : list[EvidenceBlock]
            Retrieved evidence blocks.
        question_pmid : str
            Gold PMID for this question (for record-keeping only).

        Returns
        -------
        GenerationResult
        """
        if self.backend == "ollama":
            return self._generate_ollama(question, evidence_blocks, question_pmid)
        elif self.backend == "mlx":
            return self._generate_mlx(question, evidence_blocks, question_pmid)
        else:
            return self._generate_hf(question, evidence_blocks, question_pmid)

    def _generate_hf(
        self,
        question: str,
        evidence_blocks: list[EvidenceBlock],
        question_pmid: str = "",
    ) -> GenerationResult:
        """Generate using HuggingFace transformers (original CUDA path)."""
        import time

        import torch

        if self._model is None:
            self.load_model()

        prompt = self.format_prompt(question, evidence_blocks)

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._model.device)

        # Generate
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=self.do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen_time = time.perf_counter() - start_time

        # Decode (only the generated part, not the prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        raw_output = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return GenerationResult(
            question=question,
            question_pmid=question_pmid,
            evidence_pmids=[b.pmid for b in evidence_blocks],
            evidence_texts=[b.text for b in evidence_blocks],
            retrieval_scores=[b.score for b in evidence_blocks],
            raw_output=raw_output.strip(),
            model_name=self.model_name,
            generation_time=gen_time,
        )

    def _generate_ollama(
        self,
        question: str,
        evidence_blocks: list[EvidenceBlock],
        question_pmid: str = "",
    ) -> GenerationResult:
        """Generate using Ollama HTTP API (local Qwen)."""
        import json
        import time
        import urllib.request

        if self._model is None:
            self.load_model()

        prompt = self.format_prompt(question, evidence_blocks)
        ollama_model = self.local_name or self.model_name

        payload = json.dumps({
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repeat_penalty": self.repetition_penalty,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        start_time = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode())
        gen_time = time.perf_counter() - start_time

        raw_output = result.get("response", "").strip()

        return GenerationResult(
            question=question,
            question_pmid=question_pmid,
            evidence_pmids=[b.pmid for b in evidence_blocks],
            evidence_texts=[b.text for b in evidence_blocks],
            retrieval_scores=[b.score for b in evidence_blocks],
            raw_output=raw_output,
            model_name=ollama_model,
            generation_time=gen_time,
        )

    def _generate_mlx(
        self,
        question: str,
        evidence_blocks: list[EvidenceBlock],
        question_pmid: str = "",
    ) -> GenerationResult:
        """Generate using Apple MLX (local MedGemma)."""
        import time

        import mlx_vlm

        if self._mlx_model is None:
            self.load_model()

        prompt = self.format_prompt(question, evidence_blocks)
        mlx_model_name = self.local_name or self.model_name

        start_time = time.perf_counter()
        result = mlx_vlm.generate(
            self._mlx_model,
            self._mlx_processor,
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0 else 0.01,
            verbose=False,
        )
        gen_time = time.perf_counter() - start_time

        # mlx_vlm.generate returns a GenerationResult object with a .text attribute
        if hasattr(result, "text"):
            raw_output = result.text.strip()
        else:
            raw_output = str(result).strip()

        return GenerationResult(
            question=question,
            question_pmid=question_pmid,
            evidence_pmids=[b.pmid for b in evidence_blocks],
            evidence_texts=[b.text for b in evidence_blocks],
            retrieval_scores=[b.score for b in evidence_blocks],
            raw_output=raw_output,
            model_name=mlx_model_name,
            generation_time=gen_time,
        )

    def generate_batch(
        self,
        records: list[dict],
        evidence_map: dict[str, list[EvidenceBlock]],
        output_path: str | Path | None = None,
        resume: bool = True,
    ) -> list[GenerationResult]:
        """Generate answers for a batch of records with checkpoint/resume.

        Parameters
        ----------
        records : list[dict]
            Dataset records with 'question' and 'pmid'.
        evidence_map : dict
            Mapping from PMID → list of EvidenceBlock.
        output_path : str | Path | None
            If provided, each result is appended to this JSONL file.
        resume : bool
            If True and output_path exists, skip already-processed questions.

        Returns
        -------
        list[GenerationResult]
        """
        # Load existing results for resume
        completed_pmids = set()
        if resume and output_path and Path(output_path).exists():
            existing = load_jsonl(output_path)
            completed_pmids = {r["question_pmid"] for r in existing}
            logger.info(f"Resuming: {len(completed_pmids)} questions already completed")

        results = []
        for i, rec in enumerate(records):
            pmid = str(rec["pmid"])

            if pmid in completed_pmids:
                continue

            question = rec["question"]
            evidence = evidence_map.get(pmid, [])

            if not evidence:
                logger.warning(f"No evidence for PMID {pmid}, skipping")
                continue

            result = self.generate(
                question=question,
                evidence_blocks=evidence,
                question_pmid=pmid,
            )
            results.append(result)

            # Save incrementally
            if output_path:
                append_jsonl(result.to_dict(), output_path)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"Generated {i + 1}/{len(records)} | "
                    f"Last: {result.generation_time:.1f}s"
                )

        logger.info(f"Generation complete: {len(results)} new results")
        return results
