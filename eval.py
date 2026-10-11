"""
Small retrieval-evaluation harness for finRAG.py.

Run from the project directory, where finRAG.py and Data/ are available:
		python eval.py --embedding_model "Octen/Octen-Embedding-0.6B" --top_k 3 5 10 --show_chunks
		python eval_set.py --embedding_model "Octen/Octen-Embedding-8B" --top_k 5 10 25
		python eval_set.py --embedding_model "Octen/Octen-Embedding-8B" --top_k 5 10 25 --show_chunks

This measures retrieval, not generated-answer correctness. It reuses finRAG's
document loading, chunking, embedding and retrieval functions, plus its on-disk
index-cache convention. No LLM is loaded, which makes retrieval experiments
cheaper and helps isolate retrieval problems from generation problems.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

import finRAG as rag


# Tune these expectations after manually verifying the supplied PDFs.
# source_any: at least one term must occur in the source filename/path.
# evidence_any: at least one term must occur in the retrieved chunk text.
# A case with source_any=[] is useful for general evidence retrieval without
# asserting a particular report identity.
EVAL_SET: list[dict[str, Any]] = [
		{
				"id": "tesla_energy_revenue_2022",
				"question": "What was Tesla's Energy generation and storage revenue in 2022?",
				"source_any": ["tesla"],
				"evidence_any": ["3,909", "3909"],
				"expected_note": "Tesla 2023 annual report gives the three-year segment revenue series.",
		},
		{
				"id": "tesla_energy_revenue_2021",
				"question": "What was Tesla's Energy generation and storage revenue in 2021?",
				"source_any": ["tesla"],
				"evidence_any": ["2,789", "2789"],
				"expected_note": "Tesla 2023 annual report gives the three-year segment revenue series.",
		},
		{
				"id": "tesla_energy_revenue_2023",
				"question": "What was Tesla's Energy generation and storage revenue in 2023?",
				"source_any": ["tesla"],
				"evidence_any": ["6,035", "6035"],
				"expected_note": "Tesla 2023 annual report gives the three-year segment revenue series.",
		},
		{
				"id": "ford_2022_investment_losses",
				"question": "Why did Ford report a net loss in 2022 despite strong sales?",
				"source_any": ["ford"],
				"evidence_any": ["Rivian", "Argo", "impairment", "mark-to-market"],
				"expected_note": "Check whether retrieved passages explain investment-related losses/special items.",
		},
		{
				"id": "bmw_group_revenue_2023",
				"question": "What was BMW Group's total revenue in 2023?",
				# Deliberately stricter than just 'BMW': this can expose BMW Finance vs
				# BMW Group source confusion. Adjust after checking actual filenames.
				"source_any": ["bmw"],
				"evidence_any": ["revenue", "sales"],
				"expected_note": "Manually verify the report is BMW Group consolidated reporting, not BMW Finance N.V.",
		},
		{
				"id": "ford_f150_rate_hike",
				"question": "How might a central-bank rate hike affect demand for Ford's F-150 Lightning?",
				"source_any": ["ford", "news"],
				"evidence_any": ["interest rate", "F-150 Lightning", "financing", "demand"],
				"expected_note": "Cross-document causal question; retrieval may find only one side of the evidence.",
		},
		{
				"id": "rivian_deliveries_2022",
				"question": "How many vehicles did Rivian deliver in 2022?",
				"source_any": ["rivian"],
				"evidence_any": ["deliveries", "vehicles"],
				"expected_note": "Unsupported by the supplied company reports unless a source explicitly mentions Rivian deliveries.",
		},
		{
				"id": "tesla_stock_price_2023",
				"question": "What was Tesla's closing stock price on 31 December 2023?",
				"source_any": ["tesla"],
				"evidence_any": ["stock price", "closing price", "share price"],
				"expected_note": "Likely unsupported by annual-report text; use as a retrieval/abstention diagnostic.",
		},
]

def norm(value: str) -> str:
		return re.sub(r"\s+", " ", value.casefold()).strip()

def matches_any(text: str, terms: list[str]) -> bool:
		if not terms:
				return True
		haystack = norm(text)
		return any(norm(term) in haystack for term in terms)

def get_cache_dir() -> str:
	return rag.cache_directory.get(os.getenv("USER", ""), os.getcwd())

def load_or_build_index(
	embedding_model_name: str,
	chunk_size: int,
	chunk_overlap: int,
	rebuild: bool,
):
	data_dir = os.environ.get("FINRAG_DATA_DIR", rag.DATA_DIR)
	cache_key = rag.compute_cache_key(
		data_dir=data_dir,
		embed_model=embedding_model_name,
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)
	index_dir = os.path.join(
			get_cache_dir(),
			f"index_{embedding_model_name.replace('/', '_')}_{cache_key}",
	)
	os.makedirs(index_dir, exist_ok=True)
	chunks_path = os.path.join(index_dir, "chunks.pkl")
	embeddings_path = os.path.join(index_dir, "embeddings.npy")
	meta_path = os.path.join(index_dir, "meta.json")
	valid_cache = (
			not rebuild
			and os.path.isfile(chunks_path)
			and os.path.isfile(embeddings_path)
			and os.path.isfile(meta_path)
	)
	if valid_cache:
			try:
					with open(meta_path, encoding="utf-8") as f:
							meta = json.load(f)
					valid_cache = meta.get("cache_key") == cache_key
			except (OSError, json.JSONDecodeError):
					valid_cache = False
	if valid_cache:
			print(f"[index] Loading cached index: {index_dir}")
			with open(chunks_path, "rb") as f:
					chunks = pickle.load(f)
			embeddings = np.load(embeddings_path)
	else:
			print(f"[index] Building index from {data_dir}")
			chunks = rag.load_and_process_documents(data_dir, chunk_size, chunk_overlap)
			attention, dtype = rag.get_model_kwargs(verbose=False)
			model = rag._load(
					model_id=embedding_model_name,
					attention=attention,
					dtype=dtype,
					cache_folder=get_cache_dir(),
					token=os.getenv("HUGGINGFACE_TOKEN"),
			)
			embeddings = rag.build_index(model, chunks)
			with open(chunks_path, "wb") as f:
					pickle.dump(chunks, f)
			np.save(embeddings_path, embeddings)
			with open(meta_path, "w", encoding="utf-8") as f:
					json.dump(
							{
									"cache_key": cache_key,
									"embedding_model": embedding_model_name,
									"chunk_size": chunk_size,
									"chunk_overlap": chunk_overlap,
									"n_chunks": len(chunks),
							},
							f,
							indent=2,
					)
			return model, chunks, embeddings
	attention, dtype = rag.get_model_kwargs(verbose=False)
	model = rag._load(
			model_id=embedding_model_name,
			attention=attention,
			dtype=dtype,
			cache_folder=get_cache_dir(),
			token=os.getenv("HUGGINGFACE_TOKEN"),
	)
	return model, chunks, embeddings


def evaluate_case(
		model,
		chunks: list[dict[str, Any]],
		embeddings: np.ndarray,
		case: dict[str, Any],
		k_values: list[int],
		show_chunks: bool = False,
) -> dict[str, Any]:
		max_k = min(max(k_values), len(chunks))
		retrieved = rag.retrieve(model, embeddings, case["question"], chunks, max_k)

		result: dict[str, Any] = {
				"id": case["id"],
				"question": case["question"],
				"top_k": {},
		}

		for k in k_values:
				top = retrieved[: min(k, len(retrieved))]
				source_texts = [str(doc.get("source", "")) for doc, _ in top]
				chunk_texts = [str(doc.get("text", "")) for doc, _ in top]
				combined_sources = "\n".join(source_texts)
				combined_chunks = "\n".join(chunk_texts)

				source_hit = matches_any(combined_sources, case.get("source_any", []))
				evidence_hit = matches_any(combined_chunks, case.get("evidence_any", []))
				result["top_k"][str(k)] = {
						"source_hit": source_hit,
						"evidence_hit": evidence_hit,
						"both_hit": source_hit and evidence_hit,
						"top_score": top[0][1] if top else None,
				}

		if show_chunks:
				result["chunks"] = [
						{
								"rank": rank,
								"score": round(score, 5),
								"source": os.path.basename(doc.get("source", "")),
								"page": int(doc.get("page", -1)) + 1,
								"text": str(doc.get("text", ""))[:700],
						}
						for rank, (doc, score) in enumerate(retrieved[:max_k], start=1)
				]
		return result


def summarize(results: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
		summary: dict[str, Any] = {"n_cases": len(results), "by_k": {}}
		for k in k_values:
				for metric in ("source_hit", "evidence_hit", "both_hit"):
						hits = sum(bool(r["top_k"][str(k)][metric]) for r in results)
						summary["by_k"].setdefault(str(k), {})[metric] = {
								"hits": hits,
								"total": len(results),
								"rate": round(hits / len(results), 4) if results else 0.0,
						}
		return summary


def main() -> int:
		parser = argparse.ArgumentParser(description="Evaluate finRAG retrieval against a small diagnostic set.")
		parser.add_argument("--embedding_model", "-emb", default="google/embeddinggemma-2")
		parser.add_argument("--data_dir", default=rag.DATA_DIR)
		parser.add_argument("--chunk_size", type=int, default=800)
		parser.add_argument("--chunk_overlap", type=int, default=120)
		parser.add_argument("--top_k", type=int, nargs="+", default=[5, 10, 25])
		parser.add_argument("--rebuild_index", action="store_true")
		parser.add_argument("--show_chunks", action="store_true")
		parser.add_argument("--output", default="eval_results.json")
		parser.add_argument("--cases", default=None, help="Optional JSON file replacing the built-in evaluation cases.")
		args = parser.parse_args()

		if not args.top_k or any(k < 1 for k in args.top_k):
				parser.error("--top_k values must all be positive integers")

		data_dir = Path(args.data_dir)
		if not data_dir.is_dir():
				parser.error(f"Data directory not found: {data_dir.resolve()}")
		# finRAG's cache key/index builder uses rag.DATA_DIR. Set it to the chosen
		# directory so --data_dir behaves consistently.
		rag.DATA_DIR = str(data_dir)

		cases = EVAL_SET
		if args.cases:
				with open(args.cases, encoding="utf-8") as f:
						cases = json.load(f)
				if not isinstance(cases, list) or not cases:
						parser.error("--cases must point to a non-empty JSON list")

		attention, dtype = rag.get_model_kwargs(verbose=True)
		model, chunks, embeddings = load_or_build_index(
				embedding_model_name=args.embedding_model,
				chunk_size=args.chunk_size,
				chunk_overlap=args.chunk_overlap,
				rebuild=args.rebuild_index,
		)
		print(f"[ready] chunks={len(chunks)}, embeddings={embeddings.shape}, cases={len(cases)}")

		results = []
		for i, case in enumerate(cases, start=1):
				print(f"\n[{i}/{len(cases)}] {case['id']}: {case['question']}")
				result = evaluate_case(
						model, chunks, embeddings, case, sorted(set(args.top_k)), args.show_chunks
				)
				results.append(result)
				for k in sorted(set(args.top_k)):
						m = result["top_k"][str(k)]
						print(
								f"k: {k:3d} source_hit={int(m['source_hit'])} "
								f"evidence_hit={int(m['evidence_hit'])} both_hit={int(m['both_hit'])} "
								f"top_score={m['top_score']}"
								if m["top_score"] is not None
								else f"  k={k:>2}: no results"
						)
				if args.show_chunks:
						for item in result.get("chunks", []):
								print(
										f"{item['rank']:4d} score={item['score']:.6f} "
										f"{item['source']:<30} p{item['page']:3d} {item['text'][:250]!r}"
								)

		report = {
				"config": {
						"embedding_model": args.embedding_model,
						"data_dir": str(data_dir.resolve()),
						"chunk_size": args.chunk_size,
						"chunk_overlap": args.chunk_overlap,
						"top_k": sorted(set(args.top_k)),
						"n_chunks": len(chunks),
				},
				"summary": summarize(results, sorted(set(args.top_k))),
				"cases": results,
		}

		output_path = Path(os.path.join(rag.cache_directory.get(os.getenv("USER"), os.getcwd()), args.output))
		output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
		print("\n=== Retrieval summary ===")
		print(json.dumps(report["summary"], indent=2))
		print(f"\n[written] {output_path.resolve()}")
		print(
				"\nInterpretation: source_hit/evidence_hit are lightweight diagnostic heuristics, "
				"not semantic ground truth. Review failed cases and the top chunks before changing "
				"the retriever. This script does not score generated answers."
		)
		return 0


if __name__ == "__main__":
		raise SystemExit(main())
