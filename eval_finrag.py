import argparse
import json
import re
import os
import time
import torch
import rich
import rich.table

import finrag as F

CASES = [
	{
		"id": "bmw_pbt_2020_partial", "type": "partial-answer",
		"question": "What were BMW's profit figures for 2020 and 2023?",
		"need": [("BMW", [2021])],
		"expect": [["5222"]],
	},
	{
		"id": "bmw_pbt_2021", "type": "single-fact",
		"question": "What was BMW's profit before tax in 2021?",
		"need": [("BMW", [2021])],
		"expect": [["16060"]],
	},
	{
		"id": "tesla_rev_2023", "type": "single-fact",
		"question": "How much revenue did Tesla generate in 2023?",
		"need": [("Tesla", [2023])],
		"expect": [["96773", "96.8", "96.77"]],
	},
	{
		"id": "ford_rev_2022", "type": "single-fact",
		"question": "What was Ford's revenue for the year 2022?",
		"need": [("Ford", [2022, 2023])],
		"expect": [["158057", "158.1", "158.06"]],
	},
	{
		"id": "tesla_vs_ford_rev_2022", "type": "cross-document",
		"question": "Compare the 2022 revenue of Tesla and Ford.",
		"need": [("Tesla", [2022, 2023]), ("Ford", [2022, 2023])],
		"expect": [["81462", "81.5"], ["158057", "158.1", "158.06"]],
	},
	{
		"id": "tesla_energy_segment", "type": "table-heavy",
		"question": "What were Tesla's revenues from the 'Energy generation and storage' segment in 2021, 2022, and 2023?",
		"need": [("Tesla", [2022, 2023])],
		"expect": [["2789", "2.8"], ["3909", "3.9"], ["6035", "6.0"]],
	},
	{
		"id": "ford_net_loss_2022_why", "type": "explanatory",
		"question": "Why did Ford report a net loss in 2022 despite strong automotive sales?",
		"need": [("Ford", [2022, 2023])],
		"expect": None,
	},
	{
		"id": "distractor_rivian", "type": "distractor",
		"question": "How many vehicles did Rivian deliver in 2022?",
		"need": None, "expect": None, "expect_missing": True,
	},
	{
		"id": "distractor_stock_price", "type": "distractor",
		"question": "What was Tesla's stock price on December 31, 2023?",
		"need": None, "expect": None, "expect_missing": True,
	},
	{
		"id": "distractor_central_bank", "type": "distractor",
		"question": "How did the central bank's 0.5% interest rate hike affect Ford's F-150 Lightning sales?",
		"need": None, "expect": None, "expect_missing": True,
	},
]

CITE_RE = re.compile(r"\[([^\[\],]+?\.pdf),\s*p\.?\s*(\d+)\]", re.IGNORECASE)


def norm(s: str) -> str:
	return re.sub(r"[,\s$€]", "", s.lower())

def retrieval_hit(retrieved, need) -> bool:
	for company, years in need:
		if not any(
			d.get("company") == company and (years is None or d.get("report_year") in years)
			for d, _ in retrieved
		):
			return False
	return True

def numeric_ok(answer: str, expect) -> bool:
	a = norm(answer)
	return all(any(norm(v) in a for v in group) for group in expect)

def figure_in_context(retrieved, expect) -> bool:
	text = norm(" ".join(d["text"] for d, _ in retrieved))
	return all(any(norm(v) in text for v in group) for group in expect)

def citation_ok(answer: str, kept) -> "bool | None":
	cites = CITE_RE.findall(answer)
	if not cites:
		return False
	seen = {
		(os.path.basename(d["source"]).lower(), p)
		for d in kept
		for p in (d.get("pages") or [d.get("page")])
	}
	return all((f.strip().lower(), int(p)) in seen for f, p in cites)

def generate(tokenizer, llm, query, retrieved, max_seq_len, max_new_tokens):
	# same budgeting + prompt construction as finRAG.main(), without chat history
	probe = [
		{"role": "system", "content": F.TEMPLATE__PROMPT.format(context="")},
		{"role": "user", "content": query},
	]
	probe_text = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
	overhead = len(tokenizer.encode(probe_text, add_special_tokens=False))
	budget = max(0, max_seq_len - overhead - max_new_tokens - 64)

	context, kept = F.build_context_within_budget(retrieved, tokenizer, budget)
	msgs = [
		{"role": "system", "content": F.TEMPLATE__PROMPT.format(context=context)},
		{"role": "user", "content": query},
	]
	prompt_text = tokenizer.apply_chat_template(
		msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
	)
	inputs = tokenizer(
		prompt_text, return_tensors="pt", truncation=True, max_length=max_seq_len
	).to(llm.device)
	with torch.inference_mode():
		out = llm.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			do_sample=False,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)
	answer = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
	return F.clean_response(answer), kept

def mark(v):
	return "–" if v is None else ("[green]✓[/green]" if v else "[red]✗[/red]")

def rate(vals):
	vals = [v for v in vals if v is not None]
	return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

def main():
	ap = argparse.ArgumentParser(description="finRAG evaluation")
	ap.add_argument("--llm_model", "-llm", default="Qwen/Qwen3.5-0.8B")
	ap.add_argument("--embedding_model", "-emb", default="Qwen/Qwen3-Embedding-0.6B")
	ap.add_argument("--batch_size", "-bs", type=int, default=8)
	ap.add_argument("--top_k", type=int, default=8)
	ap.add_argument("--max_seq_len", type=int, default=8192)
	ap.add_argument("--max_new_tokens", type=int, default=512)
	ap.add_argument("--load_in_4bit", action="store_true")
	ap.add_argument("--retrieval_only", action="store_true", help="skip the LLM")
	ap.add_argument("--out", default="eval_results.json")
	args = ap.parse_args()

	emb_model, chunks, embeddings = F.get_document_embedding(
		model_id=args.embedding_model, batch_size=args.batch_size, verbose=False
	)
	tokenizer = llm = None
	if not args.retrieval_only:
		tokenizer, llm = F.get_llm(args.llm_model, load_in_4bit=args.load_in_4bit)

	rows = []
	for case in CASES:
		t0 = time.time()
		retrieved = F.retrieve(
			model=emb_model, embeddings=embeddings, query=case["question"],
			chunks=chunks, k=args.top_k, verbose=False,
		)
		r = {"id": case["id"], "type": case["type"], "question": case["question"]}
		r["retrieval_hit"] = retrieval_hit(retrieved, case["need"]) if case.get("need") else None

		r["figure_in_retrieved"] = (
			figure_in_context(retrieved, case["expect"]) if case.get("expect") else None
		)



		if args.retrieval_only:
			r.update(numeric_ok=None, refusal_ok=None, citation_ok=None, answer="")
		else:
			answer, kept = generate(
				tokenizer, llm, case["question"], retrieved,
				args.max_seq_len, args.max_new_tokens,
			)
			missing = "DATA_MISSING" in answer
			expect_missing = case.get("expect_missing", False)
			r["answer"] = answer
			r["refusal_ok"] = missing == expect_missing
			r["numeric_ok"] = (
				numeric_ok(answer, case["expect"]) if case.get("expect") and not missing else
				(False if case.get("expect") else None)
			)
			r["citation_ok"] = None if (expect_missing or missing) else citation_ok(answer, kept)
		r["seconds"] = round(time.time() - t0, 1)
		rows.append(r)

	# ---- report ------------------------------------------------------------
	table = rich.table.Table(title="finRAG evaluation")
	for col in ("case", "type", "retrieval", "numeric", "refusal", "citation", "sec"):
		table.add_column(col)
	for r in rows:
		table.add_row(
			r["id"], r["type"], mark(r["retrieval_hit"]), mark(r["numeric_ok"]),
			mark(r["refusal_ok"]), mark(r["citation_ok"]), str(r["seconds"]),
		)
	F.console.print(table)

	summary = {}
	for key in ("retrieval_hit", "figure_in_retrieved", "numeric_ok", "refusal_ok", "citation_ok"):
		val, n = rate([r[key] for r in rows])
		summary[key] = None if val is None else round(val, 3)
		if val is not None:
			F.console.print(f"{key:>14}: {val:.0%}  ({n} cases)")

	if not args.retrieval_only:
		failed = [r for r in rows if False in (r["numeric_ok"], r["refusal_ok"], r["citation_ok"])]
		for r in failed:
			F.console.print(f"\n[bold red]FAILED[/bold red] {r['id']}: {r['question']}")
			F.console.print(r["answer"][:600])

	with open(args.out, "w", encoding="utf-8") as f:
		json.dump({"args": vars(args), "summary": summary, "cases": rows}, f, indent=2, ensure_ascii=False)
	print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
	main()