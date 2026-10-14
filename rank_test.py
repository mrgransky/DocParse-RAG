#!/usr/bin/env python3
"""rank_test.py: ranking smoke test with hard negatives for a VL page embedder."""
import os
import glob
import math
import argparse
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

try:
	from sentence_transformers.sentence_transformer.modules import Normalize
except ImportError:
	try:
		from sentence_transformers.base.modules import Normalize
	except ImportError:
		from sentence_transformers.models import Normalize

if not getattr(Normalize, "_rogue_kwargs_patched", False):
	_orig_norm_init = Normalize.__init__
	def _patched_norm_init(self, *args, **kwargs):
		kwargs.pop("normalize_embeddings", None)
		return _orig_norm_init(self, *args, **kwargs)
	Normalize.__init__ = _patched_norm_init
	Normalize._rogue_kwargs_patched = True

cache_directory = {
	"farid": "/home/farid/datasets/models",
	"alijanif": "/scratch/project_2004072/models",
	"ubuntu": "/media/volume/models",
}

# (query, relevant pages, what makes it hard). Empty set = answer is on no page.
QUERIES = [
	("What were Tesla's total assets in 2021?", {"Tesla_2021_p19.png"}, "cross-company"),
	("Tesla total liabilities as of December 31, 2022", {"Tesla_2021_p19.png"}, "metric next to 'total assets'"),
	("BMW Group revenues in 2017", {"BMW_2021_p10.png"}, "year: 2017 column only in the 2021 report"),
	("BMW Group net profit in 2022", {"BMW_2022_p10.png"}, "year: near-identical BMW 2021 layout"),
	("BMW Financial Services profit before tax in 2022", {"BMW_2022_p10.png"}, "segment row + year"),
	("BMW Group revenues in 2021", {"BMW_2021_p10.png", "BMW_2022_p10.png"}, "both BMW pages contain 2021"),
	("Ford market share in Germany in 2021", {"Ford_2021_p5.png"}, "single table cell"),
	("Ford U.S. retail sales of SUVs in 2021", {"Ford_2021_p5.png"}, "second table on the page"),
	("BMW Group total assets 2021", set(), "unanswerable: 'total assets' only on the Tesla page"),
	("Ford net income 2021", set(), "unanswerable"),
]

def load_image(path: str, max_pixels: int) -> Image.Image:
	img = Image.open(path).convert("RGB")
	w, h = img.size
	if max_pixels and w * h > max_pixels:
		s = math.sqrt(max_pixels / (w * h))
		img = img.resize((max(32, int(w * s)), max(32, int(h * s))), Image.LANCZOS)
	return img

def encode(model, items, side: str, mode: str, query_prompt, bs: int):
	kw = dict(batch_size=bs, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
	if mode == "plain":  # your original test
		return model.encode(items, **kw)
	if side == "query":
		if query_prompt:
			return model.encode(items, prompt=query_prompt, **kw)
		return model.encode_query(items, **kw)
	return model.encode_document(items, **kw)

def main():
	p = argparse.ArgumentParser()
	p.add_argument("-d", "--image_dir", default=".")
	p.add_argument("-emb", "--embedding_model", default="Qwen/Qwen3-VL-Embedding-2B")
	p.add_argument("--max_pixels", type=int, default=1_000_000)
	p.add_argument("--mode", choices=["plain", "asym"], default="asym",
		help="plain = model.encode on both sides; asym = encode_query / encode_document")
	p.add_argument("--query_prompt", default=None, help="explicit query instruction (overrides model prompts)")
	p.add_argument("-bs", "--batch_size", type=int, default=1)
	args = p.parse_args()

	paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
	names = [os.path.basename(x) for x in paths]
	missing = {n for _, rel, _ in QUERIES for n in rel} - set(names)
	if missing:
		raise SystemExit(f"Missing labelled pages: {sorted(missing)}")

	cuda = torch.cuda.is_available()
	dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
	model = SentenceTransformer(
		args.embedding_model,
		device="cuda:0" if cuda else "cpu",
		trust_remote_code=True,
		cache_folder=cache_directory.get(os.getenv("USER"), os.getcwd()),
		token=os.getenv("HUGGINGFACE_TOKEN"),
		model_kwargs={"dtype": dtype, "attn_implementation": "sdpa"},
		processor_kwargs={"padding_side": "left"},
	)
	print(f"Model: {args.embedding_model} | mode={args.mode} | max_pixels={args.max_pixels:,}")
	print(f"Configured prompts: {getattr(model, 'prompts', {})}")

	images = [load_image(x, args.max_pixels) for x in paths]
	print(f"\n{len(names)} pages:")
	for n, im in zip(names, images):
		print(f"  {n:<26} {im.size[0]}x{im.size[1]}")

	doc_emb = encode(model, images, "document", args.mode, args.query_prompt, args.batch_size)
	q_emb = encode(model, [q for q, _, _ in QUERIES], "query", args.mode, args.query_prompt, args.batch_size)
	sims = model.similarity(q_emb, doc_emb).float().cpu()

	hits1, rrs, margins, top_ans, top_unans = [], [], [], [], []
	for qi, (q, rel, why) in enumerate(QUERIES):
		s = sims[qi]
		order = torch.argsort(s, descending=True).tolist()
		print(f"\nQ{qi + 1}: {q}\n    [{why}]")
		for r, j in enumerate(order, 1):
			tag = "✓" if names[j] in rel else " "
			print(f"   {r}. {tag} {float(s[j]):.4f}  {names[j]}")

		if not rel:
			top_unans.append(float(s.max()))
			print(f"   -> no relevant page; top score {float(s.max()):.4f} (lower is better)")
			continue

		first = next(r for r, j in enumerate(order, 1) if names[j] in rel)
		best_rel = max(float(s[j]) for j in range(len(names)) if names[j] in rel)
		negs = [float(s[j]) for j in range(len(names)) if names[j] not in rel]
		margin = best_rel - max(negs) if negs else float("nan")
		hits1.append(first == 1)
		rrs.append(1.0 / first)
		margins.append(margin)
		top_ans.append(float(s.max()))
		flag = "OK" if margin > 0.02 else ("THIN" if margin > 0 else "FAIL")
		print(f"   -> first relevant @ rank {first} | margin {margin:+.4f} [{flag}]")

	n = len(hits1)
	print("\n" + "=" * 60)
	print(f"Hit@1          : {sum(hits1)}/{n} = {sum(hits1) / n:.2f}")
	print(f"MRR            : {sum(rrs) / n:.3f}")
	print(f"Margin mean/min: {sum(margins) / n:+.4f} / {min(margins):+.4f}")
	if top_unans:
		print(f"Top score      : answerable {sum(top_ans) / len(top_ans):.4f} "
			f"vs unanswerable {sum(top_unans) / len(top_unans):.4f}")

if __name__ == "__main__":
	main()