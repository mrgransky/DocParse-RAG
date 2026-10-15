# from sentence_transformers import SentenceTransformer
# import os
# import glob

# cache_directory = {
# 	"farid": "/home/farid/datasets/models",
# 	"alijanif": "/scratch/project_2004072/models",
# 	"ubuntu": "/media/volume/models",
# }

# # Load the model
# model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", cache_folder=cache_directory.get(os.getenv('USER')))

# queries = ["Assets of Tesla in 2021"]

# documents = [
#   "BMW_2021_p10.png",
#   "BMW_2022_p10.png",
#   "Ford_2021_p5.png",
#   "Tesla_2021_p19.png",
# ]

# # original implemetation available at Qwen huggingface repo
# # # Text queries
# # queries = [
# # 		"A woman playing with her dog on a beach at sunset.",
# # 		"Pet owner training dog outdoors near water.",
# # 		"Woman surfing on waves during a sunny day.",
# # 		"City skyline view from a high-rise building at night.",
# # ]

# # # Documents: text, image, and text+image
# # documents = [
# # 		"A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
# # 		"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
# # 		{"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
# # ]

# # Encode queries and documents
# query_embeddings = model.encode(queries)
# doc_embeddings = model.encode(documents)
# print(query_embeddings.shape, doc_embeddings.shape)

# similarities = model.similarity(query_embeddings, doc_embeddings)
# print(similarities)


# python finrag_vlm.py -d /scratch/project_2004072/datasets/automotive_finance -vlm "Qwen/Qwen3.8-27B" -emb "Qwen/Qwen3-VL-Embedding-8B" --dpi 200 -bs 16 -v


"""
finrag_vlm.py — ReportVision-RAG
Visual RAG for financial reports:
  PDFs -> page images (pypdfium2) -> VL embeddings -> ranking [+ optional BM25 fusion / VL reranker]
       -> top pages (images) -> VLM answer with [file, p. N] citations

Requirements (tested API levels, verify on your env):
  pip install -U pypdfium2 pillow "sentence-transformers[image]>=5.4" "transformers>=4.57" accelerate rich numpy
  (optional) pip install bitsandbytes flash-attn

Examples:
  # build index only (e.g. as a batch job)
  python finrag_vlm.py -d /path/to/pdfs --index_only -v
  # chat (whole-page index, hybrid text fusion, reranker)
  python finrag_vlm.py -d /path/to/pdfs --hybrid --reranker_model Qwen/Qwen3-VL-Reranker-2B
  # page strips instead of / in addition to whole pages
  python finrag_vlm.py -d /path/to/pdfs --index_mode both --tile_rows 3
  # evaluation
  python finrag_vlm.py -d /path/to/pdfs --eval_file questions.jsonl --eval_generate
"""
import os
import gc
import re
import json
import math
import hashlib
import pathlib
import argparse
import collections
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import packaging.version
import pypdfium2 as pdfium
from PIL import Image
import rich
import rich.console
import rich.panel
import rich.prompt
import rich.table
from rich.progress import track
import transformers as tfs
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Normalize kwargs patch (kept from finrag.py)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
cache_directory = {
	"farid": "/home/farid/datasets/models",
	"alijanif": "/scratch/project_2004072/models",
	"ubuntu": "/media/volume/models",
}
console = rich.console.Console()

MAX_HISTORY_TURNS = 5
COMPANY_MAP = {"bmw": "BMW", "ford": "Ford", "tesla": "Tesla"}
ENTITY_MAP = {
	"BMW_Annual_Report_2021.pdf": "BMW Group",
	"BMW_Annual_Report_2022.pdf": "BMW Finance N.V. (subsidiary)",
	"BMW_Annual_Report_2023.pdf": "BMW Finance N.V. (subsidiary)",
}

DEFAULT_QUERY_INSTRUCTION = (
	"Given a question about a company's annual report, retrieve the report pages "
	"(text, tables, or charts) that contain the answer."
)

TEMPLATE__PROMPT = """\
You are a financial analyst specializing in the analysis of Annual Reports within the automotive sector.
You are given rendered PAGE IMAGES from annual reports. Each image is preceded by a header naming
its source file, company, reporting entity, report year, and page number.

EVIDENCE AND ACCURACY RULES
1. Answer using only what is visible on the supplied pages. Do not introduce outside facts.
2. Answer the user's actual question directly and concisely.
3. Do not reveal internal reasoning or describe your thought process.
4. Read figures exactly as printed. Preserve company, reporting period, currency, units
   (e.g. "in € million"), accounting metric, and sign (parentheses usually mean negative).
   Check column headers (year / period) and row labels before quoting a value.
5. If a needed value is not clearly legible, say so instead of guessing.
6. Distinguish among revenue, operating income, net income, and adjusted EBIT.
7. Distinguish consolidated figures from figures attributable to a parent or its shareholders.
   If the header names a subsidiary rather than the group, say so explicitly and do not present
   its figures as group-wide results.
8. Cite supporting evidence using the source filename and page number from the page header,
   e.g. [Ford_Annual_Report_2023.pdf, p. 54].
9. For comparisons, address each company and period separately; if only part is supported,
   answer that part and state what is unavailable.
10. Do not claim causal relationships unless the pages support them.
11. Do not perform arithmetic unless explicitly asked; if asked, show formula and inputs.
12. If the pages contain no evidence relevant to the question, respond with exactly: DATA_MISSING.
    If only some parts are supported, answer those and state which parts are unavailable.
"""

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def get_user_cache_dir() -> str:
	return cache_directory.get(os.getenv("USER"), os.getcwd())

def get_file_metadata(path: str) -> dict:
	p = pathlib.Path(path)
	company = None
	for part in p.parts:
		if part.lower() in COMPANY_MAP:
			company = COMPANY_MAP[part.lower()]
			break
	if company is None:
		for k, v in COMPANY_MAP.items():
			if k in p.stem.lower():
				company = v
				break
	m = re.search(r"(19|20)\d{2}", p.stem)
	is_annual = "annual_report" in p.stem.lower() or company is not None
	return {
		"company": company,
		"report_year": int(m.group(0)) if m else None,
		"document_type": "annual_report" if is_annual else "news",
		"entity": ENTITY_MAP.get(p.name),
	}

def atomic_write_json(obj: Any, path: str) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	tmp = path + ".tmp"
	try:
		with open(tmp, "w", encoding="utf-8") as f:
			json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
		os.replace(tmp, path)
	except Exception:
		if os.path.exists(tmp):
			os.remove(tmp)
		raise

def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	tmp = path + ".tmp"
	with open(tmp, "w", encoding="utf-8") as f:
		for r in records:
			f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
	os.replace(tmp, path)

def list_pdfs(data_dir: str) -> List[str]:
	out = []
	for root, _, files in os.walk(data_dir):
		for fn in sorted(files):
			if fn.lower().endswith(".pdf"):
				out.append(os.path.join(root, fn))
	return sorted(out)

def pdf_fingerprint(path: str) -> str:
	st = os.stat(path)
	return f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"

def fit_to_pixel_budget(img: Image.Image, max_pixels: int) -> Image.Image:
	"""Downscale (never upscale) so that w*h <= max_pixels, preserving aspect ratio."""
	w, h = img.size
	if max_pixels <= 0 or w * h <= max_pixels:
		return img
	s = math.sqrt(max_pixels / float(w * h))
	return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)

def clean_response(text: str) -> str:
	return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def free_cuda():
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

def get_model_kwargs(verbose: bool = False):
	dtype, attention = torch.float32, "eager"
	if torch.cuda.is_available():
		dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
		major, minor = torch.cuda.get_device_capability()
		compute_cap = major + minor / 10
		if compute_cap >= 8.0:
			try:
				import flash_attn  # noqa: F401
				attention = "flash_attention_2"
			except ImportError:
				if verbose:
					print("[WARN] Flash Attention 2 not installed (pip install flash-attn)")
		if (attention == "eager" and compute_cap >= 7.0 and
				packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0")):
			attention = "sdpa"
	if verbose:
		cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
		print(f"\n[MODEL KWARGS] CUDA={torch.version.cuda} | torch={torch.__version__} | "
					f"compute_cap={cc} | attention={attention} | dtype={dtype}")
	return attention, dtype

# ---------------------------------------------------------------------------
# 1) Rendering (pypdfium2) — shared across embedding models
# ---------------------------------------------------------------------------
def render_corpus(
	data_dir: str,
	render_root: str,
	dpi: int,
	image_format: str = "png",
	extract_text: bool = True,
	verbose: bool = False,
) -> List[Dict[str, Any]]:
	"""Render every PDF page to an image file. Per-document cache keyed by file fingerprint + dpi."""
	os.makedirs(render_root, exist_ok=True)
	scale = dpi / 72.0
	pages: List[Dict[str, Any]] = []

	for path in list_pdfs(data_dir):
		doc_id = hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:12]
		out_dir = os.path.join(render_root, doc_id)
		manifest_path = os.path.join(out_dir, "_pages.json")
		fp = pdf_fingerprint(path)

		if os.path.exists(manifest_path):
			try:
				with open(manifest_path, encoding="utf-8") as f:
					man = json.load(f)
				if man.get("fingerprint") == fp and man.get("dpi") == dpi and \
						all(os.path.exists(p["image_path"]) for p in man["pages"]):
					pages.extend(man["pages"])
					if verbose:
						print(f"[CACHE] {os.path.basename(path)}: {len(man['pages'])} pages")
					continue
			except Exception as e:
				print(f"<!> Bad render manifest for {path} ({e}); re-rendering.")

		os.makedirs(out_dir, exist_ok=True)
		meta = get_file_metadata(path)
		doc_pages = []
		try:
			pdf = pdfium.PdfDocument(path)
		except Exception as e:
			print(f"<!> Failed to open {path}: {e}")
			continue
		try:
			n = len(pdf)
			for i in track(range(n), description=f"Rendering {os.path.basename(path)}", console=console):
				page = pdf[i]
				try:
					img = page.render(scale=scale).to_pil().convert("RGB")
					page_no = i + 1
					img_path = os.path.join(out_dir, f"p{page_no:04d}.{image_format}")
					if image_format == "jpg":
						img.save(img_path, quality=92)
					else:
						img.save(img_path)
					text = ""
					if extract_text:
						tp = page.get_textpage()
						text = tp.get_text_range() or ""
						tp.close()
					doc_pages.append({
						"page_id": f"{doc_id}:p{page_no:04d}",
						"doc_id": doc_id,
						"source": path,
						"filename": os.path.basename(path),
						"page": page_no,
						"n_pages": n,
						"image_path": img_path,
						"width_px": img.width,
						"height_px": img.height,
						"text": text,  # native text layer (may be noisy) — only for optional lexical fusion
						**meta,
					})
				except Exception as e:
					print(f"<!> Failed page {i + 1} of {path}: {e}")
				finally:
					page.close()
		finally:
			pdf.close()

		atomic_write_json({"fingerprint": fp, "dpi": dpi, "pages": doc_pages}, manifest_path)
		pages.extend(doc_pages)

	# corpus summary (useful diagnostics: pages without text layer are likely scanned/image-only)
	atomic_write_json({
		"n_pages": len(pages),
		"n_documents": len({p["source"] for p in pages}),
		"dpi": dpi,
		"n_pages_without_text_layer": sum(len(p["text"].strip()) < 20 for p in pages),
		"mean_size_px": [
			float(np.mean([p["width_px"] for p in pages])) if pages else 0,
			float(np.mean([p["height_px"] for p in pages])) if pages else 0,
		],
	}, os.path.join(render_root, "summary.json"))
	print(f"[RENDER] {len(pages):,} pages from {len({p['source'] for p in pages})} PDFs @ {dpi} dpi")
	return pages

def load_page_image(page: Dict[str, Any], max_pixels: int, bbox: Optional[List[int]] = None) -> Image.Image:
	img = Image.open(page["image_path"]).convert("RGB")
	if bbox:
		img = img.crop(tuple(bbox))
	return fit_to_pixel_budget(img, max_pixels)

# ---------------------------------------------------------------------------
# 2) Index units: whole pages and/or overlapping horizontal strips
# ---------------------------------------------------------------------------
def build_units(pages: List[Dict[str, Any]], index_mode: str, tile_rows: int, tile_overlap: float):
	units = []
	for pi, p in enumerate(pages):
		if index_mode in ("page", "both"):
			units.append({"unit_id": f"{p['page_id']}:full", "page_idx": pi, "kind": "page", "bbox": None})
		if index_mode in ("tiles", "both") and tile_rows > 1:
			W, H = p["width_px"], p["height_px"]
			strip_h = H / (1 + (tile_rows - 1) * (1 - tile_overlap))
			for r in range(tile_rows):
				top = int(round(r * strip_h * (1 - tile_overlap)))
				bottom = H if r == tile_rows - 1 else min(H, int(round(top + strip_h)))
				units.append({
					"unit_id": f"{p['page_id']}:t{r}",
					"page_idx": pi, "kind": "tile", "bbox": [0, top, W, bottom],
				})
	return units

# ---------------------------------------------------------------------------
# 3) VL embedding model (sentence-transformers multimodal)
# ---------------------------------------------------------------------------
def _has_internal_normalize(m) -> bool:
	return hasattr(m, "modules") and any(isinstance(mod, Normalize) for mod in m.modules())

def st_encode(model, inputs: List[Any], role: str, batch_size: int,
							prompt: Optional[str] = None, show_progress_bar: bool = True) -> np.ndarray:
	"""role in {"query", "document"} -> uses encode_query / encode_document when available."""
	kw = {"batch_size": batch_size, "show_progress_bar": show_progress_bar,
				"convert_to_numpy": True, "precision": "float32"}
	if prompt:
		kw["prompt"] = prompt
	if not _has_internal_normalize(model):
		kw["normalize_embeddings"] = True
	fn = getattr(model, f"encode_{role}", None) or model.encode
	return np.asarray(fn(list(inputs), **kw), dtype=np.float32)

def load_embedder(model_id: str, truncate_dim: Optional[int] = None, verbose: bool = False) -> SentenceTransformer:
	attention, dtype = get_model_kwargs(verbose)

	def _build(attn):
		kw = dict(
			model_name_or_path=model_id,
			trust_remote_code=True,
			device="cuda:0" if torch.cuda.is_available() else "cpu",
			cache_folder=get_user_cache_dir(),
			token=os.getenv("HUGGINGFACE_TOKEN"),
			model_kwargs={"attn_implementation": attn, "dtype": dtype},
			processor_kwargs={"padding_side": "left"},
		)
		if truncate_dim:
			kw["truncate_dim"] = truncate_dim
		m = SentenceTransformer(**kw)
		# probes: text query + a tiny image (verifies image modality support)
		st_encode(m, ["probe"], "query", 1, show_progress_bar=False)
		st_encode(m, [Image.new("RGB", (64, 64), "white")], "document", 1, show_progress_bar=False)
		return m

	try:
		return _build(attention)
	except Exception as e:
		if attention == "flash_attention_2":
			print(f"[WARN] flash_attention_2 failed for {model_id} ({str(e)[:120]}...). Falling back to SDPA.")
			free_cuda()
			return _build("sdpa")
		raise RuntimeError(
			f"Could not load/probe '{model_id}' as an image-capable embedder. "
			f"Requires sentence-transformers>=5.4 with [image] extras and a multimodal model. Error: {e}"
		) from e

def compute_index_key(pdfs: List[str], args) -> str:
	h = hashlib.sha256()
	for part in (args.embedding_model, args.truncate_dim, args.emb_max_pixels, args.index_mode,
							 args.tile_rows, args.tile_overlap, args.dpi, args.image_format):
		h.update(str(part).encode())
	for p in pdfs:
		h.update(pdf_fingerprint(p).encode())
	return h.hexdigest()[:16]

def embed_units(embedder, pages, units, args, emb_path: str) -> np.ndarray:
	"""Embed index units in blocks; resumable via a partial file."""
	partial = emb_path.replace(".npy", ".partial.npy")
	done = np.load(partial) if os.path.exists(partial) else None
	start = 0 if done is None else len(done)
	chunks = [] if done is None else [done]
	block = max(args.batch_size * 16, 32)
	if start:
		print(f"[RESUME] {start}/{len(units)} units already embedded")

	for s in track(range(start, len(units), block), description="Embedding page images", console=console):
		batch_units = units[s:s + block]
		imgs = [load_page_image(pages[u["page_idx"]], args.emb_max_pixels, u["bbox"]) for u in batch_units]
		chunks.append(st_encode(embedder, imgs, "document", args.batch_size, show_progress_bar=False))
		np.save(partial, np.concatenate(chunks, axis=0))
		del imgs

	emb = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype=np.float32)
	np.save(emb_path, emb)
	if os.path.exists(partial):
		os.remove(partial)
	return emb

def get_visual_index(args):
	user_cache = get_user_cache_dir()
	pdfs = list_pdfs(args.dataset_directory)
	if not pdfs:
		raise SystemExit(f"No PDFs found under {args.dataset_directory}")

	data_hash = hashlib.sha256(os.path.abspath(args.dataset_directory).encode()).hexdigest()[:10]
	render_root = os.path.join(user_cache, "rendered_pages", f"data_{data_hash}_dpi{args.dpi}")
	pages = render_corpus(args.dataset_directory, render_root, args.dpi, args.image_format,
												extract_text=True, verbose=args.verbose)
	units = build_units(pages, args.index_mode, args.tile_rows, args.tile_overlap)

	key = compute_index_key(pdfs, args)
	index_dir = os.path.join(user_cache, f"vindex_{args.embedding_model.replace('/', '_')}_{key}")
	os.makedirs(index_dir, exist_ok=True)
	emb_path = os.path.join(index_dir, "embeddings.npy")
	meta_path = os.path.join(index_dir, "meta.json")
	units_path = os.path.join(index_dir, "units.jsonl")

	embedder = load_embedder(args.embedding_model, args.truncate_dim, args.verbose)

	need = args.rebuild_index or not os.path.exists(emb_path)
	if not need and os.path.exists(meta_path):
		with open(meta_path) as f:
			meta = json.load(f)
		if meta.get("index_key") != key or meta.get("n_units") != len(units):
			print("[WARN] Index metadata mismatch -> rebuilding.")
			need = True

	if need:
		if args.rebuild_index:
			partial = emb_path.replace(".npy", ".partial.npy")
			if os.path.exists(partial):
				os.remove(partial)
		print(f"[INDEX] Embedding {len(units):,} units ({args.index_mode}) with {args.embedding_model}")
		embeddings = embed_units(embedder, pages, units, args, emb_path)
		save_jsonl([{**u, "page_id": pages[u["page_idx"]]["page_id"],
								 "source": pages[u["page_idx"]]["source"],
								 "page": pages[u["page_idx"]]["page"]} for u in units], units_path)
		atomic_write_json({
			"index_key": key, "embedding_model": args.embedding_model, "n_units": len(units),
			"n_pages": len(pages), "dim": int(embeddings.shape[1]), "index_mode": args.index_mode,
			"dpi": args.dpi, "emb_max_pixels": args.emb_max_pixels,
			"tile_rows": args.tile_rows, "tile_overlap": args.tile_overlap,
		}, meta_path)
	else:
		embeddings = np.load(emb_path)
		print(f"[CACHED] Visual index: {embeddings.shape[0]:,} units x {embeddings.shape[1]} dims")

	free_cuda()
	return embedder, pages, units, embeddings

# ---------------------------------------------------------------------------
# 4) Optional lexical path (BM25 over native PDF text layer) for exact-value queries
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:[.,]\d+)*")

def tokenize(text: str) -> List[str]:
	return [t.lower().replace(",", "") for t in _TOKEN_RE.findall(text or "")]

class BM25:
	def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
		self.N, self.k1, self.b = len(docs), k1, b
		self.dl = np.array([len(d) for d in docs], dtype=np.float32)
		self.avgdl = float(self.dl.mean()) if self.N and self.dl.sum() else 1.0
		self.postings: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
		for i, d in enumerate(docs):
			for term, tf in collections.Counter(d).items():
				self.postings[term].append((i, tf))
		self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

	def get_scores(self, q_tokens: List[str]) -> np.ndarray:
		scores = np.zeros(self.N, dtype=np.float32)
		for t in set(q_tokens):
			if t not in self.postings:
				continue
			idf = self.idf[t]
			for i, tf in self.postings[t]:
				denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
				scores[i] += idf * tf * (self.k1 + 1) / denom
		return scores

# ---------------------------------------------------------------------------
# 5) Optional VL reranker
# ---------------------------------------------------------------------------
def load_reranker(model_id: str):
	from sentence_transformers import CrossEncoder
	attention, dtype = get_model_kwargs()
	return CrossEncoder(
		model_id,
		trust_remote_code=True,
		device="cuda:0" if torch.cuda.is_available() else "cpu",
		cache_folder=get_user_cache_dir(),
		token=os.getenv("HUGGINGFACE_TOKEN"),
		model_kwargs={"dtype": dtype, "attn_implementation": attention if attention != "flash_attention_2" else "sdpa"},
	)

# ---------------------------------------------------------------------------
# 6) Retrieval: units -> pages (max-pool), optional RRF fusion, optional rerank
# ---------------------------------------------------------------------------
def retrieve_pages(query, embedder, embeddings, units, pages, args,
									 bm25: Optional[BM25] = None, reranker=None, k: Optional[int] = None) -> List[Dict[str, Any]]:
	k = k or args.top_k
	instr = None if (args.query_instruction or "").lower() == "none" else args.query_instruction
	q = st_encode(embedder, [query], "query", 1, prompt=instr, show_progress_bar=False)[0]
	unit_scores = embeddings @ q  # cosine (normalized)

	# page score = best unit score on that page
	page_visual: Dict[int, Tuple[float, int]] = {}
	for ui in np.argsort(-unit_scores):
		pi = units[ui]["page_idx"]
		if pi not in page_visual:
			page_visual[pi] = (float(unit_scores[ui]), int(ui))
	visual_order = list(page_visual.keys())
	cand_n = max(args.candidate_k, k)

	lex_scores, fused = None, None
	if bm25 is not None:
		lex_scores = bm25.get_scores(tokenize(query))
		lex_order = [int(i) for i in np.argsort(-lex_scores) if lex_scores[i] > 0][:cand_n * 2]
		fused = collections.defaultdict(float)
		for r, pi in enumerate(visual_order[:cand_n * 2]):
			fused[pi] += 1.0 / (args.rrf_k + r + 1)
		for r, pi in enumerate(lex_order):
			fused[pi] += 1.0 / (args.rrf_k + r + 1)
		order = sorted(fused, key=fused.get, reverse=True)
	else:
		order = visual_order

	results = []
	for pi in order[:cand_n]:
		results.append({
			"page_idx": pi,
			"visual_score": page_visual[pi][0],
			"best_unit": units[page_visual[pi][1]],
			"lexical_score": float(lex_scores[pi]) if lex_scores is not None else None,
			"fused_score": float(fused[pi]) if fused is not None else None,
			"rerank_score": None,
		})

	if reranker is not None and results:
		imgs = [load_page_image(pages[r["page_idx"]], args.rerank_max_pixels) for r in results]
		scores = reranker.predict([(query, im) for im in imgs], batch_size=1, show_progress_bar=False)
		for r, s in zip(results, scores):
			r["rerank_score"] = float(s)
		results.sort(key=lambda r: r["rerank_score"], reverse=True)

	return results[:k]

def final_score(r: Dict[str, Any]) -> float:
	for key in ("rerank_score", "fused_score", "visual_score"):
		if r.get(key) is not None:
			return r[key]
	return 0.0

def show_retrieval(retrieved, pages, n_sent: Optional[int] = None, verbose: bool = False):
	t = rich.table.Table(title="Retrieved pages", show_lines=False)
	for col in ("#", "file", "page", "visual", "lexical", "rerank", "best unit", "→VLM"):
		t.add_column(col)
	for i, r in enumerate(retrieved, 1):
		p = pages[r["page_idx"]]
		fmt = lambda v: "-" if v is None else f"{v:.4f}"
		t.add_row(str(i), p["filename"], str(p["page"]), fmt(r["visual_score"]), fmt(r["lexical_score"]),
							fmt(r["rerank_score"]), r["best_unit"]["kind"] + (f" {r['best_unit']['bbox']}" if r["best_unit"]["bbox"] else ""),
							"✓" if (n_sent is not None and i <= n_sent) else "")
	console.print(t)
	if verbose:
		for i, r in enumerate(retrieved, 1):
			p = pages[r["page_idx"]]
			console.print(rich.panel.Panel(
				(p["text"][:600] or "[no text layer]"),
				title=f"[{i}] {p['filename']} p.{p['page']} — {p['image_path']}",
			))

# ---------------------------------------------------------------------------
# 7) VLM
# ---------------------------------------------------------------------------
def get_vlm(model_id: str, load_in_4bit: bool = False):
	attention, dtype = get_model_kwargs()
	common = {"trust_remote_code": True, "token": os.getenv("HUGGINGFACE_TOKEN"), "cache_dir": get_user_cache_dir()}
	processor = tfs.AutoProcessor.from_pretrained(model_id, **common)
	tok = getattr(processor, "tokenizer", None)
	if tok is not None:
		tok.padding_side = "left"
		if tok.pad_token is None:
			tok.pad_token = tok.eos_token

	kw = {**common, "device_map": "auto", "attn_implementation": attention, "dtype": dtype}
	if load_in_4bit:
		kw["quantization_config"] = tfs.BitsAndBytesConfig(
			load_in_4bit=True, bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.float16,
			bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
		)
	try:
		vlm = tfs.AutoModelForImageTextToText.from_pretrained(model_id, **kw)
	except Exception as e:
		if attention == "flash_attention_2":
			print(f"[WARN] flash_attention_2 failed for VLM ({str(e)[:120]}...). Using SDPA.")
			kw["attn_implementation"] = "sdpa"
			vlm = tfs.AutoModelForImageTextToText.from_pretrained(model_id, **kw)
		else:
			raise
	vlm.eval()
	return processor, vlm

def page_header(p: Dict[str, Any], rank: int) -> str:
	entity = f" | Entity: {p['entity']}" if p.get("entity") else ""
	return (f"[Page {rank} | Source: {p['filename']} | Company: {p.get('company') or 'unknown'}{entity} | "
					f"Report year: {p.get('report_year') or '?'} | Pg. {p['page']}]")

def build_vlm_messages(query, history, page_items: List[Tuple[str, Image.Image]]):
	msgs = [{"role": "system", "content": [{"type": "text", "text": TEMPLATE__PROMPT}]}]
	for u, a in history[-MAX_HISTORY_TURNS:]:  # text-only history; images are not re-sent
		msgs.append({"role": "user", "content": [{"type": "text", "text": u}]})
		msgs.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
	content = []
	for header, img in page_items:
		content.append({"type": "text", "text": header})
		content.append({"type": "image", "image": img})
	content.append({"type": "text", "text": f"Question: {query}"})
	msgs.append({"role": "user", "content": content})
	return msgs

def _apply_chat_template(processor, msgs):
	kw = dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
	try:
		return processor.apply_chat_template(msgs, enable_thinking=False, **kw)
	except TypeError:
		return processor.apply_chat_template(msgs, **kw)

def prepare_inputs_within_budget(processor, query, history, retrieved, pages, args):
	"""Drop whole pages (lowest-ranked first) until prompt + max_new_tokens fits max_seq_len."""
	n = min(len(retrieved), args.max_pages_to_vlm)
	items = [(page_header(pages[r["page_idx"]], i + 1), load_page_image(pages[r["page_idx"]], args.vlm_max_pixels))
					 for i, r in enumerate(retrieved[:n])]
	while True:
		inputs = _apply_chat_template(processor, build_vlm_messages(query, history, items[:n]))
		L = int(inputs["input_ids"].shape[1])
		if L + args.max_new_tokens <= args.max_seq_len or n == 0:
			if n == 0 and retrieved:
				console.print("[yellow][WARN] No page fits the token budget; lower --vlm_max_pixels or raise --max_seq_len.[/yellow]")
			return inputs, n, L
		n -= 1

def generate_answer(processor, vlm, query, history, retrieved, pages, args) -> Tuple[str, int, int]:
	inputs, n_sent, n_tokens = prepare_inputs_within_budget(processor, query, history, retrieved, pages, args)
	inputs = inputs.to(vlm.device)
	tok = getattr(processor, "tokenizer", processor)
	with torch.inference_mode():
		out = vlm.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
											 pad_token_id=tok.pad_token_id)
	answer = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
	return clean_response(answer), n_sent, n_tokens

# ---------------------------------------------------------------------------
# 8) Evaluation
#   JSONL lines: {"question": "...", "relevant": [{"source": "Ford_Annual_Report_2023.pdf", "page": 54}],
#                 "answer": "optional expected string", "category": "table|chart|text|mixed"}
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
	return re.sub(r"\s+", " ", (s or "").lower().replace(",", "")).strip()

def evaluate(args, embedder, embeddings, units, pages, bm25, reranker, processor=None, vlm=None):
	with open(args.eval_file, encoding="utf-8") as f:
		items = [json.loads(l) for l in f if l.strip()]
	ks = sorted({1, 3, 5, args.top_k})
	kmax = max(ks)
	rows, agg = [], collections.defaultdict(lambda: collections.defaultdict(list))

	for it in track(items, description="Evaluating", console=console):
		gold = {(os.path.basename(g["source"]), int(g["page"])) for g in it.get("relevant", [])}
		ret = retrieve_pages(it["question"], embedder, embeddings, units, pages, args, bm25, reranker, k=kmax)
		got = [(pages[r["page_idx"]]["filename"], pages[r["page_idx"]]["page"]) for r in ret]
		first = next((i + 1 for i, g in enumerate(got) if g in gold), None)
		row = {"question": it["question"], "category": it.get("category", "all"),
					 "gold": sorted(gold), "retrieved": got, "first_hit_rank": first,
					 "mrr": (1.0 / first) if first else 0.0}
		for k in ks:
			row[f"hit@{k}"] = float(first is not None and first <= k)
		if args.eval_generate and vlm is not None:
			ans, n_sent, _ = generate_answer(processor, vlm, it["question"], [], ret, pages, args)
			row["answer"], row["n_pages_sent"] = ans, n_sent
			if it.get("answer"):
				row["answer_contains_expected"] = float(_norm(it["answer"]) in _norm(ans))
		rows.append(row)
		for cat in ("ALL", row["category"]):
			for key in [f"hit@{k}" for k in ks] + ["mrr", "answer_contains_expected"]:
				if key in row:
					agg[cat][key].append(row[key])

	t = rich.table.Table(title=f"Retrieval evaluation ({len(items)} questions)")
	metric_keys = [f"hit@{k}" for k in ks] + ["mrr", "answer_contains_expected"]
	t.add_column("category")
	t.add_column("n")
	for m in metric_keys:
		t.add_column(m)
	summary = {}
	for cat, d in agg.items():
		n_cat = len(d["mrr"])
		summary[cat] = {m: float(np.mean(d[m])) for m in metric_keys if d.get(m)}
		summary[cat]["n"] = n_cat
		t.add_row(cat, str(n_cat), *[f"{summary[cat][m]:.3f}" if m in summary[cat] else "-" for m in metric_keys])
	console.print(t)

	base = os.path.splitext(args.eval_output)[0]
	save_jsonl(rows, base + ".jsonl")
	atomic_write_json({"config": vars(args), "summary": summary}, base + "_summary.json")
	print(f"[EVAL] per-question -> {base}.jsonl | summary -> {base}_summary.json")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
	ap = argparse.ArgumentParser(description="ReportVision-RAG: visual RAG over financial PDFs")
	ap.add_argument("--dataset_directory", "-d", type=str, required=True)
	# models
	ap.add_argument("--vlm_model", "-vlm", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
	ap.add_argument("--embedding_model", "-emb", type=str, default="Qwen/Qwen3-VL-Embedding-2B")
	ap.add_argument("--reranker_model", type=str, default=None, help="e.g. Qwen/Qwen3-VL-Reranker-2B (optional)")
	ap.add_argument("--truncate_dim", type=int, default=None, help="MRL dim truncation for the embedder")
	ap.add_argument("--query_instruction", type=str, default=DEFAULT_QUERY_INSTRUCTION, help="'none' to use model default")
	# rendering / indexing
	ap.add_argument("--dpi", type=int, default=80, help="DPI for PDF rendering")
	ap.add_argument("--image_format", choices=["png", "jpg"], default="png")
	ap.add_argument("--index_mode", choices=["page", "tiles", "both"], default="page")
	ap.add_argument("--tile_rows", type=int, default=3)
	ap.add_argument("--tile_overlap", type=float, default=0.15)
	ap.add_argument("--emb_max_pixels", type=int, default=1_000_000, help="pixel budget per unit for embedding")
	ap.add_argument("--batch_size", "-bs", type=int, default=4)
	ap.add_argument("--rebuild_index", action="store_true")
	ap.add_argument("--index_only", action="store_true")
	# retrieval
	ap.add_argument("--top_k", "-k", type=int, default=5, help="pages retrieved/shown")
	ap.add_argument("--candidate_k", type=int, default=10, help="candidates before rerank/fusion cut")
	ap.add_argument("--hybrid", action="store_true", help="fuse with BM25 over native PDF text layer (RRF)")
	ap.add_argument("--rrf_k", type=int, default=60)
	ap.add_argument("--rerank_max_pixels", type=int, default=1_000_000)
	# generation
	ap.add_argument("--max_pages_to_vlm", type=int, default=3)
	ap.add_argument("--vlm_max_pixels", type=int, default=1_600_000, help="pixel budget per page sent to VLM")
	ap.add_argument("--max_seq_len", type=int, default=16384)
	ap.add_argument("--max_new_tokens", type=int, default=1024)
	ap.add_argument("--load_in_4bit", action="store_true")
	# evaluation
	ap.add_argument("--eval_file", type=str, default=None)
	ap.add_argument("--eval_generate", action="store_true", help="also run the VLM during evaluation")
	ap.add_argument("--eval_output", type=str, default="reportvision_eval.jsonl")
	ap.add_argument("--verbose", "-v", action="store_true")
	args = ap.parse_args()

	embedder, pages, units, embeddings = get_visual_index(args)
	if args.index_only:
		return

	bm25 = None
	if args.hybrid:
		bm25 = BM25([tokenize(p["text"]) for p in pages])
		print(f"[HYBRID] BM25 over text layer of {len(pages):,} pages")
	reranker = load_reranker(args.reranker_model) if args.reranker_model else None

	processor = vlm = None
	if not args.eval_file or args.eval_generate:
		processor, vlm = get_vlm(args.vlm_model, args.load_in_4bit)

	if args.eval_file:
		evaluate(args, embedder, embeddings, units, pages, bm25, reranker, processor, vlm)
		return

	console.print(rich.panel.Panel.fit(
		f"ReportVision-RAG: Visual Financial Reports Assistant\n"
		f"VLM:       {args.vlm_model}\n"
		f"Embedding: {args.embedding_model}\n"
		f"Reranker:  {args.reranker_model or '-'} | Hybrid BM25: {args.hybrid}\n"
		f"Index:     {len(pages):,} pages / {len(units):,} units ({args.index_mode}) @ {args.dpi} dpi\n"
		f"Top-{args.top_k} pages retrieved, up to {args.max_pages_to_vlm} sent to VLM\n"
		f"Type 'exit' to quit."
	))

	history: List[Tuple[str, str]] = []
	while True:
		query = rich.prompt.Prompt.ask("\n[bold magenta]You[/bold magenta]").strip()
		if query.lower() in ("exit", "quit"):
			break
		if not query:
			continue
		with console.status("[bold cyan]Retrieving pages...[/bold cyan]"):
			retrieved = retrieve_pages(query, embedder, embeddings, units, pages, args, bm25, reranker)
		with console.status("[bold cyan]Reading pages with VLM...[/bold cyan]"):
			answer, n_sent, n_tok = generate_answer(processor, vlm, query, history, retrieved, pages, args)

		history.append((query, answer))
		console.print("\n[bold green]AI:[/bold green]")
		console.print(answer)
		show_retrieval(retrieved, pages, n_sent=n_sent, verbose=args.verbose)
		console.print(f"[bold cyan]{n_sent}/{len(retrieved)} pages sent to VLM "
									f"(prompt {n_tok} tok, budget {args.max_seq_len - args.max_new_tokens})[/bold cyan]")

if __name__ == "__main__":
	main()