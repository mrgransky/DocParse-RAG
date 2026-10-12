import os
import argparse
import torch
import gc
import hashlib
import json
import pickle
import re
import pathlib
import rich
import rich.panel
import rich.prompt
# from pypdf import PdfReader
import pypdf
from typing import Optional, List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import transformers as tfs

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

console = rich.console.Console()
DATA_DIR = "Data"

# TEMPLATE__PROMPT = """\
# You are a senior automotive financial analyst.

# RULES:
# 1. Answer ONLY using the provided context. Using outside knowledge is strictly forbidden.
# 2. Do NOT perform arithmetic.
# 3. CRITICAL VERIFICATION: Before answering, you MUST output a <verification> block. 
# 	 Inside this block, ask yourself: "Is the exact requested figure explicitly stated in the context for the correct company and year?"
# 4. If the answer to your verification is NO, your final output MUST be exactly DATA_MISSING.
# 5. If YES, provide the concise answer and cite the source document.

# Context:
# {context}

# """

TEMPLATE__PROMPT = """\
You are a financial analyst specializing in the analysis of Annual Reports within the automotive sector.

EVIDENCE AND ACCURACY RULES
1. Answer using only the supplied context. Do not introduce outside facts.
2. Answer the user's actual question directly and concisely.
3. Do not reveal internal reasoning, describe your thought process, or output a
	 verification block.
4. Preserve the company, reporting period, currency, units, accounting metric,
	 and sign of every financial figure.
5. Distinguish among revenue, operating income, net income, and adjusted EBIT.
	 Do not treat these metrics as interchangeable.
6. Distinguish consolidated figures from figures attributable to a parent company
	 or its shareholders when the context makes that distinction.
7. Cite supporting evidence using the source filename and page number provided
	 in the context. For example: [Ford_Annual_Report_2023.pdf, p. 54].
8. For comparisons, address each company and reporting period separately.
	 If only part of a comparison is supported, provide the supported part and
	 clearly identify what is unavailable.
9. For explanatory questions, synthesize the relevant statements in the context.
	 Do not claim that one event caused another unless the context supports that
	 relationship.
10. Do not perform arithmetic unless the user explicitly requests a calculation.
		If calculations are requested, show the formula and identify the input figures.
11. If the context provides no evidence for ANY part of the question, respond
		with exactly: DATA_MISSING

CONTEXT:
{context}
"""

MAX_HISTORY_TURNS = 5

COMPANY_MAP = {"bmw": "BMW", "ford": "Ford", "tesla": "Tesla"}

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HierarchicalChunker

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False              # Vector PDF: native text extraction
pipeline_options.do_table_structure = True   # Keep financial tables intact
pipeline_options.table_structure_options.do_cell_matching = True

converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

chunker = HierarchicalChunker()

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
				"source": str(p),
		}

def save_chunks_summary(
	chunks: List[Dict[str, Any]],
	output_path: str,
):
	lengths = [len(c.get("raw_text", "")) for c in chunks]
	summary = {
		"n_chunks": len(chunks),
		"n_documents": len({c["source"] for c in chunks}),
		"n_companies": len(
			{
				c["company"]
				for c in chunks
				if c.get("company")
			}
		),
		"n_empty_chunks": sum(
			not c.get("raw_text", "").strip()
			for c in chunks
		),
		"n_chunks_without_page": sum(
			c.get("page") is None
			for c in chunks
		),
		"n_chunks_without_headings": sum(
			not c.get("headings")
			for c in chunks
		),
		"text_length_chars": {
			"min": min(lengths, default=0),
			"max": max(lengths, default=0),
			"mean": sum(lengths) / len(lengths) if lengths else 0,
		},
	}

	os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2,)

def save_chunks_jsonl(
	chunks: List[Dict[str, Any]],
	output_path: str,
	verbose: bool = False,
) -> None:
	os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
	tmp_path = output_path + ".tmp"



	try:
		with open(tmp_path, "w", encoding="utf-8") as f:
			for chunk in chunks:
				f.write(
						json.dumps(
								chunk,
								ensure_ascii=False,
								default=str,
						)
						+ "\n"
				)
		# Replace the destination only after the complete write
		os.replace(tmp_path, output_path)




	except Exception:
		if os.path.exists(tmp_path):
			os.remove(tmp_path)
		raise
	
	if verbose:
		print(f"[JSONL] {len(chunks)} chunks -> {output_path}")

def save_chunks_pickle(
	chunks: List[Dict[str, Any]],
	output_path: str,
	verbose: bool = False,
) -> None:
	os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
	tmp_path = output_path + ".tmp"
	try:
		with open(tmp_path, "wb") as f:
			pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL,)
		os.replace(tmp_path, output_path)
	except Exception:
		if os.path.exists(tmp_path):
			os.remove(tmp_path)
		raise

	if verbose:
		print(f"[PICKLE] {len(chunks):,} chunks -> {output_path}")

def get_chucked_docs_old(
	data_dir: str,
	chunks_path: str,
	verbose: bool = False,
) -> List[Dict[str, Any]]:
	if verbose:
		print(f"\n[CHUNKING] {data_dir} via HierarchicalChunker...")
	
	chunks = []
	for root, _, files in os.walk(data_dir):
		for filename in sorted(files):
			if not filename.lower().endswith(".pdf"):
				continue
			path = os.path.join(root, filename)
			meta = get_file_metadata(path)
			try:
				result = converter.convert(path)
				doc = result.document
			except Exception as e:
				print(f"<!> Failed to convert {path}: {e}")
				continue
			for chunk in chunker.chunk(doc):
				# Extract page provenance
				page_numbers = sorted(
					{
						prov.page_no
						for item in chunk.meta.doc_items
						for prov in getattr(item, "prov", [])
						if hasattr(prov, "page_no")
					}
				)
				
				# Extract heading hierarchy
				headings = getattr(chunk.meta, "headings", []) or []
				
				# Enrich chunk text with heading context
				header_prefix = (
					f"[{' > '.join(headings)}]\n"
					if headings
					else ""
				)
				enriched_text = header_prefix + chunk.text
				
				# Create deterministic chunk ID
				identity = {
					"source": os.path.normpath(os.path.abspath(path)),
					"pages": page_numbers,
					"headings": headings,
					"raw_text": chunk.text,
				}
				identity_json = json.dumps(
					identity,
					ensure_ascii=False,
					sort_keys=True,
				)
				chunk_id = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()[:16]
				
				# Construct chunk record
				chunk_record = {
					"chunk_id": chunk_id,
					"text": enriched_text,
					"raw_text": chunk.text,
					"source": path,
					"page": page_numbers[0] if page_numbers else None,
					"pages": page_numbers,
					"headings": headings,
					"company": meta.get("company"),
					"report_year": meta.get("report_year"),
					"document_type": meta.get("document_type"),
				}
				chunks.append(chunk_record)

				if verbose:
					print(
						f"Chunk_ID: {chunk_id} | "
						f"Pages: {page_numbers} | "
						f"Headings: {headings} | "
						f"Length: {len(chunk.text)} chars"
					)

	print(f"[DONE] Created {len(chunks):,} structural chunks.")

	# Save both pkl and jsonl files:
	base_path = os.path.splitext(chunks_path)[0]
	jsonl_path = base_path + ".jsonl"
	pickle_path = base_path + ".pkl"
	chuncks_summary_path = base_path + "_summary.json"

	save_chunks_jsonl(
		chunks=chunks,
		output_path=jsonl_path,
		verbose=verbose,
	)

	save_chunks_pickle(
		chunks=chunks,
		output_path=pickle_path,
		verbose=verbose,
	)

	save_chunks_summary(
		chunks=chunks,
		output_path=chuncks_summary_path,
	)

	return chunks

def get_chucked_docs(
	data_dir: str,
	chunks_path: str,
	verbose: bool = False,
) -> List[Dict[str, Any]]:
	# Get paths for cached files
	base_path = os.path.splitext(chunks_path)[0]
	jsonl_path = base_path + ".jsonl"
	pickle_path = base_path + ".pkl"
	chunks_summary_path = base_path + "_summary.json"

	# ========== CACHE CHECK ==========
	if os.path.exists(pickle_path):
		if verbose:
			print(f"[CACHE] Found existing structural chunks at: {pickle_path}")
			print(f"[CACHE] Loading cached chunks instead of parsing PDFs...")
		try:
			with open(pickle_path, "rb") as f:
				chunks = pickle.load(f)
			print(f"[DONE] Loaded {len(chunks):,} cached structural chunks.")
			return chunks
		except Exception as e:
			print(f"<!> Failed to load cached pickle ({e}). Re-processing PDFs...")

	# ========== PROCESS PDFs (Only runs if cache is missing) ==========
	if verbose:
		print(f"\n[CHUNKING] {data_dir} via HierarchicalChunker...")
	
	chunks = []
	for root, _, files in os.walk(data_dir):
		for filename in sorted(files):
			if not filename.lower().endswith(".pdf"):
				continue
			path = os.path.join(root, filename)
			meta = get_file_metadata(path)
			try:
				result = converter.convert(path)
				doc = result.document
			except Exception as e:
				print(f"<!> Failed to convert {path}: {e}")
				continue
			for chunk in chunker.chunk(doc):
				# Extract page provenance
				page_numbers = sorted(
					{
						prov.page_no
						for item in chunk.meta.doc_items
						for prov in getattr(item, "prov", [])
						if hasattr(prov, "page_no")
					}
				)
				
				# Extract heading hierarchy
				headings = getattr(chunk.meta, "headings", []) or []
				
				# Enrich chunk text with heading context
				header_prefix = (
					f"[{' > '.join(headings)}]\n"
					if headings
					else ""
				)
				enriched_text = header_prefix + chunk.text
				
				# Create deterministic chunk ID
				identity = {
					"source": os.path.normpath(os.path.abspath(path)),
					"pages": page_numbers,
					"headings": headings,
					"raw_text": chunk.text,
				}
				identity_json = json.dumps(
					identity,
					ensure_ascii=False,
					sort_keys=True,
				)
				chunk_id = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()[:16]
				
				# Construct chunk record
				chunk_record = {
					"chunk_id": chunk_id,
					"text": enriched_text,
					"raw_text": chunk.text,
					"source": path,
					"page": page_numbers[0] if page_numbers else None,
					"pages": page_numbers,
					"headings": headings,
					"company": meta.get("company"),
					"report_year": meta.get("report_year"),
					"document_type": meta.get("document_type"),
				}
				chunks.append(chunk_record)

				if verbose:
					print(
						f"Chunk_ID: {chunk_id} | "
						f"Pages: {page_numbers} | "
						f"Headings: {headings} | "
						f"Length: {len(chunk.text)} chars"
					)

	print(f"[DONE] Created {len(chunks):,} structural chunks.")

	# Save outputs for future runs
	save_chunks_jsonl(
		chunks=chunks,
		output_path=jsonl_path,
		verbose=verbose,
	)

	save_chunks_pickle(
		chunks=chunks,
		output_path=pickle_path,
		verbose=verbose,
	)

	save_chunks_summary(
		chunks=chunks,
		output_path=chunks_summary_path,
	)

	return chunks

def get_model_kwargs(verbose: bool=False):
	dtype = torch.float32
	attention = "eager"
	if torch.cuda.is_available():
		dtype = (
			torch.bfloat16
			if torch.cuda.is_bf16_supported()
			else torch.float32
		)
		major, minor = torch.cuda.get_device_capability()
		compute_cap = major + minor / 10
		if compute_cap >= 8.0:
			try:
				import flash_attn
				attention = "flash_attention_2"
			except ImportError:
				if verbose:
					print(
						"[WARN] Flash Attention 2 not installed "
						"(pip install flash-attn)"
					)
		if attention == "eager" and compute_cap >= 7.0 and packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0"):
			if verbose:
				print(
					f"[INFO] Using SDPA attention (compute {compute_cap}, "
					f"PyTorch {torch.__version__})"
				)
			attention = "sdpa"

	if verbose:
		print(f"\n[MODEL KWARGS]")
		print(f"  ├─ CUDA        : {torch.version.cuda}")
		print(f"  ├─ PyTorch     : {torch.__version__}")
		print(f"  ├─ compute_cap : {torch.cuda.get_device_capability()}")
		print(f"  ├─ attention   : {attention}")
		print(f"  └─ dtype       : {dtype}")
		print("-"*40)

	return attention, dtype

def _load(
	model_id: str,
	attention: str,
	dtype,
	cache_folder: str,
	token: Optional[str] = None,
) -> SentenceTransformer:
	common_kwargs = {
		"model_name_or_path": model_id,
		"trust_remote_code": True,
		"device": "cuda:0" if torch.cuda.is_available() else "cpu",
		"cache_folder": cache_folder,
		"token": token,
		"processor_kwargs": {"padding_side": "left"},
	}

	model_kwargs = {"attn_implementation": attention, "dtype": dtype}
	model = SentenceTransformer(
		**common_kwargs,
		model_kwargs=model_kwargs,
	)

	# Probe: trigger the attention kernel with a dummy forward pass
	try:
		model.encode("probe", show_progress_bar=False)
		return model
	except RuntimeError as e:
		err_msg = str(e)
		is_flash_head_dim_error = (
			"head dimension at most 256" in err_msg
			and attention == "flash_attention_2"
		)
		if is_flash_head_dim_error:
			print(
				f"[WARN] Flash Attention incompatible with {model_id} "
				f"(head_dim > 256). Falling back to SDPA."
			)
			# Re-load with SDPA
			model_kwargs["attn_implementation"] = "sdpa"
			del model  # free the broken model
			gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			return SentenceTransformer(
				**common_kwargs,
				model_kwargs=model_kwargs,
			)

		raise  # Re-raise if it's a different error

def clean_response(text: str) -> str:
	text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
	return text.strip()

def compute_cache_key(data_dir: str, embed_model: str, chunk_size: int, chunk_overlap: int) -> str:
	h = hashlib.sha256()
	h.update(embed_model.encode())
	h.update(f"{chunk_size}|{chunk_overlap}".encode())

	for root, _, files in os.walk(data_dir):
		for f in sorted(files):
			if f.lower().endswith(".pdf"):
				p = os.path.join(root, f)
				st = os.stat(p)
				h.update(p.encode())
				h.update(str(st.st_size).encode())
				h.update(str(int(st.st_mtime)).encode())

	return h.hexdigest()[:16]

def build_index(model: SentenceTransformer, chunks: List[Dict[str, Any]], batch_size: int, emb_path: str):
	texts = [c["text"] for c in chunks]
	# embeddings = model.encode(
	# 	texts,
	# 	show_progress_bar=True,
	# 	convert_to_numpy=True,
	# 	normalize_embeddings=True,
	# 	batch_size=32,
	# ).astype(np.float32)
	print(f"[ENCODING] {len(texts)} chunks with batch size {batch_size}...")
	embeddings = _encode_(model=model, texts=texts, batch_size=batch_size)
	np.save(emb_path, embeddings)

	return embeddings

def retrieve(
	model: SentenceTransformer,
	embeddings: np.ndarray,
	query: str,
	chunks: List[Dict[str, Any]],
	k: int,
	verbose: bool,
):
	# q = model.encode(
	# 	[query], 
	# 	convert_to_numpy=True, 
	# 	normalize_embeddings=True
	# )[0].astype(np.float32)
	q = _encode_(model=model, texts=[query], batch_size=1, show_progress_bar=False)

	# scores = embeddings @ q  # cosine sim (both normalized)
	scores = (embeddings @ q.T).squeeze(axis=1) # (3092, 1024) @ (1024, 1) -> (3092,)
	
	# sort only the k we keep
	top = np.argpartition(-scores, k - 1)[:k]
	top = top[np.argsort(-scores[top])]

	results = [
		(chunks[i], float(scores[i])) 
		for i in top
	]

	if verbose:
		print(type(q), q.shape)
		print(type(embeddings), embeddings.shape)
		print(type(scores), scores.shape)

		k = min(k, len(scores))
		print(k)
		print(type(top), top.shape)
		print(type(results), len(results))

		debug_retrieval(retrieved=results)

	return results

def build_context_within_budget(
	retrieved: List[Tuple[Dict[str, Any], float]],
	tokenizer,
	max_context_tokens: int,
) -> Tuple[str, List[Dict[str, Any]]]:
	"""Add ranked chunks until token budget is exhausted. Drops whole chunks, never slices."""
	kept, parts, used = [], [], 0

	for doc, _ in retrieved:
		company = doc.get("company") or "unknown"
		ryear = doc.get("report_year") or "?"
		doc_page = doc.get("page") or "?"
		header = (
			f"[Source: {os.path.basename(doc['source'])} | Company: {company} | "
			f"Report year: {ryear} | Pg. {doc_page}]\n"
		)
		block = header + doc["text"] + "\n\n"
		block_tokens = len(tokenizer.encode(block, add_special_tokens=False))
		if used + block_tokens > max_context_tokens:
			if not parts:
				# First chunk alone exceeds budget — truncate it as a last resort
				avail = max_context_tokens - len(tokenizer.encode(header, add_special_tokens=False)) - 8
				if avail <= 0:
					break
				ids = tokenizer.encode(doc["text"], add_special_tokens=False)[:avail]
				block = header + tokenizer.decode(ids, skip_special_tokens=True) + "\n\n"
				parts.append(block)
				kept.append(doc)
			break
		parts.append(block)
		kept.append(doc)
		used += block_tokens
	
	context = "".join(parts)

	return context, kept

def _encode_(
	model,
	texts: List[str],
	batch_size: int = 32,
	prompt: Optional[str] = None,
	show_progress_bar: bool = True,
	normalize: bool = True,
) -> np.ndarray:
	def _has_internal_normalize(m) -> bool:
		if hasattr(m, "modules"):
			return any(isinstance(mod, Normalize) for mod in m.modules())
		return False

	kw = {
		"batch_size": batch_size,
		"show_progress_bar": show_progress_bar,
		"convert_to_numpy": True,
		"precision": "float32",
	}
	
	if prompt is not None:
		kw["prompt"] = prompt
	
	# Only request external normalization if the model pipeline doesn't already do it
	if normalize and not _has_internal_normalize(model):
		kw["normalize_embeddings"] = True

	embeddings = model.encode(list(texts), **kw)
	encoded_embeddings = np.asarray(embeddings, dtype=np.float32)	
	# print(kw)s
	# print(type(encoded_embeddings), encoded_embeddings.shape)
	return encoded_embeddings

def debug_retrieval(retrieved):
	for rank, (doc, score) in enumerate(retrieved, start=1):
		company = doc.get("company") or "?"
		ryear = doc.get("report_year") or "?"
		doc_type = doc.get("document_type") or "?"
		doc_page = doc.get("page") or "?"
		ranked_chunk_msg = (
			f"\n[bold cyan][Rank {rank}] score={score:.4f} | "
			f"{company} | {ryear} | {doc_type} | "
			f"{doc['source']} | page {doc_page}[/bold cyan]"
		)
		console.print(ranked_chunk_msg)
		console.print(
			rich.panel.Panel(
				doc["text"],
				title="Chunk preview",
				# expand=False
			)
		)

def get_document_embedding(
	model_id: str,
	chunk_size: int,       # Kept for compatibility / embedding metadata
	chunk_overlap: int,    # Kept for compatibility / embedding metadata
	batch_size: int,
	rebuild_index: bool = False, 
	verbose: bool = False
) -> Tuple[Any, List[Dict[str, Any]], np.ndarray]:
	
	attention, dtype = get_model_kwargs()
	user_cache_dir = cache_directory.get(os.getenv("USER"), os.getcwd())

	# ========== 1. GLOBAL / SHARED CHUNK CACHE ==========
	# Create a deterministic key for the DATA directory (independent of embedding model)
	data_hash = hashlib.sha256(os.path.abspath(DATA_DIR).encode("utf-8")).hexdigest()[:10]
	shared_chunks_dir = os.path.join(user_cache_dir, "structural_chunks")
	os.makedirs(shared_chunks_dir, exist_ok=True)
	
	# The chunks file is saved globally based ONLY on the source data directory
	shared_chunks_path = os.path.join(shared_chunks_dir, f"chunks_data_{data_hash}.pkl")

	# ========== 2. MODEL-SPECIFIC EMBEDDING INDEX ==========
	cache_key = compute_cache_key(
		data_dir=DATA_DIR,
		embed_model=model_id,
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)
	
	index_dir = os.path.join(
		user_cache_dir, 
		f"index_{model_id.replace('/', '_')}_{cache_key}"
	)
	os.makedirs(index_dir, exist_ok=True)

	emb_path = os.path.join(index_dir, "embeddings.npy")
	meta_path = os.path.join(index_dir, "meta.json")

	# Check if we need to regenerate embeddings
	need_embedding_rebuild = rebuild_index or not os.path.exists(emb_path)

	if not need_embedding_rebuild and os.path.exists(meta_path):
		with open(meta_path) as f:
			meta = json.load(f)
		if meta.get("cache_key") != cache_key:
			if verbose:
				print("[WARN] Cache key mismatch, rebuilding embeddings.")
			need_embedding_rebuild = True

	# ========== 3. LOAD OR GENERATE STRUCTURAL CHUNKS ==========
	# This will load instantly (0.1 seconds) if any other model has already chunked this data_dir
	chunks = get_chucked_docs(
		data_dir=DATA_DIR,
		chunks_path=shared_chunks_path, # Passed to the shared cache location
		verbose=verbose,
	)

	# ========== 4. LOAD OR GENERATE EMBEDDINGS ==========
	if need_embedding_rebuild:
		if verbose:
			print(f"\n[EMBEDDING] Generating embeddings using model: {model_id}...")
		
		# Load the model only when actual embedding generation is required
		emb_model = _load(
			model_id=model_id,
			attention=attention,
			dtype=dtype,
			cache_folder=user_cache_dir,
			token=os.getenv("HUGGINGFACE_TOKEN"),
		)
		
		embeddings = build_index(
			model=emb_model, 
			chunks=chunks, 
			batch_size=batch_size,
			emb_path=emb_path
		)

		# Save embedding metadata
		with open(meta_path, "w") as f:
			cache_data = {
				"cache_key": cache_key,
				"embedding_model": model_id,
				"chunk_size": chunk_size,
				"chunk_overlap": chunk_overlap,
				"n_chunks": len(chunks),
			}
			json.dump(cache_data, f, indent=2)
	else:
		if verbose:
			print(f"[CACHED] Embedding index found for: {model_id}")
		
		# Load model on-demand for retrieval, even if index is cached
		emb_model = _load(
			model_id=model_id,
			attention=attention,
			dtype=dtype,
			cache_folder=user_cache_dir,
			token=os.getenv("HUGGINGFACE_TOKEN"),
		)
		embeddings = np.load(emb_path)

	return emb_model, chunks, embeddings

def get_llm(model_id, load_in_4bit: bool=False):
	attention, dtype = get_model_kwargs()

	llm_model_kw = {
		"trust_remote_code": True,
		"device_map": "auto",
		"token": os.getenv("HUGGINGFACE_TOKEN"),
		"cache_dir": cache_directory.get(os.getenv("USER"), os.getcwd()),
		"attn_implementation": attention,
		"dtype": dtype,
	}

	console.print(f"[bold cyan][LOADING] {model_id}[/bold cyan]")
	tokenizer = tfs.AutoTokenizer.from_pretrained(model_id, **llm_model_kw,)

	# safety net: never want drop the tail (question + Answer: cue).
	# budget context explicitly
	if hasattr(tokenizer, "truncation_side"):
		tokenizer.truncation_side = "left"
	
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id
	
	if hasattr(tokenizer, "padding_side") and tokenizer.padding_side is not None:
		tokenizer.padding_side = "left"

	if load_in_4bit:
		llm_model_kw["quantization_config"] = tfs.BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_compute_dtype=torch.float16,
			bnb_4bit_use_double_quant=True,
			bnb_4bit_quant_type="nf4",
		)
		
	llm = tfs.AutoModelForCausalLM.from_pretrained(model_id, **llm_model_kw,)
	llm.eval()

	return tokenizer, llm

def main():
	parser = argparse.ArgumentParser(description="Financial RAG System (finRAG)")
	parser.add_argument("--llm_model", "-llm", type=str, default="Qwen/Qwen3.5-0.8B")
	parser.add_argument("--embedding_model", "-emb", type=str, default="google/embeddinggemma-2",help="context embedder")
	parser.add_argument("--batch_size", '-bs', type=int, default=8, help="batch size for embeddings")
	parser.add_argument("--chunk_size", type=int, default=800, help="Must be < embedder max_seq_length * ~4 chars/token")
	parser.add_argument("--chunk_overlap", type=int, default=120)
	parser.add_argument("--top_k", type=int, default=8)
	parser.add_argument("--max_seq_len", type=int, default=8192, help="LLM context window (tokens)")
	parser.add_argument("--max_new_tokens", type=int, default=512)
	parser.add_argument("--load_in_4bit", action="store_true")
	parser.add_argument("--rebuild_index", action="store_true")
	parser.add_argument("--verbose", '-v', action="store_true")
	args = parser.parse_args()

	emb_model, chunks, embeddings = get_document_embedding(
		model_id=args.embedding_model,
		batch_size=args.batch_size,
		chunk_size=args.chunk_size,
		chunk_overlap=args.chunk_overlap,
		rebuild_index=args.rebuild_index,
		verbose=args.verbose,
	)

	tokenizer, llm = get_llm(
		model_id=args.llm_model,
		load_in_4bit=args.load_in_4bit
	)
	
	# Chat loop
	chat_history: List[Tuple[str, str]] = []
	console.print(
		rich.panel.Panel.fit(
			f"finRAG: Financial Reports AI Assistant\n"
			f"LLM:       {args.llm_model}\n"
			f"Embedding: {args.embedding_model}\n"
			f"Chunks:    {len(chunks)}\n"
			f"Top-{args.top_k} Chunks for Retrieval\n"
			f"Type 'exit' to quit."
		)
	)

	while True:
		query = rich.prompt.Prompt.ask("\n[bold magenta]You[/bold magenta]")
		if query.lower() in ("exit", "quit"):
			break
		with console.status("[bold cyan]Retrieving...[/bold cyan]"):
			retrieved = retrieve(
				model=emb_model,
				embeddings=embeddings,
				chunks=chunks,
				query=query, 
				k=args.top_k,
				verbose=args.verbose,
			)
	
		# overhead probe 
		# mirror REAL message structure
		probe_msgs = [{"role": "system", "content": TEMPLATE__PROMPT.format(context="")}]
		for u, a in chat_history[-MAX_HISTORY_TURNS:]:
			probe_msgs.append({"role": "user", "content": u})
			probe_msgs.append({"role": "assistant", "content": a})
		probe_msgs.append({"role": "user", "content": query})

		probe_text = tokenizer.apply_chat_template(
			probe_msgs, 
			tokenize=False, 
			add_generation_prompt=True
		)
		overhead = len(tokenizer.encode(probe_text, add_special_tokens=False))
		max_context_tokens = max(0, args.max_seq_len - overhead - args.max_new_tokens - 64)

		context, kept = build_context_within_budget(retrieved, tokenizer, max_context_tokens)

		# Real prompt: 
		# system (instructions + current context) + history + current query
		system_prompt = TEMPLATE__PROMPT.format(context=context)
		prompt_msgs = [{"role": "system", "content": system_prompt}]
		for u, a in chat_history[-MAX_HISTORY_TURNS:]:
			prompt_msgs.append({"role": "user", "content": u})
			prompt_msgs.append({"role": "assistant", "content": a})
		prompt_msgs.append({"role": "user", "content": query})

		# if args.verbose:
		# 	print(json.dumps(probe_msgs, indent=2, ensure_ascii=False))

		with console.status("[bold cyan]Generating...[/bold cyan]"):
			prompt_text = tokenizer.apply_chat_template(
				prompt_msgs, 
				tokenize=False, 
				add_generation_prompt=True,
				enable_thinking=False,
			)

			inputs = tokenizer(
				prompt_text,
				return_tensors="pt",
				truncation=True,
				max_length=args.max_seq_len,
			).to(llm.device)

			with torch.inference_mode(), torch.amp.autocast(
				device_type=llm.device.type,
				enabled=torch.cuda.is_available(),
				dtype=torch.bfloat16,
			):
				outputs = llm.generate(
					**inputs,
					max_new_tokens=args.max_new_tokens,
					do_sample=False,
					pad_token_id=tokenizer.pad_token_id,
					eos_token_id=tokenizer.eos_token_id,
				)
			answer = tokenizer.decode(
				outputs[0][inputs.input_ids.shape[1]:],
				skip_special_tokens=True,
			)
			answer = clean_response(answer)

		chat_history.append((query, answer))
		console.print("\n[bold green]AI:[/bold green]")
		console.print(answer)

		console.print(
			f"\n[bold cyan]Retrieved {len(kept)}/{len(retrieved)} chunks "
			f"(budget: {max_context_tokens} tok)[/bold cyan]"
		)
				
		# if "DATA_MISSING" in answer:
		# 	console.print(
		# 		"\n[bold yellow]Requested query was not found in the loaded documents. "
		# 		"Try rephrasing the question or loading more documents.[/bold yellow]"
		# 	)

if __name__ == "__main__":
	main()