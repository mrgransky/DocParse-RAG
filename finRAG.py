import os
import argparse
import torch
import gc
import hashlib
import json
import pickle
import re
import rich
import rich.panel
import rich.prompt
from pypdf import PdfReader
from typing import Optional, List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import transformers as tfs

HOME, USER = os.getenv('HOME'), os.getenv('USER')

cache_directory = {
	"farid": "/home/farid/datasets/models",
	"alijanif": "/scratch/project_2004072/models",
	"ubuntu": "/media/volume/models",
}

hf_tk: str = os.getenv("HUGGINGFACE_TOKEN")

console = rich.console.Console()
DATA_DIR = "./Data"
TEMPLATE__PROMPT = """\
You are a senior automotive financial analyst.

RULES:
1. Answer ONLY using the provided context. Never use outside knowledge.
2. Do NOT perform arithmetic.
3. CRITICAL VERIFICATION: Before answering, you MUST output a <verification> block. 
   Inside this block, ask yourself: "Is the exact requested figure explicitly stated in the context for the correct company and year?"
4. If the answer to your verification is NO, your final output MUST be exactly DATA_MISSING.
5. If YES, provide the concise answer and cite the source document.

Context:
{context}

Question: {question}
"""

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

def load_pdfs(data_dir: str) -> List[Dict[str, Any]]:
	docs = []
	for root, _, files in os.walk(data_dir):
		for f in sorted(files):
			if f.lower().endswith(".pdf"):
				path = os.path.join(root, f)
				try:
					reader = PdfReader(path)
					for page_num, page in enumerate(reader.pages):
						text = page.extract_text() or ""
						if text.strip():
							docs.append(
								{
									"text": text,
									"source": path,
									"page": page_num,  # 0-based
								}
							)
				except Exception as e:
					console.print(f"[red]Error reading {path}: {e}[/red]")
	
	return docs

def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
	"""Recursive splitter. Chunks are bounded by chunk_size + chunk_overlap."""
	separators = ["\n\n", "\n", ". ", " ", ""]
	def _split(t: str, seps: List[str]) -> List[str]:
		if len(t) <= chunk_size:
				return [t]
		sep = seps[0] if seps else ""
		if sep == "":
				step = max(1, chunk_size - chunk_overlap)
				return [t[i:i + chunk_size] for i in range(0, len(t), step)]
		parts = t.split(sep)
		chunks, current = [], ""
		for part in parts:
				candidate = (current + sep + part) if current else part
				if len(candidate) <= chunk_size:
						current = candidate
				else:
						if current:
								chunks.append(current)
						if len(part) > chunk_size:
								chunks.extend(_split(part, seps[1:]))
								current = ""
						else:
								current = part
		if current:
				chunks.append(current)
		return chunks

	base = _split(text, separators)
	# Prepend overlap but hard-cap so chunks stay bounded.
	if chunk_overlap <= 0:
			return base
	final = []
	for i, ch in enumerate(base):
		if i == 0:
			final.append(ch[:chunk_size])
			continue
		prev_tail = base[i - 1][-chunk_overlap:]
		combined = prev_tail + ch
		# keep the meaningful tail of the chunk if we overshoot
		final.append(combined[-chunk_size - chunk_overlap:] if len(combined) > chunk_size + chunk_overlap else combined)
	return final

def load_and_process_documents(data_dir: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
	console.print(f"[bold cyan][LOADING] {data_dir} (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...[/bold cyan]")
	raw_docs = load_pdfs(data_dir)
	console.print(f"[green]Loaded {len(raw_docs)} raw document pages.[/green]")
	
	chunks = []
	for doc in raw_docs:

		splited_text = split_text(
			text=doc["text"], 
			chunk_size=chunk_size, 
			chunk_overlap=chunk_overlap
		)
		
		for ch in splited_text:
			chunk = {"text": ch, "source": doc["source"], "page": doc["page"]}
			chunks.append(chunk)
	
	console.print(f"[green][SPLITED] {len(chunks)} chunks.[/green]")
	
	return chunks

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

def build_index(model: SentenceTransformer, chunks: List[Dict[str, Any]]):
	texts = [c["text"] for c in chunks]
	embeddings = model.encode(
		texts,
		show_progress_bar=True,
		convert_to_numpy=True,
		normalize_embeddings=True,
		batch_size=32,
	).astype(np.float32)

	return model, embeddings

def retrieve(query: str, chunks, embeddings, model, k: int = 8):
	q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
	scores = embeddings @ q  # cosine sim (both normalized)
	k = min(k, len(scores))
	
	# argpartition is O(n); then sort only the k we keep
	top = np.argpartition(-scores, k - 1)[:k]
	top = top[np.argsort(-scores[top])]
	return [(chunks[i], float(scores[i])) for i in top]

def build_context_within_budget(
	retrieved: List[Tuple[Dict[str, Any], float]],
	tokenizer,
	max_context_tokens: int,
) -> Tuple[str, List[Dict[str, Any]]]:
	"""Add ranked chunks until token budget is exhausted. Drops whole chunks, never slices."""
	kept, parts, used = [], [], 0
	for doc, _ in retrieved:
		header = f"[Source: {os.path.basename(doc['source'])} page {doc['page'] + 1}]\n"
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
	return "".join(parts), kept

def main():
	parser = argparse.ArgumentParser(description="Financial RAG System (finRAG) — Transformers edition")
	parser.add_argument("--llm_model", "-llm", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
	parser.add_argument("--embedding_model", "-emb", type=str, default="google/embeddinggemma-2",help="context embedder")
	parser.add_argument("--chunk_size", type=int, default=800, help="Must be < embedder max_seq_length * ~4 chars/token")
	parser.add_argument("--chunk_overlap", type=int, default=120)
	parser.add_argument("--top_k", type=int, default=8)
	parser.add_argument("--max_seq_len", type=int, default=8192, help="LLM context window (tokens)")
	parser.add_argument("--max_new_tokens", type=int, default=512)
	parser.add_argument("--load_in_4bit", action="store_true")
	parser.add_argument("--rebuild_index", action="store_true")
	args = parser.parse_args()

	attention, dtype = get_model_kwargs()

	emb_model = _load(
		model_id=args.embedding_model,
		attention=attention,
		dtype=dtype,
		cache_folder=cache_directory[os.getenv("USER")],
		token=os.getenv("HUGGINGFACE_TOKEN"),
	)
	emb_max_tokens = getattr(emb_model, "max_seq_length", None)

	# ~4 chars/token heuristic (conservative)
	if emb_max_tokens is not None:
		max_chars = emb_max_tokens * 4
		if args.chunk_size > max_chars:
			console.print(
				f"[yellow]chunk_size={args.chunk_size} exceeds ~{max_chars} chars "
				f"({emb_max_tokens} tokens) for {args.embedding_model}. "
				f"Tail of every chunk will be invisible to the retriever.[/yellow]"
			)

	cache_key = compute_cache_key(
		data_dir=DATA_DIR,
		embed_model=args.embedding_model,
		chunk_size=args.chunk_size, 
		chunk_overlap=args.chunk_overlap,
	)
	
	index_dir = f"index_{args.embedding_model.replace('/', '_')}_{cache_key}"
	os.makedirs(index_dir, exist_ok=True)
	chunks_path = os.path.join(index_dir, "chunks.pkl")
	emb_path = os.path.join(index_dir, "embeddings.npy")
	meta_path = os.path.join(index_dir, "meta.json")
	need_build = args.rebuild_index or not (os.path.exists(chunks_path) and os.path.exists(emb_path))

	if not need_build:
		with open(meta_path) as f:
			meta = json.load(f)
		if meta.get("cache_key") != cache_key:
			console.print("[yellow]Cache key mismatch, rebuilding index.[/yellow]")
			need_build = True
	
	if need_build:
		chunks = load_and_process_documents(
			data_dir=DATA_DIR,
			chunk_size=args.chunk_size,
			chunk_overlap=args.chunk_overlap,
		)
		_, embeddings = build_index(model=emb_model, chunks=chunks)
		
		with open(chunks_path, "wb") as f:
			pickle.dump(chunks, f)
		
		np.save(emb_path, embeddings)
		
		with open(meta_path, "w") as f:
			cache_data = {
					"cache_key": cache_key,
					"embedding_model": args.embedding_model,
					"chunk_size": args.chunk_size,
					"chunk_overlap": args.chunk_overlap,
					"n_chunks": len(chunks),
				}
			json.dump(cache_data, f, indent=2)
	else:
		console.print("[yellow]Loading cached index...[/yellow]")
		with open(chunks_path, "rb") as f:
			chunks = pickle.load(f)
		embeddings = np.load(emb_path)

	model_kwargs = {
		"trust_remote_code": True,
		"device_map": "auto",
		"token": os.getenv("HUGGINGFACE_TOKEN"),
		"cache_dir": cache_directory[USER],
	}

	console.print(f"[bold cyan]Loading LLM {args.llm_model}...[/bold cyan]")
	tokenizer = tfs.AutoTokenizer.from_pretrained(args.llm_model, **model_kwargs,)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	# we never want to drop the tail (question + Answer: cue).
	# We budget context explicitly, so this is a safety net only.
	tokenizer.truncation_side = "left"
	
	if args.load_in_4bit:
		from transformers import BitsAndBytesConfig
		model_kwargs["quantization_config"] = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_compute_dtype=torch.float16,
			bnb_4bit_use_double_quant=True,
			bnb_4bit_quant_type="nf4",
		)
	else:
		model_kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
		
	llm = tfs.AutoModelForCausalLM.from_pretrained(args.llm_model, **model_kwargs,)
	llm.eval()
	
	# Chat loop
	chat_history: List[Tuple[str, str]] = []
	console.print(
		rich.panel.Panel.fit(
			f"[bold green]finRAG: Financial Reports Assistant[/bold green]\n"
			f"LLM: {args.llm_model} | Embeddings: {args.embedding_model}\n"
			f"Chunks: {len(chunks)} | Top-k: {args.top_k}\n"
			f"Type 'exit' to quit."
		)
	)

	while True:
		query = rich.prompt.Prompt.ask("\n[bold magenta]You[/bold magenta]")
		if query.lower() in ("exit", "quit"):
			break
		with console.status("[bold cyan]Retrieving...[/bold cyan]"):
			retrieved = retrieve(query, chunks, embeddings, emb_model, k=args.top_k)

		# token budget for context
		# Overhead = system prompt + history + question + generation tokens + safety margin
		history_str = "".join(f"User: {u}\nAssistant: {a}\n" for u, a in chat_history[-5:])
		# Build a probe prompt with empty context to count fixed overhead
		probe = TEMPLATE__PROMPT.format(context="", chat_history=history_str, question=query)
		# apply_chat_template adds role tokens; count them too
		probe_msgs = [
				{"role": "system", "content": probe},
				{"role": "user", "content": query},
		]
		probe_text = tokenizer.apply_chat_template(
				probe_msgs, tokenize=False, add_generation_prompt=True
		)
		overhead = len(tokenizer.encode(probe_text, add_special_tokens=False))
		max_context_tokens = max(0, args.max_seq_len - overhead - args.max_new_tokens - 64)
		context, kept = build_context_within_budget(retrieved, tokenizer, max_context_tokens)
		prompt_msgs = [
			{
				"role": "system", 
				"content": TEMPLATE__PROMPT.format(context=context, chat_history=history_str, question=query)
			},
			{"role": "user", "content": query},
		]

		with console.status("[bold cyan]Generating...[/bold cyan]"):
			prompt_text = tokenizer.apply_chat_template(
				prompt_msgs, 
				tokenize=False, 
				add_generation_prompt=True
			)

			inputs = tokenizer(
				prompt_text,
				return_tensors="pt",
				truncation=True,
				max_length=args.max_seq_len,
			).to(llm.device)

			with torch.no_grad(), torch.amp.autocast(
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
		
		for i, doc in enumerate(kept, start=1):
			source = doc["source"]
			page = doc["page"]
			page_label = str(page + 1) if isinstance(page, int) else "unknown"
			console.print(
				rich.panel.Panel(
					doc["text"][:1600],
					title=f"[S{i}] {os.path.basename(source)} | PDF page {page_label}",
					subtitle=source,
				)
			)
		
		if "DATA_MISSING" in answer:
			console.print(
				"\n[bold yellow]⚠  The requested figure was not found in the loaded documents. "
				"Try rephrasing or checking a different report year.[/bold yellow]"
			)

if __name__ == "__main__":
	main()