def normalize_pdf_text(text: str) -> str:
	if not text:
		return ""

	# Normalize line endings
	text = text.replace("\r\n", "\n").replace("\r", "\n")

	# Remove null characters occasionally emitted by PDF extraction
	text = text.replace("\x00", "")

	# Normalize horizontal whitespace without removing line boundaries
	text = re.sub(r"[ \t\f\v]+", " ", text)

	# Remove whitespace at the beginning and end of each line
	text = "\n".join(line.strip() for line in text.split("\n"))

	# Collapse runs of empty lines while preserving paragraph breaks
	text = re.sub(r"\n{3,}", "\n\n", text)

	text = text.strip()

	return text

def load_pdfs_bad_results(data_dir: str) -> List[Dict[str, Any]]:
	docs = []
	for root, _, files in os.walk(data_dir):
		for f in sorted(files):
			if f.lower().endswith(".pdf"):
				path = os.path.join(root, f)
				meta = get_file_metadata(path=path)
				try:
					reader = pypdf.PdfReader(path)
					for page_num, page in enumerate(reader.pages):
						text = page.extract_text() or ""
						text = normalize_pdf_text(text)
						if text.strip():
							doc = {
								"text": text,
								"source": path,
								"page": page_num,
								"company": meta["company"],
								"report_year": meta["report_year"],
								"document_type": meta["document_type"],
							}
							docs.append(doc)
				except Exception as e:
					console.print(f"[red]Error reading {path}: {e}[/red]")
	print(f"Found {len(docs)} documents in DIR: {data_dir}")

	return docs

def get_file_metadata(path: str) -> dict:
	p = pathlib.Path(path)
	company = None
	for part in p.parts:
			if part.lower() in COMPANY_MAP:
					company = COMPANY_MAP[part.lower()]
					break
	if company is None: # fallback: check filename itself
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

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HierarchicalChunker

# Annual reports are digitally-generated (vector) PDFs with native embedded text,
# so OCR is unnecessary. Disabling it avoids RapidOCR weight downloads
# (which fail on read-only filesystems) and speeds up conversion significantly.
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False              # <-- disable OCR
pipeline_options.do_table_structure = True   # keep table structure for RAG
pipeline_options.table_structure_options.do_cell_matching = True

converter = DocumentConverter(
	format_options={
		InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
	}
)

def load_pdfs(data_dir: str, verbose: bool = True):
	docs = []
	for root, _, files in os.walk(data_dir):
		for f in sorted(files):
			if not f.lower().endswith(".pdf"):
				continue

			path = os.path.join(root, f)
			meta = get_file_metadata(path)

			try:
				result = converter.convert(path)
			except Exception as e:
				print(f"<!> Failed to convert {path}: {e}")
				continue

			# whole doc as markdown - best for RAG
			md = result.document.export_to_markdown()

			if verbose:
				print(f"\n{'='*100}")
				print(path)
				print(f"{'='*100}")
				print(md)

			docs.append({
				"text": md,
				"source": path,
				"company": meta["company"],
				"report_year": meta["report_year"],
				"document_type": meta["document_type"],
			})
	return docs

def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
	#Recursive splitter
	# Chunks are bounded by chunk_size + chunk_overlap
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
	print(f"[LOADING] {data_dir} (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
	raw_docs = load_pdfs(data_dir)
	
	chunks = []
	for doc in raw_docs:
		splited_text = split_text(
			text=doc["text"], 
			chunk_size=chunk_size, 
			chunk_overlap=chunk_overlap
		)
		
		for ch in splited_text:
			chunk = {
				"text": ch, 
				"source": doc.get("source"),
				"page": doc.get("page"),
				"company": doc.get("company"),
				"report_year": doc.get("report_year"),
				"document_type": doc.get("document_type"),
			}
			chunks.append(chunk)
	
	print(f"[SPLITED] {len(chunks)} chunks.")
	
	return chunks

