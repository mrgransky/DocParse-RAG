#!/usr/bin/env bash
set -uo pipefail

# Configuration
if [ $# -eq 0 ]; then
		BASE_DIR="automotive_finance"
else
		BASE_DIR="$1"
fi

LOG_FILE="$BASE_DIR/download_errors.log"
USER_AGENT="FinancialRAGBot/1.0 (contact: your-email@example.com)"

: > "$LOG_FILE"

mkdir -p "$BASE_DIR/BMW" "$BASE_DIR/Tesla" "$BASE_DIR/Ford"

log_error() {
	echo "[$(date -Is)] $*" | tee -a "$LOG_FILE" >&2
}

# download <output-file> <url> <expected-type: pdf|html>
download() {
	local out="$1"
	local url="$2"
	local expected_type="$3"
	local tmp="${out}.part"
	
	echo "Downloading: $url"
	echo "      -> $out"
	
	# -f fails on HTTP 4xx/5xx
	# -L follows redirects
	# --max-time avoids hangs
	if ! curl -fL \
			--retry 3 \
			--retry-delay 5 \
			--connect-timeout 20 \
			--max-time 600 \
			-A "$USER_AGENT" \
			-o "$tmp" \
			"$url"; then
			log_error "FAILED download: $url"
			rm -f "$tmp"
			return 1
	fi
	
	# Basic integrity validation
	if [[ ! -s "$tmp" ]]; then
			log_error "EMPTY file: $url"
			rm -f "$tmp"
			return 1
	fi
	
	if [[ "$expected_type" == "pdf" ]]; then
			# PDF files begin with %PDF
			if ! head -c 5 "$tmp" | grep -q "%PDF"; then
					log_error "NOT A VALID PDF: $url"
					rm -f "$tmp"
					return 1
			fi
	fi
	
	mv "$tmp" "$out"
	echo "OK: $out"
}

# BMW
download "$BASE_DIR/BMW/BMW_Group_Report_2023.pdf" \
"https://www.bmwgroup.com/en/report/2023/downloads/BMW-Group-Report-2023-en.pdf" pdf

download "$BASE_DIR/BMW/BMW_Group_Report_2024.pdf" \
"https://www.bmwgroup.com/en/report/2024/downloads/BMW-Group-Report-2024-en.pdf" pdf

download "$BASE_DIR/BMW/BMW_Group_Report_2025.pdf" \
"https://www.bmwgroup.com/en/report/2025/downloads/BMW-Group-Report-2025-en.pdf" pdf

# Tesla
download "$BASE_DIR/Tesla/Tesla_10-K_31_12_2023.pdf" \
"https://ir.tesla.com/_flysystem/s3/sec/000162828024002390/tsla-20231231-gen.pdf" pdf

download "$BASE_DIR/Tesla/Tesla_10-K_31_12_2024.pdf" \
"https://ir.tesla.com/_flysystem/s3/sec/000162828025003063/tsla-20241231-gen.pdf" pdf

download "$BASE_DIR/Tesla/Tesla_10-K_31_12_2025.pdf" \
"https://ir.tesla.com/_flysystem/s3/sec/000162828026003952/tsla-20251231-gen.pdf" pdf

# Ford 
download "$BASE_DIR/Ford/Ford_Annual_Report_31_12_2023.pdf" \
"https://d18rn0p25nwr6d.cloudfront.net/CIK-0000037996/18354c35-a340-4f1d-9b1a-e8ff0bae917e.pdf" pdf

download "$BASE_DIR/Ford/Ford_Annual_Report_31_12_2024.pdf" \
"https://d18rn0p25nwr6d.cloudfront.net/CIK-0000037996/36cee874-3415-4088-a03e-8f255a43cd89.pdf" pdf


download "$BASE_DIR/Ford/Ford_Annual_Report_31_12_2025.pdf" \
"https://d18rn0p25nwr6d.cloudfront.net/CIK-0000037996/34de982b-4464-440c-b0a8-7e1b7a0e02b9.pdf" pdf

echo
echo "Download complete."
if [[ -s "$LOG_FILE" ]]; then
		echo "Some downloads failed. See: $LOG_FILE"
		exit 1
else
		echo "All downloads succeeded."
fi