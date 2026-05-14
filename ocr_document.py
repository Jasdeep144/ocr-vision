#!/usr/bin/env python3
"""
OCR Document Script — powered by Claude Vision API
Extracts text from images and PDFs with high accuracy.

Requirements:
    pip install anthropic Pillow pdf2image
    Poppler (for large PDFs): already installed via winget

    Set ANTHROPIC_API_KEY environment variable, or use --api-key
"""

import argparse
import base64
import glob
import io
import os
import sys
from pathlib import Path

MAX_PDF_MB = 30          # Stay safely under the 32 MB API limit
MAX_PDF_PAGES_NATIVE = 20   # Max pages per native PDF chunk
MAX_IMAGE_BYTES = 4_500_000  # Stay under the 5 MB image limit

OCR_PROMPT = (
    "Extract ALL text from this document exactly as it appears. "
    "Preserve the original structure, layout, tables, headings, and formatting as closely as possible. "
    "Do not summarize, interpret, paraphrase, or omit any text. "
    "Output only the extracted text with no extra commentary."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_poppler_path() -> str | None:
    """Auto-detect Poppler bin directory from the winget install location."""
    if os.name != "nt":
        return None
    winget_base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    pattern = os.path.join(winget_base, "oschwartz10612.Poppler*", "poppler-*", "Library", "bin")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def get_api_key(provided: str | None) -> str:
    key = provided or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ERROR: No Anthropic API key found.")
        print("  Option 1: set the environment variable  ANTHROPIC_API_KEY=sk-ant-...")
        print("  Option 2: pass --api-key sk-ant-...")
        sys.exit(1)
    return key


def make_client(api_key: str):
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def extract_text(response) -> str:
    return response.content[0].text


# ---------------------------------------------------------------------------
# OCR: image file
# ---------------------------------------------------------------------------

def ocr_image_file(path: Path, client, model: str) -> str:
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/png",
        ".tiff": "image/png",  ".tif": "image/png",
    }
    suffix = path.suffix.lower()
    media_type = mime_map.get(suffix, "image/png")

    # BMP/TIFF: convert to PNG in memory first (Claude doesn't accept them directly)
    if suffix in (".bmp", ".tiff", ".tif"):
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.open(path).save(buf, format="PNG")
        data = base64.standard_b64encode(buf.getvalue()).decode()
    else:
        data = base64.standard_b64encode(path.read_bytes()).decode()

    response = client.messages.create(
        model=model,
        max_tokens=8096,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
            {"type": "text",  "text": OCR_PROMPT},
        ]}],
    )
    return extract_text(response)


# ---------------------------------------------------------------------------
# OCR: PDF — native (chunked by page range)
# ---------------------------------------------------------------------------

def ocr_pdf_native_chunk(pdf_bytes: bytes, client, model: str) -> str:
    """Send a (possibly partial) PDF to Claude via native PDF support."""
    data = base64.standard_b64encode(pdf_bytes).decode()
    response = client.beta.messages.create(
        model=model,
        max_tokens=8096,
        betas=["pdfs-2024-09-25"],
        messages=[{"role": "user", "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}},
            {"type": "text",     "text": OCR_PROMPT},
        ]}],
    )
    return extract_text(response)


def ocr_pdf_native(path: Path, client, model: str, total_pages: int) -> str:
    """Split PDF into chunks of MAX_PDF_PAGES_NATIVE pages and OCR each chunk."""
    try:
        import pypdf
    except ImportError:
        # pypdf not available — fall back to sending the whole file and let caller handle errors
        return ocr_pdf_native_chunk(path.read_bytes(), client, model)

    reader = pypdf.PdfReader(str(path))
    chunks = range(0, total_pages, MAX_PDF_PAGES_NATIVE)
    results = []

    for start in chunks:
        end = min(start + MAX_PDF_PAGES_NATIVE, total_pages)
        print(f"  Native PDF: pages {start + 1}–{end} of {total_pages}...")

        writer = pypdf.PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])
        buf = io.BytesIO()
        writer.write(buf)
        chunk_text = ocr_pdf_native_chunk(buf.getvalue(), client, model)
        results.append(chunk_text)

    return "\n\n".join(results)


# ---------------------------------------------------------------------------
# OCR: PDF — page-by-page images (fallback)
# ---------------------------------------------------------------------------

def compress_page(page) -> tuple[bytes, str]:
    """Compress a PIL image page to stay under the 5 MB API limit."""
    for quality in (85, 70, 55, 40):
        buf = io.BytesIO()
        page.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= MAX_IMAGE_BYTES:
            return buf.getvalue(), "image/jpeg"
    # Last resort: halve the resolution
    w, h = page.size
    small = page.resize((w // 2, h // 2))
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=70, optimize=True)
    return buf.getvalue(), "image/jpeg"


def ocr_pdf_images(path: Path, client, model: str, dpi: int) -> str:
    """Convert each PDF page to an image and OCR individually."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("pdf2image not installed. Run: pip install pdf2image")
        sys.exit(1)

    poppler_path = find_poppler_path()
    print(f"  Converting PDF to images at {dpi} DPI...")
    pages = convert_from_path(path, dpi=dpi, poppler_path=poppler_path)

    results = []
    for i, page in enumerate(pages, 1):
        img_bytes, media_type = compress_page(page)
        size_kb = len(img_bytes) / 1024
        print(f"  Page {i}/{len(pages)} ({size_kb:.0f} KB)...")
        data = base64.standard_b64encode(img_bytes).decode()
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text",  "text": OCR_PROMPT},
            ]}],
        )
        results.append(f"--- Page {i} ---\n{extract_text(response)}")
    return "\n\n".join(results)


# ---------------------------------------------------------------------------
# OCR: PDF — router
# ---------------------------------------------------------------------------

def pdf_page_count(path: Path) -> int | None:
    """Return page count without converting pages. Returns None if unavailable."""
    try:
        from pdf2image import pdfinfo_from_path
        poppler_path = find_poppler_path()
        info = pdfinfo_from_path(str(path), poppler_path=poppler_path)
        return int(info.get("Pages", 0))
    except Exception:
        pass
    try:
        import pypdf
        return len(pypdf.PdfReader(str(path)).pages)
    except Exception:
        return None


def ocr_pdf(path: Path, client, model: str, dpi: int, force_images: bool) -> str:
    size_mb = path.stat().st_size / (1024 * 1024)
    pages = pdf_page_count(path)
    pages_str = str(pages) if pages else "unknown"

    if not force_images and size_mb <= MAX_PDF_MB:
        n_chunks = ((pages or 1) + MAX_PDF_PAGES_NATIVE - 1) // MAX_PDF_PAGES_NATIVE
        print(f"  Using native PDF mode ({size_mb:.1f} MB, {pages_str} pages, {n_chunks} chunk(s))...")
        try:
            return ocr_pdf_native(path, client, model, pages or MAX_PDF_PAGES_NATIVE)
        except Exception as e:
            err = str(e).lower()
            if "authentication_error" in err or "invalid x-api-key" in err or "401" in err:
                raise RuntimeError(
                    "Anthropic API authentication failed — check your ANTHROPIC_API_KEY."
                ) from e
            print(f"  Native PDF failed ({e}), falling back to image-per-page...")

    print(f"  Using image-per-page mode ({size_mb:.1f} MB, {pages_str} pages)...")
    return ocr_pdf_images(path, client, model, dpi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="High-accuracy OCR powered by Claude Vision API."
    )
    parser.add_argument("input", help="Path to image or PDF file")
    parser.add_argument("-o", "--output", help="Output text file (default: print to stdout)")
    parser.add_argument("--api-key",    help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)"
    )
    parser.add_argument(
        "--force-images", action="store_true",
        help="Always convert PDF to images instead of using native PDF mode"
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="DPI for PDF-to-image conversion in fallback mode (default: 200)"
    )
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    client  = make_client(api_key)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    suffix    = input_path.suffix.lower()
    image_ext = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}

    print(f"Processing: {input_path.name}  [model: {args.model}]")

    if suffix == ".pdf":
        text = ocr_pdf(input_path, client, args.model, args.dpi, args.force_images)
    elif suffix in image_ext:
        text = ocr_image_file(input_path, client, args.model)
    else:
        print(f"ERROR: Unsupported file type '{suffix}'")
        print(f"  Supported: {', '.join(sorted(image_ext))} and .pdf")
        sys.exit(1)

    if args.output:
        out = Path(args.output)
        out.write_text(text, encoding="utf-8")
        print(f"Saved → {out}")
    else:
        print("\n--- OCR Result ---")
        print(text)


if __name__ == "__main__":
    main()
