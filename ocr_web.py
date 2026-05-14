#!/usr/bin/env python3
"""
OCR Web UI — Flask server wrapping ocr_document.py
Run:  python ocr_web.py
Open: http://localhost:5000
"""

import io
import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).parent))
from ocr_document import make_client, ocr_image_file, ocr_pdf

app = Flask(__name__, template_folder="html")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}

TRANSLATE_PROMPT = (
    "You are a professional translator. "
    "Translate the following text to English, preserving the original structure, "
    "paragraph breaks, headings, tables, and formatting exactly as they appear. "
    "If the text is already entirely in English, return it completely unchanged. "
    "Output only the translated text — no notes, explanations, or commentary."
)

LEGAL_SUMMARY_PROMPT = (
    "You are a senior legal professional with expertise in contract review and document analysis. "
    "Analyze the following document and produce a structured legal summary using Markdown formatting.\n\n"
    "Use these exact ## section headers and cover each one:\n\n"
    "## 1. Document Type & Purpose\n"
    "Identify the type of document and its primary legal purpose.\n\n"
    "## 2. Parties Involved\n"
    "List all parties, their roles (e.g. Licensor, Licensee, Employer, Employee), "
    "and any relevant identifying details.\n\n"
    "## 3. Key Terms & Obligations\n"
    "Summarize the core duties, commitments, and obligations of each party.\n\n"
    "## 4. Important Dates & Deadlines\n"
    "List all significant dates: effective date, expiry, notice periods, milestones.\n\n"
    "## 5. Financial Terms\n"
    "Detail payment amounts, schedules, penalties, interest, or any financial conditions.\n\n"
    "## 6. Rights & Liabilities\n"
    "Outline rights granted and any limitations or caps on liability.\n\n"
    "## 7. Termination & Exit Provisions\n"
    "Describe how and when the agreement can be terminated and the consequences.\n\n"
    "## 8. Governing Law & Jurisdiction\n"
    "State the applicable law, jurisdiction, and dispute resolution mechanism.\n\n"
    "## 9. Red Flags & Notable Clauses\n"
    "Highlight any unusual, onerous, ambiguous, or potentially risky provisions "
    "that a lawyer would flag for the client.\n\n"
    "## 10. Overall Assessment\n"
    "Provide a brief professional assessment: is this document balanced, "
    "standard, or does it favour one party? What is the key advice?\n\n"
    "Rules: Use **bold** for key legal terms. Use bullet points for lists. "
    "Write 'Not applicable' for any section that does not apply. "
    "Use precise legal language. Be thorough but concise. "
    "Output only the formatted summary — no preamble or closing remarks."
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ocr", methods=["POST"])
def ocr_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED:
        return jsonify({"error": f'Unsupported file type "{suffix}"'}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return jsonify({
            "error": "ANTHROPIC_API_KEY is not set. "
                     "Add it to your environment and restart the server."
        }), 400

    model = request.form.get("model", "claude-sonnet-4-6")
    force_images = request.form.get("force_images", "false").lower() == "true"
    dpi = int(request.form.get("dpi", 200))

    client = make_client(api_key)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        if suffix == ".pdf":
            text = ocr_pdf(tmp_path, client, model, dpi, force_images)
        else:
            text = ocr_image_file(tmp_path, client, model)
        return jsonify({"text": text, "filename": file.filename})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        tmp_path.unlink(missing_ok=True)


@app.route("/api/translate", methods=["POST"])
def translate_endpoint():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY is not set"}), 400

    model = (data or {}).get("model", "claude-sonnet-4-6")
    client = make_client(api_key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=8096,
            messages=[{"role": "user", "content": TRANSLATE_PROMPT + "\n\n---\n\n" + text}],
        )
        return jsonify({"text": response.content[0].text})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/legal-summary", methods=["POST"])
def legal_summary_endpoint():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY is not set"}), 400

    model = (data or {}).get("model", "claude-sonnet-4-6")
    client = make_client(api_key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": LEGAL_SUMMARY_PROMPT + "\n\n---\n\n" + text}],
        )
        return jsonify({"summary": response.content[0].text})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("━" * 48)
    print("  OCR Vision UI")
    print(f"  ➜  http://localhost:{port}")
    print("━" * 48)
    app.run(debug=False, host="0.0.0.0", port=port)
