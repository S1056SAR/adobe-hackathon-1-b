# src/ingest.py

import fitz
import pytesseract
import multiprocessing
from PIL import Image
from io import BytesIO
from pathlib import Path

def extract_headings(page):
    """
    Extracts headings from a page based on font size and boldness.
    A more robust implementation could analyze the document's font distribution.
    """
    blocks = page.get_text("dict")["blocks"]
    headings = []
    for b in blocks:
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            # Check the first span of a line for heading properties
            if spans:
                span = spans[0]
                size, flags = span["size"], span["flags"]
                # Flag 2 corresponds to "bold"
                is_bold = flags & 2
                if is_bold:
                    # Simple level detection based on font size
                    if size > 18:
                        level = "H1"
                    elif size > 14:
                        level = "H2"
                    else:
                        level = "H3"
                    headings.append((span["text"].strip(), level))
    return headings

def ocr_fallback(page):
    """
    Performs OCR on a page. This is a fallback for scanned PDFs or pages
    that are images.
    """
    try:
        pix = page.get_pixmap(alpha=False)
        img = Image.open(BytesIO(pix.tobytes()))
        gray = img.convert("L")
        txt = pytesseract.image_to_string(gray, config="--psm 6")
        # If OCR finds text, treat it as a single content block
        return [("OCR Content", "H3")] if txt.strip() else []
    except Exception as e:
        print(f"   - OCR fallback failed for page: {e}")
        return []

def process_pdf(pdf_path):
    """
    Processes a single PDF file, extracting text and identifying sections.
    """
    sections = []
    try:
        doc = fitz.open(pdf_path)
        for pno, page in enumerate(doc, start=1):
            text = page.get_text("text")
            # If a page has very little text, it might be an image; try OCR.
            if len(text.strip()) < 100:
                heads = ocr_fallback(page)
            else:
                heads = extract_headings(page)

            # If no specific headings are found, treat the whole page as a single section
            # to ensure its content is indexed.
            if not heads:
                heads = [(f"Page {pno} Content", "H3")]

            for title, level in heads:
                sections.append({
                    "id": f"{pdf_path.name}-{pno}-{title[:20].replace(' ', '_')}",
                    "doc": pdf_path.name,
                    "page": pno,
                    "title": title,
                    "level": level,
                    "text": text,  # Associate the full page text with each heading on that page
                })
    except Exception as e:
        print(f"   - Critical error processing file {pdf_path.name}: {e}")
    return sections

def build_sections(input_dir):
    """
    Finds all PDFs in the input directory and processes them in parallel
    with detailed logging.
    """
    print("   - Starting PDF ingestion...")
    files = list(Path(input_dir).glob("*.pdf"))
    if not files:
        print(f"   - Warning: No PDF files found in the directory: {input_dir}")
        return []

    print(f"   - Found {len(files)} PDF(s) to process: {[f.name for f in files]}")

    # Use a multiprocessing pool to process PDFs in parallel, leveraging all available CPUs.
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_pdf, files)

    # Flatten the list of lists into a single list of all sections from all documents
    all_sections = [section for sublist in results for section in sublist]
    print(f"   - PDF ingestion complete. Extracted {len(all_sections)} total sections.")
    return all_sections
