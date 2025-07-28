# src/ingest.py

import fitz
import pytesseract
import multiprocessing
from PIL import Image
from io import BytesIO
from pathlib import Path
import numpy as np


def ocr_fallback(page):
    """
    Performs OCR on a page only if no text is extractable by PyMuPDF.
    Returns the OCR'd text.
    """
    try:
        pix = page.get_pixmap(alpha=False)
        img = Image.open(BytesIO(pix.tobytes()))
        gray = img.convert("L") 
        txt = pytesseract.image_to_string(gray, config="--psm 3") # --psm 3 for page segmentation
        return txt.strip()
    except Exception as e:
        print(f"   - OCR fallback failed for page {page.number}: {e}")
        return ""

def process_pdf(pdf_path):
    """
    Processes a single PDF file, extracting logical chunks/sections with titles
    based on PDF structure, headings, and paragraphs.
    """
    doc_sections = []
    try:
        doc = fitz.open(pdf_path)
        doc_name = pdf_path.name
        
        # Collect all page raw texts and OCR fallbacks first
        page_texts = []
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            text = page.get_text("text").strip()
            if not text:
                text = ocr_fallback(page)
            page_texts.append(text)

        # 1. Prioritize PDF's internal Table of Contents (TOC) if available
        toc = doc.get_toc()
        if toc:
            print(f"   - Found TOC for {doc_name}. Extracting sections from TOC...")
            # For TOC entries, we try to grab the content spanning until the next TOC entry
            # or end of document/page. This is a common pattern for high-quality sections.
            toc_sections_map = {} # {page_num: {title, level}}
            for item in toc:
                level, title, page_num = item
                if title and page_num > 0 and page_num <= doc.page_count:
                    toc_sections_map[page_num] = {"title": title.strip(), "level": f"H{min(level, 3)}"}

            # Now iterate through pages and match content to TOC entries
            for pno in range(doc.page_count):
                current_page_num = pno + 1
                if current_page_num in toc_sections_map:
                    # This page starts a TOC section. Collect text until next TOC section or end of doc.
                    section_title = toc_sections_map[current_page_num]["title"]
                    section_level = toc_sections_map[current_page_num]["level"]
                    
                    full_section_text = ""
                    # Find the next TOC entry's page to limit this section's text
                    next_toc_page = float('inf')
                    for page_in_toc in sorted(toc_sections_map.keys()):
                        if page_in_toc > current_page_num:
                            next_toc_page = page_in_toc
                            break
                    
                    # Collect text from current_page_num up to (but not including) next_toc_page
                    for collect_pno in range(current_page_num, min(next_toc_page, doc.page_count + 1)):
                        full_section_text += "\n" + page_texts[collect_pno - 1] # page_texts is 0-indexed

                    if full_section_text.strip():
                        doc_sections.append({
                            "id": f"{doc_name}-{current_page_num}-{section_title[:50].replace(' ', '_').replace('.', '').replace('/', '_').replace(':', '_')}",
                            "doc": doc_name,
                            "page": current_page_num,
                            "title": section_title,
                            "level": section_level,
                            "text": full_section_text.strip(),
                        })
            # If TOC existed, clear page_texts so we don't re-process these pages with heuristics
            # unless there are gaps or the TOC was incomplete.
            # For simplicity, let's keep all pages and allow the heuristic to add more granular sections.

        # 2. Process page blocks for finer granularity or if no TOC was found
        for pno, page in enumerate(doc, start=1):
            page_text_content = page_texts[pno - 1] # Get already extracted/OCR'd text for this page
            if not page_text_content:
                continue

            blocks = page.get_text("dict")["blocks"] # Get structured text blocks

            current_section_text_lines = []
            current_section_title_candidate = f"Page {pno} Content" # Default title
            current_section_level_candidate = "H4" 
            
            section_id_counter_on_page = 0 # Unique counter per page for heuristic sections

            for block_idx, block in enumerate(blocks):
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span["text"]
                    block_text += "\n"
                block_text = block_text.strip()

                if not block_text:
                    continue
                
                is_detected_heading = False
                temp_detected_title = ""
                temp_detected_level = "H4" 

                if block.get("lines") and block["lines"][0].get("spans"):
                    first_span = block["lines"][0]["spans"][0]
                    font_size = first_span["size"]
                    is_bold = first_span["flags"] & 2
                    
                    # Heuristic for strong headings: bold, large font, and not too short
                    if is_bold and font_size > 12 and len(block_text.split()) > 2 and not block_text.startswith(("- ", "• ", "1. ", "* ")): 
                        is_detected_heading = True
                        temp_detected_title = block_text
                        if font_size > 18: temp_detected_level = "H1"
                        elif font_size > 14: temp_detected_level = "H2"
                        else: temp_detected_level = "H3" # Smaller bold text

                # Condition to finalize current section and start a new one:
                # 1. A new heading is detected.
                # 2. Accumulated text exceeds a reasonable paragraph size (e.g., 150 words).
                if is_detected_heading or (len(current_section_text_lines) > 0 and len(" ".join(current_section_text_lines).split()) > 150): 
                    if current_section_text_lines: # Finalize previous section if accumulated text exists
                        section_id_counter_on_page += 1
                        doc_sections.append({
                            "id": f"{doc_name}-{pno}-{current_section_title_candidate[:50].replace(' ', '_').replace('.', '').replace('/', '_').replace(':', '_')}-{section_id_counter_on_page}",
                            "doc": doc_name,
                            "page": pno,
                            "title": current_section_title_candidate,
                            "level": current_section_level_candidate,
                            "text": " ".join(current_section_text_lines).strip(),
                        })
                    
                    # Start new section with this block
                    current_section_text_lines = [block_text]
                    if is_detected_heading:
                        current_section_title_candidate = temp_detected_title
                        current_section_level_candidate = temp_detected_level
                    else: # Forced split due to length, use a more generic title for continuation
                        current_section_title_candidate = f"Page {pno} Section {section_id_counter_on_page + 1}"
                        current_section_level_candidate = "H4"

                else: # Accumulate text for the current section
                    current_section_text_lines.append(block_text)

            # Add any remaining accumulated text as a final section for the page
            if current_section_text_lines:
                section_id_counter_on_page += 1
                doc_sections.append({
                    "id": f"{doc_name}-{pno}-{current_section_title_candidate[:50].replace(' ', '_').replace('.', '').replace('/', '_').replace(':', '_')}-{section_id_counter_on_page}",
                    "doc": doc_name,
                    "page": pno,
                    "title": current_section_title_candidate,
                    "level": current_section_level_candidate,
                    "text": " ".join(current_section_text_lines).strip(),
                })
        
        # Final fallback for documents that yielded no sections at all (e.g., image-only PDFs without TOC)
        if not doc_sections and doc.page_count > 0: 
            print(f"   - No sections found for {doc_name} via standard extraction or TOC. Attempting full OCR.")
            full_ocr_text = ""
            for pno in range(doc.page_count):
                page = doc.load_page(pno)
                ocr_result = ocr_fallback(page)
                if ocr_result:
                    full_ocr_text += f"\n---Page {pno+1} Content---\n" + ocr_result
            
            if full_ocr_text.strip():
                doc_sections.append({
                    "id": f"{doc_name}-FullOCR-{np.random.randint(10000)}",
                    "doc": doc_name,
                    "page": 1, 
                    "title": f"{doc_name} - Entire Document OCR Content",
                    "level": "H3",
                    "text": full_ocr_text.strip()
                })

    except Exception as e:
        print(f"   - Critical error processing file {pdf_path.name}: {e}")
    
    final_sections = []
    # Filter and truncate
    for sec in doc_sections:
        if sec["text"].strip() and len(sec["text"].split()) >= 20: # Min 20 words for a meaningful chunk
            words = sec["text"].split()
            # Summarizer has n_ctx=2048, prompt takes ~100-200. So ~1800 tokens for text.
            # 1800 tokens * 0.75 words/token = ~1350 words. Let's set a safe cap.
            if len(words) > 1300: 
                sec["text"] = " ".join(words[:1300]) + " [truncated for summarization]"
                if not sec["title"].endswith(" (truncated)"): # Avoid double truncation warning
                    sec["title"] += " (truncated)"
            
            final_sections.append(sec)
        else:
            print(f"   - Skipping small or empty section from {sec.get('doc', '')} page {sec.get('page', '')}: {sec.get('title', '')} (Length: {len(sec['text'].split())} words)")

    print(f"   - Processed {doc_name}: Identified {len(final_sections)} relevant sections.")
    return final_sections

def build_sections(input_dir):
    """
    Finds all PDFs in the input directory and processes them in parallel
    with detailed logging.
    """
    print("   - Starting PDF ingestion (Heuristic-based)...")
    files = list(Path(input_dir).glob("*.pdf"))
    if not files:
        print(f"   - Warning: No PDF files found in the directory: {input_dir}")
        return []

    print(f"   - Found {len(files)} PDF(s) to process: {[f.name for f in files]}")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_pdf, files)

    
    all_sections = [s for sublist in results for s in sublist if s.get("text", "").strip()] 
    print(f"   - PDF ingestion complete. Extracted {len(all_sections)} total sections.")
    return all_sections