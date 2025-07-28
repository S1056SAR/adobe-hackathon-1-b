# src/utils.py
import time, json, pathlib, multiprocessing

def timer(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-t0:.1f}s")
        return result
    return wrapper

def write_json(extracted_sections, sub_section_analysis, persona, job, output_dir):
    # --- FIX: Collect input_documents from the original input directory ---
    # This assumes INPUT_DIR is accessible from where write_json is called
    # and it was passed through. For simplicity, we can get it from the first extracted_section
    # or assume it's passed as an argument if it's dynamic.
    # Given your main.py, INPUT_DIR is a pathlib.Path object passed indirectly.
    # Let's derive it from the documents listed in extracted_sections to avoid passing INPUT_DIR around.
    
    # Collect unique document names from all extracted sections
    all_doc_names = sorted(list(set([s["doc"] for s in extracted_sections])))
    
    metadata = {
        "input_documents": all_doc_names, # Correctly lists unique input documents
        "persona": persona,
        "job_to_be_done": job.strip(), # Ensure job is stripped of newlines
        "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()) # Match ISO 8601 with Z
    }
    output = {
        "metadata": metadata,
        "extracted_sections": [{
            "document": s["doc"],
            "section_title": s["title"],
            "importance_rank": s["rank"],
            "page_number": s["page"]
        } for s in extracted_sections],
        "subsection_analysis": [{
            "document": r["doc"],
            "refined_text": r["text"],
            "page_number": r["page"]
        } for r in sub_section_analysis]
    }
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(pathlib.Path(output_dir,"challenge1b_output.json"),"w") as f:
        json.dump(output, f, indent=2)

