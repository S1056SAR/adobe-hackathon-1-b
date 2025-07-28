import time, json, pathlib, multiprocessing

def timer(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-t0:.1f}s")
        return result
    return wrapper

def write_json(sections, refined, persona, job, output_dir):
    metadata = {
        "documents": [s["doc"] for s in sections],
        "persona": persona,
        "job": job,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    output = {
        "metadata": metadata,
        "extracted_sections": [{
            "document": s["doc"], "page": s["page"],
            "section_title": s["title"], "importance_rank": s["rank"]
        } for s in sections],
        "sub_section_analysis": [{
            "document": r["doc"], "page": r["page"],
            "refined_text": r["text"]
        } for r in refined]
    }
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(pathlib.Path(output_dir,"challenge1b_output.json"),"w") as f:
        json.dump(output, f, indent=2)
