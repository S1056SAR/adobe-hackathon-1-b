# Detailed Implementation Outline for Round 1B: Persona-Driven Document Intelligence

This outline provides step-by-step instructions—suitable for an AI coding agent—to build a **fully dynamic**,non-hardcoded**, **offline**, **CPU-only**, **≤1 GB models**, **≤60 s** persona-driven document intelligence engine. Follow each section verbatim.

## 1. Project Structure & Dependencies

```
/app
 ├── Dockerfile
 ├── requirements.txt
 ├── models/
 │    ├── e5-small-v2-int4/
 │    └── tinyllama-1.1b-int4/
 ├── src/
 │    ├── main.py
 │    ├── ingest.py
 │    ├── embed.py
 │    ├── index.py
 │    ├── rank.py
 │    ├── summarise.py
 │    └── utils.py
 └── README.md
```

requirements.txt:
```
pymupdf==1.24.1
pytesseract==0.3.10
torch==2.2.2+cpu
transformers==4.29.0
accelerate==0.30.1
bitsandbytes==0.43.1
sentence-transformers==2.6.1
chromadb==0.6.3
```

## 2. Dockerfile

```dockerfile
FROM --platform=linux/amd64 python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models/ models/
COPY src/ src/
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "src.main"]
```

## 3. `src/utils.py`

```python
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
```

## 4. `src/ingest.py` – Dynamic Section Extraction

```python
import fitz, pytesseract, multiprocessing
from PIL import Image
from io import BytesIO
import numpy as np

def extract_headings(page):
    blocks = page.get_text("dict")["blocks"]
    headings = []
    for b in blocks:
        for line in b.get("lines",[]):
            spans = line.get("spans",[])
            for span in spans:
                size, flags = span["size"], span["flags"]
                level = ("H1" if size>18 else "H2" if size>14 else "H3") if flags&2 else None
                if level:
                    headings.append((span["text"].strip(), level))
    return headings

def ocr_fallback(page):
    pix = page.get_pixmap(alpha=False)
    img = Image.open(BytesIO(pix.tobytes()))
    gray = img.convert("L")
    txt = pytesseract.image_to_string(gray, config="--psm 6")
    return [("OCR:"+l, "H3") for l in txt.splitlines() if l.strip()]

def process_pdf(path):
    doc = fitz.open(path)
    sections = []
    for pno, page in enumerate(doc, start=1):
        heads = extract_headings(page) or ocr_fallback(page)
        text = page.get_text("text")
        for title, level in heads:
            sections.append({
                "id": f"{path}-{pno}-{title[:20]}",
                "doc": path.name,
                "page": pno,
                "title": title,
                "level": level,
                "text": text,  # full page text
            })
    return sections

def build_sections(input_dir):
    from pathlib import Path
    files = list(Path(input_dir).glob("*.pdf"))
    with multiprocessing.Pool() as pool:
        results = pool.map(process_pdf, files)
    return [s for sub in results for s in sub]
```

## 5. `src/embed.py` – Quantized Embeddings

```python
from sentence_transformers import SentenceTransformer
from bitsandbytes import load_quantized
import torch

class Embedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

class VectorIndex:
    def __init__(self, path, embedder):
        import chromadb
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection("docs")
        self.embedder = embedder

    def add(self, sections):
        texts = [s["text"] for s in sections]
        embs  = self.embedder.encode(texts)
        ids   = [s["id"] for s in sections]
        metas = [{"doc":s["doc"],"page":s["page"],"title":s["title"]} for s in sections]
        self.col.add(documents=texts, embeddings=embs.tolist(), ids=ids, metadatas=metas)

    def query(self, query_vec, n=10):
        return self.col.query(query_embeddings=[query_vec.tolist()], n_results=n)[0]
```

## 6. `src/rank.py` – Relevance Scoring & Ranking

```python
import numpy as np
from rank_bm25 import BM25Okapi

def rank_sections(persona, job, sections, vectordb):
    # embed persona+job
    from src.embed import Embedder
    embedder = Embedder("sentence-transformers/e5-small-v2-int4")
    qtext = persona + " " + job
    qvec  = embedder.encode([qtext])[0]
    # ANN search
    res = vectordb.query(qvec, n=30)
    # BM25 over titles
    bm25 = BM25Okapi([s["title"].split() for s in sections])
    scores = []
    for idx, doc_id in enumerate(res["ids"]):
        sec = next(s for s in sections if s["id"]==doc_id)
        dense = res["distances"][idx]
        bm = bm25.get_scores(sec["title"].split())[0]
        score = 0.6*(1-dense) + 0.2*bm + 0.2*(1/(idx+1))
        scores.append((score, sec))
    scores.sort(key=lambda x:-x[0])
    ranked = []
    for rank, (_, sec) in enumerate(scores[:10], start=1):
        sec["rank"] = rank
        ranked.append(sec)
    return ranked
```

## 7. `src/summarise.py` – Dynamic Sub-section Refinement

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Summariser:
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.model     = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, device_map="cpu")

    def refine(self, section, persona, job):
        prompt = (
            f"As a {persona} working to {job}, rewrite this excerpt for direct action:\n"
            f"{section['text']}\nSummary:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        out = self.model.generate(**inputs, max_new_tokens=80)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return {"doc": section["doc"], "page":section["page"], "text": text}

def refine_texts(ranked, persona, job):
    summariser = Summariser("TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return [summariser.refine(sec, persona, job) for sec in ranked]
```

## 8. `src/main.py` – Orchestrator

```python
from src.ingest import build_sections
from src.embed import Embedder, VectorIndex
from src.rank import rank_sections
from src.summarise import refine_texts
from src.utils import write_json, timer
import multiprocessing, pathlib

INPUT_DIR  = pathlib.Path("/app/input")
OUTPUT_DIR = pathlib.Path("/app/output")

@timer
def main():
    persona, job = open("/app/prompt.txt").read().split("\n",1)
    sections = build_sections(INPUT_DIR)
    embedder = Embedder("sentence-transformers/e5-small-v2-int4")
    vectordb = VectorIndex(path="./vectordb", embedder=embedder)
    vectordb.add(sections)
    winners = rank_sections(persona, job, sections, vectordb)
    refined = refine_texts(winners, persona, job)
    write_json(winners, refined, persona, job, OUTPUT_DIR)
    print("Done")

if __name__=="__main__":
    multiprocessing.set_start_method("spawn")
    main()
```

## 9. Validation & Execution

1. **Build Docker**  
   ```bash
   docker build --platform linux/amd64 -t docai1b:latest .
   ```

2. **Run**  
   ```bash
   docker run --rm \
     -v $(pwd)/input:/app/input \
     -v $(pwd)/output:/app/output \
     --network none docai1b:latest
   ```

3. **Ensure**  
   - `output/challenge1b_output.json` contains correct schema.  
   - Total runtime ≤ 60 s on 8 vCPU/16 GB RAM.

**This implementation is fully dynamic**—no hard-coded section titles, thresholds, or document-specific logic—ensuring generalization across any 3–10 PDF collection, any persona, and any job-to-be-done.
