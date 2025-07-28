from src.ingest import build_sections
from src.embed import Embedder, VectorIndex
from src.rank import rank_sections
from src.summarise import refine_texts
from src.utils import write_json, timer
import multiprocessing
import pathlib
import os

INPUT_DIR = pathlib.Path("/app/input")
OUTPUT_DIR = pathlib.Path("/app/output")

@timer
def main():
    print("--- [Step 1/7] Starting main process ---")
    try:
        prompt_path = INPUT_DIR / "prompt.txt"
        print(f"--- [Step 2/7] Reading persona and job from {prompt_path} ---")
        with open(prompt_path, "r") as f:
            persona, job = f.read().split("\n", 1)
        print("   - Persona and job loaded successfully.")
    except FileNotFoundError:
        print(f"Error: 'prompt.txt' not found in the input directory '{INPUT_DIR}'. Please ensure it exists.")
        return
    except ValueError:
        print("Error: 'prompt.txt' must contain two lines: persona on the first, job on the second.")
        return

    sections = build_sections(INPUT_DIR)

    print("--- [Step 3/7] Initializing embedding model ---")
    embedder = Embedder("./models/e5-small-v2")
    print("   - Embedding model loaded.")

    print("--- [Step 4/7] Building vector index ---")
    vectordb = VectorIndex(path="./vectordb", embedder=embedder)
    vectordb.add(sections)
    print("   - Vector index built successfully.")

    print("--- [Step 5/7] Ranking sections based on relevance ---")
    winners = rank_sections(persona, job, sections, vectordb, "./models/e5-small-v2")
    print(f"   - Ranking complete. Found {len(winners)} relevant sections.")

    print("--- [Step 6/7] Refining text with summarization model (This may take several minutes) ---")
    refined = refine_texts(winners, persona, job)
    print("   - Text refinement complete.")

    print("--- [Step 7/7] Writing final output JSON ---")
    write_json(winners, refined, persona, job, OUTPUT_DIR)
    print("--- All steps completed successfully! ---")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
