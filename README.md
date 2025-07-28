# Round 1B: Persona-Driven Document Intelligence

This implementation follows the specifications for a fully dynamic, non-hardcoded, offline, CPU-only persona-driven document intelligence engine.

## Build and Run

```bash
docker build --platform linux/amd64 -t docai1b:latest .
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none docai1b:latest
```

## Project Structure

- `src/main.py` - Orchestrator
- `src/ingest.py` - Dynamic section extraction
- `src/embed.py` - Quantized embeddings
- `src/rank.py` - Relevance scoring & ranking
- `src/summarise.py` - Dynamic sub-section refinement
- `src/utils.py` - Utilities and JSON output
- `models/` - Quantized model storage
- `input/` - Input PDFs
- `output/` - Generated results
