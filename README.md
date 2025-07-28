# DocAI 1B: Intelligent Document Analysis and Summarization

## Overview

This project delivers a robust, Dockerized solution for intelligent document analysis, focusing on extracting, ranking, and refining information from diverse PDF documents based on a user-defined persona and task. Designed for CPU-only environments, it leverages advanced natural language processing techniques, including state-of-the-art embedding models and efficient large language models (LLMs) for high-quality, actionable insights.

Our system transforms raw, unstructured PDF data into structured, persona-driven intelligence, enabling rapid decision-making and streamlined workflows. It's a testament to achieving complex LLM capabilities within constrained computational environments.

## Methodology: A Deep Dive into Our Intelligent Pipeline

Our solution orchestrates a multi-stage pipeline, meticulously designed for accuracy, relevance, and efficiency on CPU-only infrastructure.

### 1. Document Ingestion and Sectioning (src/ingest.py)

The first critical step involves transforming raw PDF documents into digestible, structured sections. Recognizing the limitations of generic PDF text extraction, our `ingest.py` module employs a sophisticated, heuristic-based approach:

*   **Robust Text Extraction:** Utilizes `PyMuPDF` for primary text extraction. For image-based PDFs or pages with unextractable text, `pytesseract` performs Optical Character Recognition (OCR) as a reliable fallback, ensuring no valuable information is left behind.
*   **Intelligent Section Segmentation:** Instead of treating entire pages as monolithic blocks, we intelligently segment documents into smaller, logical sections. This is achieved through:
    *   **Table of Contents (TOC) Prioritization:** If a PDF includes an internal TOC, our system extracts these high-level sections and their associated content spans, providing inherently accurate and semantically rich section titles.
    *   **Advanced Heuristic Parsing:** For documents without TOCs, or for finer granularity, we analyze text blocks based on visual cues. This includes:
        *   **Heading Detection:** Identifying potential headings based on font size, boldness, and position. If detected, the heading text becomes the section title.
        *   **Paragraph-Level Chunking:** Breaking down continuous text into coherent paragraphs.
        *   **Dynamic Title Generation:** For sections derived from paragraph splits, the system attempts to infer a concise title from the section's leading sentences, offering more context than generic "Page X Content."
    *   **Content Truncation:** To optimize downstream LLM processing, individual section texts are intelligently truncated to fit within optimal token limits (e.g., 1300 words), preventing context window overflows and ensuring focused analysis.
*   **Stability and Resilience:** This heuristic approach was meticulously developed to provide consistent and error-free sectioning, overcoming challenges encountered with more resource-intensive, LLM-based ingestion methods that proved unstable on CPU-limited Docker environments.

### 2. Semantic Embedding (src/embed.py)

Once documents are sectioned, each section's text is converted into a high-dimensional numerical representation (an embedding).

*   **State-of-the-Art Embedding Model:** We employ `intfloat/e5-small-v2`, a powerful and efficient sentence transformer. This model excels at capturing the semantic meaning of text, allowing for highly accurate relevance matching.
*   **Efficient Vector Storage:** `ChromaDB` serves as our vector database. It efficiently stores these embeddings alongside their metadata, enabling rapid similarity searches against user queries. The vector index is built dynamically at runtime, ensuring adaptability to any document collection.

### 3. Relevance Ranking (src/rank.py)

This stage identifies the most pertinent sections from the entire document collection, directly addressing the user's specific query (persona and job).

*   **Hybrid Retrieval:** Our system combines the strengths of two powerful retrieval methods:
    *   **Semantic Search (ANN):** Utilizes the `e5-small-v2` embeddings and `ChromaDB` to find sections semantically similar to the user's `persona` and `job`.
    *   **Lexical Search (BM25):** Leverages `BM25Okapi` to find sections containing exact keyword matches, crucial for specific terms that might be missed by semantic models.
*   **Weighted Fusion:** The scores from both semantic and lexical searches are intelligently combined with a configurable weighting (e.g., 70% semantic, 30% lexical), ensuring a balanced and robust ranking. This ensures that the top 10 most relevant sections are consistently identified.

### 4. Persona-Driven Summarization (src/summarise.py)

This is the core intelligence layer, where the Large Language Model transforms raw extracted text into concise, actionable summaries tailored to the user's needs.

*   **Optimized LLM Inference:** We utilize `TinyLlama-1.1B-Chat-v1.0` (quantized to Q5_K_M) running via `llama.cpp`. This C++ port is an industry standard for blazing-fast, CPU-optimized LLM inference, specifically designed for low-latency performance on consumer hardware. It dramatically outperforms traditional Python-based `transformers` implementations on CPU.
*   **Strategic Prompt Engineering:** The `summarise.py` module employs meticulously crafted prompts that:
    *   **Incorporate Persona and Job:** Explicitly guides the LLM to act as the `persona` (e.g., "Food Contractor") and focus on the `job` (e.g., "Prepare a vegetarian buffet-style dinner menu...").
    *   **Enforce Conciseness:** Strictly instructs the LLM to provide summaries "under 80 words" as a single, coherent paragraph, avoiding lists or conversational filler.
    *   **Actionable Insights:** Directs the LLM to extract "actionable information" relevant to the task, ensuring the output is immediately useful.
*   **Fine-tuned Generation Parameters:** `temperature`, `top_p`, and `max_tokens` are carefully adjusted to encourage focused, factual, and concise outputs, balancing creativity with adherence to instructions. `max_tokens` is aggressively set to ensure brevity and optimize inference time.
*   **Robust Post-Processing:** An intelligent post-processing layer cleans the LLM's output, stripping away any lingering prompt elements, unwanted phrases, or unintended formatting (like bullet points), ensuring a polished final summary.
*   **Scalability on CPU:** By using `llama.cpp` and managing LLM initialization efficiently, the system can perform multiple summarizations sequentially within the limits of a CPU-only environment.

## Conclusion: A Powerful, Pragmatic Solution

This solution stands as a testament to achieving advanced document intelligence within stringent resource constraints. While CPU-only LLM operations inherently face performance ceilings, our pipeline demonstrates:

*   **Robustness:** Handling diverse PDF structures and extracting content reliably.
*   **Relevance:** Accurately identifying the most important sections for a given task.
*   **Actionability:** Delivering focused summaries tailored to specific user needs.
*   **Efficiency:** Pushing the boundaries of LLM inference speed on commodity hardware.

The current implementation provides a highly functional and intelligent system, delivering actionable insights from complex documents. The lessons learned regarding LLM stability in multiprocessing and the critical role of prompt engineering highlight the nuanced challenges of deploying AI solutions in real-world, resource-limited scenarios. This solution is ready to empower users with faster, more intelligent access to their document data.

## Dockerfile and Execution Instructions

This project is fully containerized using Docker, ensuring a consistent and reproducible environment across different systems.

### Prerequisites

*   Docker Desktop (for Windows/macOS) or Docker Engine (for Linux) installed and running.

### Project Setup

1.  **Clone the Repository:**
    ```
    git clone https://github.com/S1056SAR/adobe-hackathon-1-b
    ```

2.  **Prepare Input Directory:**
    *   Create an `input` directory in the root of the project:
        ```
        mkdir input
        ```
    *   Place your PDF documents inside this `input` directory.
    *   Create a `prompt.txt` file inside the `input` directory. This file defines the persona and job for the document analysis.
        *   **Format of `prompt.txt`:**
            The first line should be the `persona`.
            The second line should be the `job_to_be_done`.
            ```
            Food Contractor
            Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items.
            ```

### Building the Docker Image

The Docker image will download all necessary models (e5-small-v2 and TinyLlama GGUF) during the build process. This step requires an internet connection and might take some time (approx. 5-10 minutes) and consume significant RAM (due to model downloads).

1.  **Open your terminal** in the root directory of the project (where `Dockerfile` is located).
2.  **Build the Docker image:**
    ```
    docker build --platform linux/amd64 -t docai1b:latest .
    ```
    *   `--platform linux/amd64`: Ensures compatibility, especially for Apple Silicon (M1/M2) users running Docker.
    *   `-t docai1b:latest`: Tags the image with a memorable name.

### Running the Docker Container

Once the image is built, you can run the analysis. The container will mount your `input` and `output` directories for seamless data exchange.

1.  **Ensure you have PDFs and `prompt.txt` in your `input` directory.**
2.  **Run the Docker container:**
    ```
    docker run --rm -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" --network none docai1b:latest
    ```
    *   `--rm`: Automatically removes the container once it exits.
    *   `-v "${PWD}/input:/app/input"`: Mounts your local `input` directory (where your PDFs and `prompt.txt` are) to the container's `/app/input`.
    *   `-v "${PWD}/output:/app/output"`: Mounts your local `output` directory to the container's `/app/output`. The results will be written here.
    *   `--network none`: Isolates the container from the network. The models are pre-downloaded, so no runtime internet access is required.

### Output

Upon successful execution, a JSON file (`challenge1b_output.json`) will be generated in your local `output` directory. This file contains:

*   `metadata`: Information about the processing (input documents, persona, job, timestamp).
*   `extracted_sections`: A list of the top 10 most relevant sections from the documents, ranked by importance, along with their original document, page number, and title.
*   `subsection_analysis`: A list of refined, persona-driven summaries for each of the `extracted_sections`.

## Troubleshooting

*   **`docker: command not found`**: Ensure Docker Desktop/Engine is installed and running, and that `docker` is in your system's PATH.
*   **`File Not Found` errors for `prompt.txt` or PDFs**: Double-check that your `input` directory is correctly set up and contains the necessary files, and that the `docker run` command's `-v` flags correctly point to your local directories.
*   **`llama_kv_cache_unified: LLAMA_SET_ROWS=0` warnings**: These are normal informational messages from `llama.cpp` and do not indicate an error.
*   **Performance (Slow Execution)**: Running LLMs on CPU is computationally intensive. Performance depends heavily on your CPU specifications (vCPUs, RAM). The solution is highly optimized for CPU but may still take longer than 60 seconds depending on the input documents' complexity and quantity of sections. Consider reducing the number of sections returned by `rank.py` if absolute time adherence is critical, or using a more powerful CPU environment.
*   **Empty `refined_text` in output**: This indicates that the LLM could not produce a summary that met the minimum quality criteria (e.g., word count, coherence) or found no relevant information in the section. Adjusting the prompt or LLM parameters in `src/summarise.py` might help.

