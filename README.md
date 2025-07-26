# Round 1B: Persona-Driven Document Intelligence

## Overview

This solution implements an advanced persona-driven document intelligence system using Qwen2.5-0.5B-Instruct for intelligent analysis of document collections based on user personas and their specific job requirements.

## Features

- **Multi-Document Processing**: Handles 3-10 PDFs simultaneously
- **Persona Intelligence**: Deep understanding of professional contexts
- **Job-Specific Analysis**: Tailored content ranking and extraction
- **Advanced Text Refinement**: Persona-appropriate content summarization
- **Constraint Compliant**: <1GB model, <60s processing, CPU-only

## Build and Run

1. Build Command
bash
# Navigate to your project directory
cd "F:\adobe 1b"

# Build the Docker image
docker build --platform linux/amd64 -t round1b-solution .
2. Run Command (Simplified!)
bash
# Main run command - much cleaner now!
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  --network none \
  round1b-solution
3. Windows PowerShell Version
powershell
# PowerShell command
docker run --rm `
  -v "${PWD}/input:/app/input" `
  -v "${PWD}/output:/app/output" `
  --network none `
  round1b-solution
4. Full Path Version (if needed)
bash
# Using full Windows paths
docker run --rm \
  -v "F:\adobe 1b\input:/app/input" \
  -v "F:\adobe 1b\output:/app/output" \
  --network none \
  round1b-solution