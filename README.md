# AI Web Research Assistant

This project provides a small command line agent that can search the web, crawl sites and recall previously seen pages. It runs a local language model via `llama-cpp-python` and keeps a minimal vector store for retrieval augmented generation.

## Installation
1. Install Python 3.11 or newer.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download `openhermes-2.5-mistral-7b.Q3_K_M.gguf` and place it in the repository root so `llama-cpp-python` can load it.

## Usage
Invoke the CLI with `python main.py <command>`.

### verify_gpu
Check if the Metal/MPS backend is available for PyTorch.

```bash
python main.py verify_gpu
```

### research
Start an interactive research session. The agent issues Search/Open/Find/Crawl/Recall actions until enough evidence is gathered.

```bash
python main.py research "what is the tallest building in Europe"
```

### crawl
Pre-crawl a domain so results are immediately available for Recall.

```bash
python main.py crawl https://example.com --pages 40
```

## Project layout
- `agent.py` – main agent loop and action logic
- `crawler.py` – simple breadth-first crawler
- `fetcher.py` – HTTP fetching helpers
- `memory.py` – caches pages and stores the mini vector store
- `rag.py` – tiny TF‑IDF store used for similarity search
- `web_search.py` – DuckDuckGo/Startpage search helpers
- `llm_interface.py` – wraps the local `llama-cpp-python` model
- `parser.py` and `utils.py` – text extraction and helper functions

## License
MIT
