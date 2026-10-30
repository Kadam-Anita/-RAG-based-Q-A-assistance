# Document Q&A with Ollama

Chat with your PDFs using a local Ollama model. Pipeline: PDF → chunks → Ollama embeddings → FAISS → retrieve → (optional) LLM relevance grading → (optional) sentence filtering → streamed answer with sources.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit chat UI |
| `rag_core.py` | Indexing, LangGraph pipeline, Ollama helpers (no Streamlit code) |
| `requirements.txt` | Python dependencies (no PyTorch, so installs and deploys are small) |
| `Dockerfile.ollama`, `ollama-start.sh` | Ollama server image for Render |
| `render.yaml` | Render Blueprint (Ollama + app) |
| `.env.example` | Local settings template |

## Run locally

1. Install [Ollama](https://ollama.com) (version 0.5 or newer, needed for structured output) and make sure it is running.
2. Pull the models:
   ```
   ollama pull gemma3:1b
   ollama pull nomic-embed-text
   ```
3. Install and start the app:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

The sidebar has a **Test connection** button and can download missing models for you.

## Deploy on Render

1. Push all these files to the root of a GitHub repo.
2. In Render: **New → Blueprint**, pick the repo, and apply. This creates:
   - `ollama`: a private service with a persistent disk for models.
   - `rag-app`: the public Streamlit app, already wired to `ollama`.
3. Wait for `ollama` to finish its first boot. Its logs print `Models ready.` once both models are downloaded.
4. Open the `rag-app` URL. The sidebar should say **Connected**.

Notes:
- The `ollama` service needs roughly 2 GB of RAM for `gemma3:1b` (Standard plan) and a paid disk. The free tier cannot run it.
- Render has no GPU. Use **Fast** or **Balanced** mode to keep answers quick on CPU.
- The index lives in memory. After a restart or redeploy, upload and build the index again.

## Deploy on Streamlit Community Cloud

Streamlit Cloud cannot run Ollama itself, so the app needs a remote Ollama server (a VPS or GPU box, or your own machine through a tunnel).

1. Push the repo to GitHub and create the app from `app.py`.
2. In **Settings → Secrets** add:
   ```toml
   OLLAMA_BASE_URL = "https://your-ollama-host"
   OLLAMA_MODEL = "gemma3:1b"
   OLLAMA_EMBED_MODEL = "nomic-embed-text"
   OLLAMA_API_KEY = "optional-bearer-token"
   ```
3. Do not expose a bare Ollama port to the internet. Ollama has no login of its own, so put it behind a reverse proxy or tunnel that checks the Bearer token you set as `OLLAMA_API_KEY`.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server (`host:port` is fine too) |
| `OLLAMA_MODEL` | `gemma3:1b` | Chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_API_KEY` | empty | Optional Bearer token |

Answer modes: **Fast** (retrieve + answer), **Balanced** (adds per-chunk grading), **Thorough** (adds sentence filtering, many extra model calls).

## Troubleshooting

- `WinError 10061` / connection refused: Ollama is not running at that URL. Start it, then press **Test connection**.
- "Not installed yet" in the sidebar: press **Download missing models**.
- Always "couldn't find enough information": open **Sources & details** to see which pages were retrieved. If they are the wrong pages, the problem is search, not grading: rebuild the index (the app removes repeated page headers and adds the prefixes `nomic-embed-text` expects), ask with full terms ("generative AI" rather than "gen ai"), and raise "Chunks to retrieve". If the pages are right but still rejected, use **Fast** mode or lower "Keep chunks scoring above".
- No text extracted: the PDF is probably scanned images and needs OCR first.
- Changing the embedding model requires rebuilding the index.
