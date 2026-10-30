#!/bin/bash
# Starts the Ollama server and downloads the chat + embedding models on first boot.
# Models are stored on the persistent disk mounted at /root/.ollama, so later
# restarts skip the download.

ollama serve &
SERVER_PID=$!

echo "Waiting for Ollama to start..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

for model in "${OLLAMA_MODEL:-gemma3:1b}" "${OLLAMA_EMBED_MODEL:-nomic-embed-text}"; do
  echo "Pulling ${model}..."
  ollama pull "${model}" || echo "Could not pull ${model}; the app can download it from the sidebar."
done

echo "Models ready."
wait "${SERVER_PID}"
