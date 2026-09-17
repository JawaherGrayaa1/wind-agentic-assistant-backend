# ERP Agentic Assistant MVP

An open-source-first Python foundation for an ERP assistant with text and Tunisian Arabic speech input, tool calling, memory, approval-gated actions, and auditable traces.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
uvicorn erp_agent.api:app --reload
```

The default planner is deterministic, so the first MVP works without a paid API or a local LLM. Set AGENT_PLANNER=ollama and OLLAMA_MODEL=qwen2.5:7b to use a local Ollama model later.

Copy `.env.example` to `.env` and set `VOSK_MODEL_DIR` to the extracted LinTO Tunisian Arabic Vosk model directory. Relative paths are resolved from the repository root. The API loads the model lazily on the first voice request. Voice input is normalized to mono, 16-bit PCM, 16 kHz.

Voice transcripts are cleaned by the local Ollama model before they reach the planner. Set `TRANSCRIPTION_NORMALIZER=none` to disable this step, or configure `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `TRANSCRIPTION_NORMALIZER_TIMEOUT` for the normalizer. If Ollama is unavailable, the raw transcript is used automatically. Raw and normalized transcripts are stored in the session event trace.

Write actions return an approval token and run only after approval. Traces contain safe summaries, not private chain-of-thought.
