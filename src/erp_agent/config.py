from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from .env file at project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)

@dataclass(frozen=True)
class Settings:
    database_path: str = os.getenv("DATABASE_PATH", "data/erp_agent.db")
    vosk_model_dir: str | None = os.getenv("VOSK_MODEL_DIR") or None
    transcription_normalizer: str = os.getenv("TRANSCRIPTION_NORMALIZER", "ollama").lower()
    planner: str = os.getenv("AGENT_PLANNER", "rules").lower()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL") or os.getenv("NGROK_URL") or "http://127.0.0.1:11434"
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "180"))
    transcription_normalizer_timeout: float = float(os.getenv("TRANSCRIPTION_NORMALIZER_TIMEOUT", "60"))
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "4"))
    invoice_extractor_url: str = os.getenv("INVOICE_EXTRACTOR_URL", "http://127.0.0.1:8000").rstrip("/")
    invoice_extractor_timeout: float = float(os.getenv("INVOICE_EXTRACTOR_TIMEOUT", "180"))
    invoice_extractor_token: str | None = os.getenv("INVOICE_EXTRACTOR_TOKEN") or None
    invoice_extractor_tenant: str = os.getenv("INVOICE_EXTRACTOR_TENANT", "wind-erp")
    invoice_extractor_layout: str = os.getenv("INVOICE_EXTRACTOR_LAYOUT", "auto")

    def resolved_vosk_model_dir(self) -> Path | None:
        if not self.vosk_model_dir:
            return None
        model_dir = Path(self.vosk_model_dir).expanduser()
        return model_dir if model_dir.is_absolute() else PROJECT_ROOT / model_dir

settings = Settings()
