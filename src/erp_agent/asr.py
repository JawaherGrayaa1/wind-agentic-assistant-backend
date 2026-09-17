from __future__ import annotations

import json
import os
import tempfile
import wave
from pathlib import Path

class VoskTunisianASR:
    """Adapter for the LinTO Tunisian Arabic model distributed as Vosk."""

    def __init__(self, model_dir: str | None):
        self.model_dir = model_dir
        self._model = None

    def _load(self):
        if not self.model_dir:
            raise RuntimeError("VOSK_MODEL_DIR is not configured")
        if not Path(self.model_dir).is_dir():
            raise RuntimeError(f"Vosk model directory does not exist: {self.model_dir}")
        if self._model is None:
            from vosk import Model
            self._model = Model(self.model_dir)
        return self._model

    @staticmethod
    def _normalize(input_path: str) -> str:
        try:
            with wave.open(input_path, "rb") as source:
                if (
                    source.getnchannels() == 1
                    and source.getsampwidth() == 2
                    and source.getframerate() == 16000
                    and source.getcomptype() == "NONE"
                ):
                    return input_path
        except (wave.Error, EOFError):
            pass

        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_sample_width(2).set_frame_rate(16000)
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        audio.export(output_path, format="wav")
        return output_path

    def transcribe(self, input_path: str) -> str:
        from vosk import KaldiRecognizer
        converted = self._normalize(input_path)
        should_remove = converted != input_path
        try:
            with wave.open(converted, "rb") as audio:
                recognizer = KaldiRecognizer(self._load(), audio.getframerate())
                while chunk := audio.readframes(4000):
                    recognizer.AcceptWaveform(chunk)
                return json.loads(recognizer.FinalResult()).get("text", "").strip()
        finally:
            if should_remove:
                Path(converted).unlink(missing_ok=True)
