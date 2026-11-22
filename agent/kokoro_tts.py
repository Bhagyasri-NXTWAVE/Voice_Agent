import asyncio
import logging
import re
import uuid
import numpy as np
from kokoro_onnx import Kokoro
from livekit.agents import tts, APIConnectOptions

logger = logging.getLogger("kokoro-tts")
_KOKORO_INSTANCE: Kokoro | None = None


def _get_kokoro(model_path: str, voices_path: str) -> Kokoro:
    global _KOKORO_INSTANCE
    if _KOKORO_INSTANCE is None:
        _KOKORO_INSTANCE = Kokoro(model_path, voices_path)
    return _KOKORO_INSTANCE


class KokoroTTS(tts.TTS):
    def __init__(
        self,
        model_path: str = "/app/models/kokoro-v1.0.onnx",
        voices_path: str = "/app/models/voices-v1.0.bin",
        voice: str = "af_heart",
        speed: float = 1.0,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self._kokoro = _get_kokoro(model_path, voices_path)
        self._voice = voice
        self._speed = speed

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = APIConnectOptions()
    ) -> tts.ChunkedStream:
        return _KokoroChunkedStream(
            tts=self, input_text=text, conn_options=conn_options
        )


class _KokoroChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        text = (self._input_text or "").strip()
        text = re.sub(r"\b\d{7,15}\b", lambda m: " ".join(m.group(0)), text)
        for bad, good in (
            ("\u202f", " "),
            ("\u00a0", " "),
            ("’", "'"),
            ("‘", "'"),
            ("—", ", "),
            ("–", ", "),
        ):
            text = text.replace(bad, good)
        text = re.sub(
            r"\b(\d{1,2}):00\s*(a\.?m\.?|p\.?m\.?)\b",
            r"\1 \2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return

        samples, sample_rate = await asyncio.to_thread(
            self._tts._kokoro.create,
            text,
            voice=self._tts._voice,
            speed=self._tts._speed,
            lang="en-us",
        )
        # Add 120ms trailing silence so final consonants (e.g. "services") never clip
        pad = np.zeros(int(sample_rate * 0.12), dtype=np.float32)
        samples = np.concatenate([samples, pad])
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        logger.info(f"Kokoro synthesized {len(pcm)} bytes for: {text!r}")

        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm)
        output_emitter.flush()