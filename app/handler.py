"""RunPod Serverless inference handler for GPT-SoVITS.

Supported actions (dispatched via ``event['input']['action']``):

* ``tts`` — text-to-speech with optional character weights hot-swap.
* ``stt`` — faster-whisper STT on a folder of wav files
  (``tools.asr.fasterwhisper_asr.execute_asr``).

Network Volume layout (symlinked into the source tree by the Dockerfile
entrypoint so existing relative paths just work)::

    /runpod-volume/gpt-sovits/
    ├── pretrained_models/   -> GPT_SoVITS/pretrained_models
    └── asr/faster-whisper-large-v3/   -> tools/asr/models/...

Base TTS weights are loaded at cold start from the defaults baked into
``GPT_SoVITS/configs/tts_infer.yaml``; character-specific
``gpt_weights``/``sovits_weights`` S3 URIs in the ``tts`` payload trigger a
hot-swap via ``init_t2s_weights`` / ``init_vits_weights`` when they differ
from what is currently in memory.
"""
import base64
import io
import logging
import os
import pathlib
import shutil
import sys
import traceback
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import runpod
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[1])

GPTSOVITS_BASE_DIR = os.environ.get("GPTSOVITS_BASE_DIR", _REPO_ROOT)
GPTSOVITS_VOLUME_PATH = os.environ.get(
    "GPTSOVITS_VOLUME_PATH", "/runpod-volume/gpt-sovits"
)
PRETRAINED_S3_BASE = os.environ.get(
    "GPTSOVITS_PRETRAINED_S3",
    "s3://shiftup-enterprise-ai-service/voice/model_registry",
)
WORK_ROOT = os.environ.get("GPTSOVITS_WORK_ROOT", "/tmp/work")
os.makedirs(WORK_ROOT, exist_ok=True)

os.chdir(GPTSOVITS_BASE_DIR)
sys.path.insert(0, GPTSOVITS_BASE_DIR)
sys.path.insert(0, os.path.join(GPTSOVITS_BASE_DIR, "GPT_SoVITS"))

from app import s3_utils
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

_CONFIG_YAML = os.environ.get(
    "GPTSOVITS_CONFIG_PATH",
    os.path.join(GPTSOVITS_BASE_DIR, "GPT_SoVITS/configs/tts_infer.yaml"),
)

_tts_config = TTS_Config(_CONFIG_YAML)
_tts_pipeline = TTS(_tts_config)
logger.info(
    f"[GPT-SoVITS] Cold start ready: version={_tts_config.version}, "
    f"t2s={_tts_config.t2s_weights_path}, vits={_tts_config.vits_weights_path}"
)

_loaded_gpt_uri: Optional[str] = None
_loaded_sovits_uri: Optional[str] = None


def _ensure_character_weights(gpt_uri: Optional[str], sovits_uri: Optional[str]) -> None:
    global _loaded_gpt_uri, _loaded_sovits_uri
    if gpt_uri and gpt_uri != _loaded_gpt_uri:
        local = s3_utils.download_to_temp(gpt_uri, suffix=".ckpt")
        _tts_pipeline.init_t2s_weights(local)
        _loaded_gpt_uri = gpt_uri
        logger.info(f"[GPT-SoVITS] GPT weights loaded: {gpt_uri}")
    if sovits_uri and sovits_uri != _loaded_sovits_uri:
        local = s3_utils.download_to_temp(sovits_uri, suffix=".pth")
        _tts_pipeline.init_vits_weights(local)
        _loaded_sovits_uri = sovits_uri
        logger.info(f"[GPT-SoVITS] SoVITS weights loaded: {sovits_uri}")


def _encode_wav_base64(audio: np.ndarray, sample_rate: int) -> str:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _action_tts(payload: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_character_weights(payload.get("gpt_weights"), payload.get("sovits_weights"))

    ref_audio = payload.get("ref_audio")
    if not ref_audio:
        return {"error": "ref_audio is required"}
    local_ref = s3_utils.download_to_temp(ref_audio, suffix=".wav")

    local_aux: List[str] = []
    for uri in payload.get("aux_ref_audios") or []:
        local_aux.append(s3_utils.download_to_temp(uri, suffix=".wav"))

    req: Dict[str, Any] = {
        "text": payload.get("text", ""),
        "text_lang": payload.get("text_lang", "ko"),
        "ref_audio_path": local_ref,
        "aux_ref_audio_paths": local_aux or None,
        "prompt_text": payload.get("prompt_text") or payload.get("ref_text") or "",
        "prompt_lang": payload.get("prompt_lang") or payload.get("text_lang", "ko"),
        "top_k": int(payload.get("top_k", 5)),
        "top_p": float(payload.get("top_p", 1.0)),
        "temperature": float(payload.get("temperature", 1.0)),
        "text_split_method": payload.get("text_split_method", "cut0"),
        "batch_size": int(payload.get("batch_size", 1)),
        "batch_threshold": float(payload.get("batch_threshold", 0.75)),
        "split_bucket": bool(payload.get("split_bucket", True)),
        "speed_factor": float(payload.get("speed_factor", 1.0)),
        "fragment_interval": float(payload.get("fragment_interval", 0.3)),
        "seed": int(payload.get("seed", -1)),
        "parallel_infer": bool(payload.get("parallel_infer", True)),
        "repetition_penalty": float(payload.get("repetition_penalty", 1.35)),
        "sample_steps": int(payload.get("sample_steps", 32)),
        "super_sampling": bool(payload.get("super_sampling", False)),
        "media_type": "wav",
        "streaming_mode": False,
    }

    n_samples = max(1, int(payload.get("n_samples", 1)))
    samples: List[Dict[str, Any]] = []
    try:
        for _ in range(n_samples):
            chunks: List[np.ndarray] = []
            sample_rate = 0
            for sr, audio in _tts_pipeline.run(req):
                chunks.append(audio)
                sample_rate = sr
            if not chunks:
                continue
            merged = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            samples.append(
                {
                    "audio_base64": _encode_wav_base64(merged, sample_rate),
                    "sample_rate": sample_rate,
                }
            )
    finally:
        for path in [local_ref, *local_aux]:
            try:
                os.unlink(path)
            except Exception:
                pass

    if not samples:
        return {"error": "tts returned no audio"}

    if n_samples == 1:
        return samples[0]
    return {"samples": samples, "n_samples": len(samples)}


def _action_stt(payload: Dict[str, Any]) -> Dict[str, Any]:
    input_s3 = payload.get("input_s3_prefix")
    if not input_s3:
        return {"error": "input_s3_prefix is required"}

    job = uuid.uuid4().hex[:8]
    local_in = os.path.join(WORK_ROOT, f"stt_in_{job}")
    local_out = os.path.join(WORK_ROOT, f"stt_out_{job}")

    try:
        names = payload.get("names")
        if names:
            os.makedirs(local_in, exist_ok=True)
            input_count = 0
            for name in names:
                dst = os.path.join(local_in, name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                s3_utils.download_file(input_s3.rstrip("/") + "/" + name, dst)
                input_count += 1
        else:
            input_count = s3_utils.download_prefix(input_s3, local_in)
        os.makedirs(local_out, exist_ok=True)

        from tools.asr.fasterwhisper_asr import execute_asr

        model_path = s3_utils.ensure_local(
            os.environ.get(
                "GPTSOVITS_ASR_PATH",
                os.path.join(GPTSOVITS_VOLUME_PATH, "asr/faster-whisper-large-v3"),
            ),
            f"{PRETRAINED_S3_BASE}/base/faster-whisper-large-v3/",
        )

        output_file = execute_asr(
            local_in,
            local_out,
            model_path,
            payload.get("language", "auto"),
            payload.get("precision", "float16"),
        )

        transcriptions: List[str] = []
        entries: List[Dict[str, str]] = []
        detected_language = "unknown"
        if isinstance(output_file, str) and os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("|")
                    if len(parts) < 4:
                        continue
                    detected_language = parts[2].lower()
                    transcriptions.append(parts[3].strip())
                    entries.append(
                        {
                            "file": os.path.relpath(parts[0], local_in),
                            "language": parts[2],
                            "text": parts[3].strip(),
                        }
                    )

        return {
            "input_file_count": input_count,
            "transcription": "\n".join(transcriptions),
            "language_detected": detected_language,
            "entries": entries,
        }
    finally:
        shutil.rmtree(local_in, ignore_errors=True)
        shutil.rmtree(local_out, ignore_errors=True)


_ACTIONS = {
    "tts": _action_tts,
    "stt": _action_stt,
}


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    inp = event.get("input") or {}
    action = inp.get("action")
    if action not in _ACTIONS:
        return {"error": f"unknown action: {action!r}; valid: {sorted(_ACTIONS)}"}
    try:
        return _ACTIONS[action](inp)
    except Exception as exc:
        logger.exception("[GPT-SoVITS] handler failed")
        return {"error": str(exc), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
