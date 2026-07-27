"""RunPod Serverless inference handler for GPT-SoVITS.

Supported actions (dispatched via ``event['input']['action']``):

* ``tts`` — text-to-speech with optional character weights hot-swap.
* ``speech_slicing`` — split long audio into clips (reuses the existing
  ``tools/slice_audio.py`` subprocess).
* ``stt`` — faster-whisper STT on a folder of wav files
  (``tools.asr.fasterwhisper_asr.execute_asr``).
* ``uvr5_separate`` — vocal/instrumental separation via UVR5
  (``tools/uvr5/vr.py``).
* ``uvr5_models`` — list UVR5 models currently bundled with the worker's
  Network Volume (no audio processing).

Network Volume layout (symlinked into the source tree by the Dockerfile
entrypoint so existing relative paths just work)::

    /runpod-volume/gpt-sovits/
    ├── pretrained_models/   -> GPT_SoVITS/pretrained_models
    ├── asr/faster-whisper-large-v3/   -> tools/asr/models/...
    └── uvr5_weights/        -> tools/uvr5/uvr5_weights

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
import subprocess
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
sys.path.insert(0, os.path.join(GPTSOVITS_BASE_DIR, "tools/uvr5"))

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


def _action_speech_slicing(payload: Dict[str, Any]) -> Dict[str, Any]:
    input_s3 = payload.get("input_s3_prefix")
    output_s3 = payload.get("output_s3_prefix")
    if not input_s3 or not output_s3:
        return {"error": "input_s3_prefix and output_s3_prefix are required"}

    job = uuid.uuid4().hex[:8]
    local_in = os.path.join(WORK_ROOT, f"slice_in_{job}")
    local_out = os.path.join(WORK_ROOT, f"slice_out_{job}")

    try:
        input_count = s3_utils.download_prefix(input_s3, local_in)
        os.makedirs(local_out, exist_ok=True)

        n_parts = int(payload.get("n_parts", 4))
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [GPTSOVITS_BASE_DIR, os.path.join(GPTSOVITS_BASE_DIR, "GPT_SoVITS")]
        )

        processes = []
        for i_part in range(n_parts):
            cmd = [
                sys.executable,
                "tools/slice_audio.py",
                local_in,
                local_out,
                str(payload.get("threshold", "-34")),
                str(payload.get("min_length", "4000")),
                str(payload.get("min_interval", "300")),
                str(payload.get("hop_size", "10")),
                str(payload.get("max_sil_kept", "500")),
                str(payload.get("_max", 0.9)),
                str(payload.get("alpha", 0.25)),
                str(i_part),
                str(n_parts),
            ]
            processes.append(subprocess.Popen(cmd, env=env, cwd=GPTSOVITS_BASE_DIR))
        for proc in processes:
            proc.wait()

        bad = [p.returncode for p in processes if p.returncode != 0]
        if bad:
            return {"error": f"slicing subprocess returned non-zero: {bad}"}

        uploaded = s3_utils.upload_dir(local_out, output_s3)
        return {
            "input_file_count": input_count,
            "output_s3_prefix": output_s3,
            "sliced_file_count": uploaded,
            "sliced_file_names": sorted(os.listdir(local_out)),
        }
    finally:
        shutil.rmtree(local_in, ignore_errors=True)
        shutil.rmtree(local_out, ignore_errors=True)


def _action_stt(payload: Dict[str, Any]) -> Dict[str, Any]:
    input_s3 = payload.get("input_s3_prefix")
    if not input_s3:
        return {"error": "input_s3_prefix is required"}

    job = uuid.uuid4().hex[:8]
    local_in = os.path.join(WORK_ROOT, f"stt_in_{job}")
    local_out = os.path.join(WORK_ROOT, f"stt_out_{job}")

    try:
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
                        {"file": parts[1], "language": parts[2], "text": parts[3].strip()}
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


def _action_uvr5_separate(payload: Dict[str, Any]) -> Dict[str, Any]:
    input_audio = payload.get("input_audio")
    output_s3 = payload.get("output_s3_prefix")
    if not input_audio or not output_s3:
        return {"error": "input_audio and output_s3_prefix are required"}

    import torch
    from tools.uvr5.vr import AudioPre, AudioPreDeEcho

    job = uuid.uuid4().hex[:8]
    work_dir = os.path.join(WORK_ROOT, f"uvr_{job}")
    vocal_dir = os.path.join(work_dir, "vocals")
    instrumental_dir = os.path.join(work_dir, "instrumentals")
    os.makedirs(vocal_dir, exist_ok=True)
    os.makedirs(instrumental_dir, exist_ok=True)

    local_raw = None
    try:
        local_raw = s3_utils.download_to_temp(input_audio)
        local_in = os.path.join(work_dir, "input.wav")
        needs_reformat = True
        try:
            import ffmpeg as ffmpeg_probe

            info = ffmpeg_probe.probe(local_raw, cmd="ffprobe")
            stream = info["streams"][0]
            if stream.get("channels") == 2 and str(stream.get("sample_rate")) == "44100":
                needs_reformat = False
        except Exception:
            pass
        if needs_reformat:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    local_raw,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ac",
                    "2",
                    "-ar",
                    "44100",
                    local_in,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            shutil.copy(local_raw, local_in)

        model_name = payload.get("model_name", "HP5_only_main_vocal")
        agg = int(payload.get("agg", 10))
        output_format = payload.get("output_format", "wav")

        weights_dir = s3_utils.ensure_local(
            os.environ.get(
                "GPTSOVITS_UVR5_WEIGHTS",
                os.path.join(GPTSOVITS_VOLUME_PATH, "uvr5_weights"),
            ),
            f"{PRETRAINED_S3_BASE}/GPT-SoVITS/tools/uvr5/uvr5_weights/",
        )
        weight_file = os.path.join(weights_dir, f"{model_name}.pth")
        if not os.path.exists(weight_file):
            return {"error": f"UVR5 weight not found: {weight_file}"}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if "DeEcho" in model_name:
            model = AudioPreDeEcho(agg=agg, model_path=weight_file, device=device, is_half=False)
        else:
            model = AudioPre(agg=agg, model_path=weight_file, device=device, is_half=False)

        model._path_audio_(
            local_in,
            instrumental_dir,
            vocal_dir,
            output_format,
            "HP3" in model_name,
        )

        vocal_file = None
        instrumental_file = None
        for name in os.listdir(vocal_dir):
            vocal_file = os.path.join(vocal_dir, name)
            break
        for name in os.listdir(instrumental_dir):
            instrumental_file = os.path.join(instrumental_dir, name)
            break

        uploaded: Dict[str, Any] = {"model_used": model_name}
        base_prefix = output_s3.rstrip("/")
        if vocal_file:
            uri = f"{base_prefix}/vocal.{output_format}"
            s3_utils.upload_file(vocal_file, uri, content_type=f"audio/{output_format}")
            uploaded["vocal_audio_uri"] = uri
        if instrumental_file:
            uri = f"{base_prefix}/instrumental.{output_format}"
            s3_utils.upload_file(instrumental_file, uri, content_type=f"audio/{output_format}")
            uploaded["instrumental_audio_uri"] = uri

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if "vocal_audio_uri" not in uploaded and "instrumental_audio_uri" not in uploaded:
            return {"error": "UVR5 produced no output files"}
        return uploaded
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        if local_raw:
            try:
                os.unlink(local_raw)
            except Exception:
                pass


def _action_uvr5_models(_payload: Dict[str, Any]) -> Dict[str, Any]:
    weights_dir = s3_utils.ensure_local(
        os.environ.get(
            "GPTSOVITS_UVR5_WEIGHTS",
            os.path.join(GPTSOVITS_VOLUME_PATH, "uvr5_weights"),
        ),
        f"{PRETRAINED_S3_BASE}/GPT-SoVITS/tools/uvr5/uvr5_weights/",
    )
    if not os.path.isdir(weights_dir):
        return {"models": []}
    names: List[str] = []
    for entry in os.listdir(weights_dir):
        if entry.endswith((".pth", ".ckpt")):
            names.append(entry.rsplit(".", 1)[0])
    return {"models": sorted(names)}


_ACTIONS = {
    "tts": _action_tts,
    "speech_slicing": _action_speech_slicing,
    "stt": _action_stt,
    "uvr5_separate": _action_uvr5_separate,
    "uvr5_models": _action_uvr5_models,
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
