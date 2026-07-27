"""RunPod Serverless training handler for GPT-SoVITS.

A single ``train`` action runs the full pipeline end-to-end on one worker:

    raw wavs (S3)
        -> slicing (tools/slice_audio.py, parallel subprocess)
        -> STT (tools.asr.fasterwhisper_asr.execute_asr, Whisper large-v3)
        -> dataset formatting (1-get-text.py / 2-get-hubert-wav32k.py /
           2-get-sv.py for Pro / 3-get-semantic.py)
        -> SoVITS fine-tuning (GPT_SoVITS/s2_train.py)
        -> GPT fine-tuning (GPT_SoVITS/s1_train.py)
        -> upload checkpoints to S3
        -> POST callback with S3 URIs

The handler intentionally keeps everything in this one module because:

* RunPod Serverless charges per worker so cold-starting the base model for
  each sub-stage would be wasteful.
* Partial retries are not meaningful here — if format-dataset fails the
  downstream stages can't run, and reruning format-dataset alone is no
  cheaper than the entire pipeline.

Progress is pushed to RunPod every few seconds via
``runpod.serverless.progress_update`` so the client can poll ``/status`` for
a live ``stage`` + ``step`` view.
"""
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

import httpx
import runpod
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default follows the code instead of naming a fixed image path, so it stays
# correct wherever the checkout is installed (/srv/gpt-sovits, /app/GPT-SoVITS,
# a dev clone). app/train_handler.py -> parents[1] is the GPT-SoVITS repo root.
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
MODEL_REGISTRY_S3 = os.environ.get(
    "GPTSOVITS_MODEL_REGISTRY_S3",
    "s3://shiftup-enterprise-ai-service/voice/model_registry/GPT-SoVITS",
)

os.makedirs(WORK_ROOT, exist_ok=True)
os.chdir(GPTSOVITS_BASE_DIR)
sys.path.insert(0, GPTSOVITS_BASE_DIR)
sys.path.insert(0, os.path.join(GPTSOVITS_BASE_DIR, "GPT_SoVITS"))

from app import s3_utils

_PROGRESS_LOCK = threading.Lock()
_progress_state: Dict[str, Any] = {"stage": "init", "step": 0, "total_steps": 0, "extra": {}}


def _set_progress(**fields: Any) -> None:
    with _PROGRESS_LOCK:
        _progress_state.update(fields)


def _snapshot_progress() -> Dict[str, Any]:
    with _PROGRESS_LOCK:
        return dict(_progress_state)


def _run_slicing(raw_dir: str, sliced_dir: str, n_parts: int = 4) -> None:
    _set_progress(stage="slicing", step=0)
    os.makedirs(sliced_dir, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [GPTSOVITS_BASE_DIR, os.path.join(GPTSOVITS_BASE_DIR, "GPT_SoVITS")]
    )
    processes = []
    for i_part in range(n_parts):
        cmd = [
            sys.executable,
            "tools/slice_audio.py",
            raw_dir,
            sliced_dir,
            "-34",
            "4000",
            "300",
            "10",
            "500",
            "0.9",
            "0.25",
            str(i_part),
            str(n_parts),
        ]
        processes.append(subprocess.Popen(cmd, env=env, cwd=GPTSOVITS_BASE_DIR))
    for proc in processes:
        proc.wait()
    bad = [p.returncode for p in processes if p.returncode != 0]
    if bad:
        raise RuntimeError(f"slice_audio.py failed: returncodes={bad}")


def _run_stt(sliced_dir: str, stt_output_dir: str) -> str:
    _set_progress(stage="stt", step=0)
    os.makedirs(stt_output_dir, exist_ok=True)
    from tools.asr.fasterwhisper_asr import execute_asr

    model_path = s3_utils.ensure_local(
        os.environ.get(
            "GPTSOVITS_ASR_PATH",
            os.path.join(GPTSOVITS_VOLUME_PATH, "asr/faster-whisper-large-v3"),
        ),
        f"{PRETRAINED_S3_BASE}/base/faster-whisper-large-v3/",
    )
    output_file = execute_asr(sliced_dir, stt_output_dir, model_path, "auto", "float16")
    if not isinstance(output_file, str) or not os.path.exists(output_file):
        raise RuntimeError(f"STT execute_asr returned invalid path: {output_file!r}")
    return output_file


def _run_format_dataset(
    *,
    exp_name: str,
    exp_dir: str,
    inp_text: str,
    inp_wav_dir: str,
    version: str,
    gpu_numbers: str,
) -> None:
    """Replicate ``api_v2.execute_dataset_formatting`` in-process.

    Runs the three preparation stages sequentially and merges the per-part
    output files the same way the legacy endpoint did.
    """
    _set_progress(stage="format_dataset", step=0)
    os.makedirs(exp_dir, exist_ok=True)

    bert_dir = os.path.join(GPTSOVITS_BASE_DIR, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
    ssl_dir = os.path.join(GPTSOVITS_BASE_DIR, "GPT_SoVITS/pretrained_models/chinese-hubert-base")
    sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
    pretrained_s2g = _pretrained_s2g_for_version(version)

    gpu_parts = gpu_numbers.split("-")
    all_parts = len(gpu_parts)

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = os.pathsep.join(
        [GPTSOVITS_BASE_DIR, os.path.join(GPTSOVITS_BASE_DIR, "GPT_SoVITS")]
    )

    def _run_stage_script(script: str, part_env: Dict[str, str]) -> None:
        env = base_env.copy()
        env.update(part_env)
        subprocess.run(
            [sys.executable, script],
            env=env,
            cwd=GPTSOVITS_BASE_DIR,
            check=True,
        )

    _set_progress(stage="format_dataset", step=0, extra={"sub_stage": "1a"})
    for i_part in range(all_parts):
        _run_stage_script(
            "GPT_SoVITS/prepare_datasets/1-get-text.py",
            {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": exp_dir,
                "bert_pretrained_dir": bert_dir,
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": gpu_parts[i_part],
                "is_half": "True",
            },
        )
    merged_text = os.path.join(exp_dir, "2-name2text.txt")
    with open(merged_text, "w", encoding="utf-8") as out:
        for i_part in range(all_parts):
            part_path = os.path.join(exp_dir, f"2-name2text-{i_part}.txt")
            if os.path.exists(part_path):
                with open(part_path, "r", encoding="utf-8") as handle:
                    out.write(handle.read().strip("\n") + "\n")
                os.remove(part_path)

    _set_progress(stage="format_dataset", step=0, extra={"sub_stage": "1b"})
    for i_part in range(all_parts):
        _run_stage_script(
            "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
            {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": exp_dir,
                "cnhubert_base_dir": ssl_dir,
                "sv_path": sv_path,
                "is_half": "True",
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": gpu_parts[i_part],
            },
        )
    if "Pro" in version:
        for i_part in range(all_parts):
            _run_stage_script(
                "GPT_SoVITS/prepare_datasets/2-get-sv.py",
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_parts[i_part],
                    "exp_dir": exp_dir,
                    "sv_path": sv_path,
                    "is_half": "True",
                },
            )

    _set_progress(stage="format_dataset", step=0, extra={"sub_stage": "1c"})
    s2_config = (
        "GPT_SoVITS/configs/s2.json"
        if version not in {"v2Pro", "v2ProPlus"}
        else f"GPT_SoVITS/configs/s2{version}.json"
    )
    for i_part in range(all_parts):
        _run_stage_script(
            "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
            {
                "inp_text": inp_text,
                "exp_name": exp_name,
                "opt_dir": exp_dir,
                "pretrained_s2G": pretrained_s2g,
                "s2config_path": s2_config,
                "is_half": "True",
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": gpu_parts[i_part],
            },
        )
    merged_semantic = os.path.join(exp_dir, "6-name2semantic.tsv")
    header = ["item_name\tsemantic_audio"]
    body: List[str] = []
    for i_part in range(all_parts):
        part_path = os.path.join(exp_dir, f"6-name2semantic-{i_part}.tsv")
        if os.path.exists(part_path):
            with open(part_path, "r", encoding="utf-8") as handle:
                body += handle.read().strip("\n").split("\n")
            os.remove(part_path)
    with open(merged_semantic, "w", encoding="utf-8") as out:
        out.write("\n".join(header + body))


def _pretrained_s2g_for_version(version: str) -> str:
    """Return the repo-relative pretrained S2G path for a training version."""
    table = {
        "v1": "GPT_SoVITS/pretrained_models/s2G488k.pth",
        "v2": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
        "v3": "GPT_SoVITS/pretrained_models/s2Gv3.pth",
        "v4": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
        "v2Pro": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
        "v2ProPlus": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
    }
    return table.get(version, table["v4"])


def _pretrained_s2d_for_version(version: str) -> str:
    table = {
        "v1": "GPT_SoVITS/pretrained_models/s2D488k.pth",
        "v2": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth",
        "v3": "GPT_SoVITS/pretrained_models/s2Dv3.pth",
        "v4": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Dv4.pth",
        "v2Pro": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth",
        "v2ProPlus": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth",
    }
    return table.get(version, table["v4"])


def _sovits_weights_dir(version: str) -> str:
    table = {
        "v1": "SoVITS_weights",
        "v2": "SoVITS_weights_v2",
        "v3": "SoVITS_weights_v3",
        "v4": "SoVITS_weights_v4",
        "v2Pro": "SoVITS_weights_v2Pro",
        "v2ProPlus": "SoVITS_weights_v2ProPlus",
    }
    return table.get(version, table["v4"])


def _gpt_weights_dir(version: str) -> str:
    table = {
        "v1": "GPT_weights",
        "v2": "GPT_weights_v2",
        "v3": "GPT_weights_v3",
        "v4": "GPT_weights_v4",
        "v2Pro": "GPT_weights_v2Pro",
        "v2ProPlus": "GPT_weights_v2ProPlus",
    }
    return table.get(version, table["v4"])


def _run_fine_tune_sovits(
    *,
    exp_name: str,
    exp_dir: str,
    version: str,
    training_config: Dict[str, Any],
    sovits_weights_dir: str,
) -> str:
    _set_progress(stage="fine_tune_sovits", step=0)

    s2_json_name = (
        "GPT_SoVITS/configs/s2.json"
        if version not in {"v2Pro", "v2ProPlus"}
        else f"GPT_SoVITS/configs/s2{version}.json"
    )
    with open(os.path.join(GPTSOVITS_BASE_DIR, s2_json_name), "r", encoding="utf-8") as handle:
        data = json.load(handle)

    total_epoch = int(training_config.get("sovits_total_epoch", 8))
    batch_size = int(training_config.get("sovits_batch_size", 2))

    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["train"]["text_low_lr_rate"] = 0.4
    data["train"]["pretrained_s2G"] = _pretrained_s2g_for_version(version)
    data["train"]["pretrained_s2D"] = _pretrained_s2d_for_version(version)
    data["train"]["if_save_latest"] = True
    data["train"]["if_save_every_weights"] = True
    data["train"]["save_every_epoch"] = 1
    data["train"]["gpu_numbers"] = "0"
    data["train"]["grad_ckpt"] = False
    data["train"]["lora_rank"] = "32"
    data["model"]["version"] = version
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = exp_dir
    data["save_weight_dir"] = sovits_weights_dir
    data["name"] = exp_name
    data["version"] = version

    tmp_config = os.path.join(WORK_ROOT, f"tmp_s2_{uuid.uuid4().hex[:8]}.json")
    with open(tmp_config, "w", encoding="utf-8") as handle:
        json.dump(data, handle)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [GPTSOVITS_BASE_DIR, os.path.join(GPTSOVITS_BASE_DIR, "GPT_SoVITS")]
    )
    script = (
        "GPT_SoVITS/s2_train.py"
        if version in {"v1", "v2", "v2Pro", "v2ProPlus"}
        else "GPT_SoVITS/s2_train_v3_lora.py"
    )
    subprocess.run(
        [sys.executable, script, "--config", tmp_config],
        env=env,
        cwd=GPTSOVITS_BASE_DIR,
        check=True,
    )

    pattern = re.compile(rf"^{re.escape(exp_name)}_e(\d+)_s(\d+)_l(\d+)\.pth$")
    best = None
    if os.path.isdir(sovits_weights_dir):
        candidates = []
        for name in os.listdir(sovits_weights_dir):
            match = pattern.match(name)
            if match:
                candidates.append((int(match.group(1)), int(match.group(2)), name))
        if candidates:
            candidates.sort(reverse=True)
            best = os.path.join(sovits_weights_dir, candidates[0][2])

    if not best:
        raise RuntimeError(f"No SoVITS checkpoint produced in {sovits_weights_dir}")
    return best


def _run_fine_tune_gpt(
    *,
    exp_name: str,
    exp_dir: str,
    version: str,
    training_config: Dict[str, Any],
    gpt_weights_dir: str,
) -> str:
    _set_progress(stage="fine_tune_gpt", step=0)

    yaml_name = "s1longer.yaml" if version == "v1" else "s1longer-v2.yaml"
    with open(os.path.join(GPTSOVITS_BASE_DIR, "GPT_SoVITS/configs", yaml_name), "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    total_epoch = int(training_config.get("gpt_total_epoch", 15))
    batch_size = int(training_config.get("gpt_batch_size", 8))
    pretrained_s1 = "GPT_SoVITS/pretrained_models/s1v3.ckpt"

    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["train"]["save_every_n_epoch"] = 5
    data["train"]["if_save_every_weights"] = True
    data["train"]["if_save_latest"] = True
    data["train"]["if_dpo"] = False
    data["train"]["half_weights_save_dir"] = gpt_weights_dir
    data["train"]["exp_name"] = exp_name
    data["pretrained_s1"] = pretrained_s1
    data["train_semantic_path"] = os.path.join(exp_dir, "6-name2semantic.tsv")
    data["train_phoneme_path"] = os.path.join(exp_dir, "2-name2text.txt")
    data["output_dir"] = os.path.join(exp_dir, f"logs_s1_{version}")

    tmp_config = os.path.join(WORK_ROOT, f"tmp_s1_{uuid.uuid4().hex[:8]}.yaml")
    with open(tmp_config, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [GPTSOVITS_BASE_DIR, os.path.join(GPTSOVITS_BASE_DIR, "GPT_SoVITS")]
    )
    env["_CUDA_VISIBLE_DEVICES"] = "0"
    env["hz"] = "25hz"
    subprocess.run(
        [sys.executable, "GPT_SoVITS/s1_train.py", "--config_file", tmp_config],
        env=env,
        cwd=GPTSOVITS_BASE_DIR,
        check=True,
    )

    pattern = re.compile(rf"^{re.escape(exp_name)}-e(\d+)\.ckpt$")
    best = None
    if os.path.isdir(gpt_weights_dir):
        candidates = []
        for name in os.listdir(gpt_weights_dir):
            match = pattern.match(name)
            if match:
                candidates.append((int(match.group(1)), name))
        if candidates:
            candidates.sort(reverse=True)
            best = os.path.join(gpt_weights_dir, candidates[0][1])

    if not best:
        raise RuntimeError(f"No GPT checkpoint produced in {gpt_weights_dir}")
    return best


def _send_callback(url: str, payload: Dict[str, Any]) -> None:
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(url, json=payload)
            logger.info(f"[GPT-SoVITS Train] callback sent: {response.status_code}")
    except Exception as exc:
        logger.error(f"[GPT-SoVITS Train] callback failed: {exc}")


def _train(event: Dict[str, Any], inp: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete GPT-SoVITS training pipeline."""
    job_id = uuid.uuid4().hex[:8]
    character_id = inp.get("character_id") or ""
    character_name = inp.get("character_name") or ""
    exp_name = inp.get("exp_name") or character_name or f"job_{job_id}"
    version = inp.get("version", "v4")
    training_config = inp.get("training_config") or {}
    audio_s3 = inp.get("audio_s3_prefix")
    callback_url = inp.get("callback_url")
    if not audio_s3:
        return {"error": "audio_s3_prefix is required"}

    work_dir = os.path.join(WORK_ROOT, f"train_{job_id}")
    raw_dir = os.path.join(work_dir, "raw")
    sliced_dir = os.path.join(work_dir, "sliced")
    stt_dir = os.path.join(work_dir, "stt")
    exp_dir = os.path.join(work_dir, "exp")
    sovits_weights_dir = os.path.join(work_dir, _sovits_weights_dir(version))
    gpt_weights_dir = os.path.join(work_dir, _gpt_weights_dir(version))
    os.makedirs(sovits_weights_dir, exist_ok=True)
    os.makedirs(gpt_weights_dir, exist_ok=True)

    progress_stop = threading.Event()

    def _push_progress() -> None:
        while not progress_stop.is_set():
            try:
                runpod.serverless.progress_update(event, _snapshot_progress())
            except Exception:
                pass
            time.sleep(5)

    pusher = threading.Thread(target=_push_progress, daemon=True)
    pusher.start()

    result: Dict[str, Any] = {
        "job_id": job_id,
        "character_id": character_id,
        "character_name": character_name,
        "exp_name": exp_name,
        "version": version,
        "status": "failed",
        "error_message": None,
    }

    try:
        _set_progress(stage="download_data", step=0)
        input_count = s3_utils.download_prefix(audio_s3, raw_dir)
        if input_count == 0:
            raise RuntimeError(f"no files downloaded from {audio_s3}")

        _run_slicing(raw_dir, sliced_dir)
        stt_list_path = _run_stt(sliced_dir, stt_dir)
        _run_format_dataset(
            exp_name=exp_name,
            exp_dir=exp_dir,
            inp_text=stt_list_path,
            inp_wav_dir=sliced_dir,
            version=version,
            gpu_numbers="0-0",
        )

        sovits_checkpoint = _run_fine_tune_sovits(
            exp_name=exp_name,
            exp_dir=exp_dir,
            version=version,
            training_config=training_config,
            sovits_weights_dir=sovits_weights_dir,
        )
        gpt_checkpoint = _run_fine_tune_gpt(
            exp_name=exp_name,
            exp_dir=exp_dir,
            version=version,
            training_config=training_config,
            gpt_weights_dir=gpt_weights_dir,
        )

        _set_progress(stage="upload", step=0)
        sovits_filename = os.path.basename(sovits_checkpoint)
        gpt_filename = os.path.basename(gpt_checkpoint)
        registry_prefix = MODEL_REGISTRY_S3.rstrip("/")
        sovits_s3_uri = f"{registry_prefix}/{_sovits_weights_dir(version)}/{sovits_filename}"
        gpt_s3_uri = f"{registry_prefix}/{_gpt_weights_dir(version)}/{gpt_filename}"
        s3_utils.upload_file(sovits_checkpoint, sovits_s3_uri, content_type="application/octet-stream")
        s3_utils.upload_file(gpt_checkpoint, gpt_s3_uri, content_type="application/octet-stream")

        result.update(
            {
                "status": "completed",
                "sovits_checkpoint_path": sovits_s3_uri,
                "gpt_checkpoint_path": gpt_s3_uri,
            }
        )
        _set_progress(stage="completed", step=0)
    except Exception as exc:
        logger.exception("[GPT-SoVITS Train] pipeline failed")
        result["status"] = "failed"
        result["error_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
        _set_progress(stage="failed", step=0)
    finally:
        progress_stop.set()
        pusher.join(timeout=2)
        shutil.rmtree(work_dir, ignore_errors=True)

    if callback_url:
        _send_callback(
            callback_url,
            {
                "job_id": job_id,
                "character_id": character_id,
                "character_name": character_name,
                "engine": "gpt-sovits",
                "status": "completed" if result["status"] == "completed" else "failed",
                "result": {
                    "sovits_checkpoint_path": result.get("sovits_checkpoint_path"),
                    "gpt_checkpoint_path": result.get("gpt_checkpoint_path"),
                }
                if result["status"] == "completed"
                else None,
                "error_message": result.get("error_message"),
            },
        )

    return result


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    inp = event.get("input") or {}
    action = inp.get("action") or "train"
    if action != "train":
        return {"error": f"train_handler only supports action='train', got {action!r}"}
    return _train(event, inp)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
