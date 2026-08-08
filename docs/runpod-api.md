# GPT-SoVITS RunPod Serverless API

GPT-SoVITS moved from the on-prem FastAPI server (`:9881`) to RunPod
Serverless. This document is the call contract clients (Forge `voice_service`)
need to use.

Two endpoints share a single image; `dockerArgs` selects the handler module:

| Endpoint | Command | Workers | Timeout |
|---|---|---|---|
| `forge-gpt-sovits-inference` | `python -u -m app.handler` | `workersMin >= 1` | 10 min |
| `forge-gpt-sovits-training` | `python -u -m app.train_handler` | `workersMin = 0` | 6 hours |

## 1. Endpoints

| Use | URL | Notes |
|---|---|---|
| Inference (sync) | `https://api.runpod.ai/v2/<inf-endpoint>/runsync` | 5-minute hard cap |
| Inference (async) | `https://api.runpod.ai/v2/<inf-endpoint>/run` | returns `{id}`, poll `/status/{id}` |
| Training (async) | `https://api.runpod.ai/v2/<train-endpoint>/run` | always async (5–30 min) |
| Job status | `https://api.runpod.ai/v2/<endpoint>/status/{job_id}` | progress included for training |

Endpoint ids are provisioned out-of-band and passed to `voice_service` via
`GPT_SOVITS_INFERENCE_ENDPOINT_ID` / `GPT_SOVITS_TRAINING_ENDPOINT_ID`.

## 2. Common conventions

### Auth

```
Authorization: Bearer <RUNPOD_API_KEY>
Content-Type: application/json
```

### Request body

Always wrap the payload in `{"input": {...}}` and set an `action` field for
inference calls:

```json
{"input": {"action": "tts", "text": "...", "ref_audio": "s3://..."}}
```

Training does not need `action` (the train handler only supports `train`).

### Response body

```json
{
  "status": "COMPLETED" | "FAILED" | "IN_QUEUE" | "IN_PROGRESS",
  "output": { ... },
  "error": "... (on FAILED)"
}
```

Always check **both** `status == "COMPLETED"` and `output.error`. Handlers
catch their own errors and surface them inside `output.error` instead of
raising, so a COMPLETED job can still represent a failure.

## 3. Inference actions (`/runsync`)

### 3.1 `action: "tts"`

```json
{
  "input": {
    "action": "tts",
    "text": "합성할 텍스트",
    "text_lang": "ko",
    "ref_audio": "s3://bucket/voice/reference/char/ref.wav",
    "ref_text": "참조 음성의 transcript (optional)",
    "prompt_lang": "ko",
    "gpt_weights": "s3://.../GPT_weights_v4/<exp>-e15.ckpt",
    "sovits_weights": "s3://.../SoVITS_weights_v4/<exp>_e15_s...pth",
    "temperature": 1.0,
    "top_k": 5,
    "top_p": 1.0,
    "text_split_method": "cut0",
    "batch_size": 1,
    "speed_factor": 1.0,
    "fragment_interval": 0.3,
    "repetition_penalty": 1.35,
    "sample_steps": 32,
    "super_sampling": false,
    "seed": -1,
    "n_samples": 1
  }
}
```

Character weights hot-swap: the worker keeps track of the last
`gpt_weights` and `sovits_weights` it loaded. Repeated calls for the same
character skip the download/`init_*_weights` cost. Omit both fields to use
the base model baked into the volume (zero-shot).

Response (single sample):

```json
{
  "output": {
    "audio_base64": "...",
    "sample_rate": 32000
  }
}
```

`n_samples > 1`:

```json
{
  "output": {
    "n_samples": 3,
    "samples": [
      {"audio_base64": "...", "sample_rate": 32000},
      {"audio_base64": "...", "sample_rate": 32000},
      {"audio_base64": "...", "sample_rate": 32000}
    ]
  }
}
```

### 3.2 `action: "stt"`

```json
{
  "input": {
    "action": "stt",
    "input_s3_prefix": "s3://bucket/voice/preprocessing/sliced/<req_id>/",
    "language": "auto",
    "precision": "float16"
  }
}
```

Response:

```json
{
  "output": {
    "input_file_count": 87,
    "transcription": "<newline-joined>",
    "language_detected": "ko",
    "entries": [
      {"file": "0001.wav", "language": "KO", "text": "..."},
      ...
    ]
  }
}
```

## 4. Training (`/run`, always async)

### 4.1 Start

```json
{
  "input": {
    "character_id": "xxx",
    "character_name": "Evie",
    "exp_name": "evie_f_xxx",
    "version": "v4",
    "audio_s3_prefix": "s3://bucket/voice/training/<char_id>/<audio_dir>/",
    "training_config": {
      "sovits_total_epoch": 8,
      "sovits_batch_size": 2,
      "gpt_total_epoch": 15,
      "gpt_batch_size": 8
    },
    "callback_url": "http://<voice_service>/training/callback"
  }
}
```

Response (immediate):

```json
{"id": "<runpod-job-id>", "status": "IN_QUEUE"}
```

### 4.2 Poll

`GET /v2/<train-endpoint>/status/<job_id>`

```json
{
  "id": "...",
  "status": "IN_PROGRESS",
  "output": {
    "stage": "fine_tune_sovits",
    "step": 0,
    "total_steps": 0,
    "extra": {}
  }
}
```

`stage` progresses through `download_data → slicing → stt → format_dataset →
fine_tune_sovits → fine_tune_gpt → upload → completed | failed`.

Terminal COMPLETED payload:

```json
{
  "status": "COMPLETED",
  "output": {
    "job_id": "internal-uuid",
    "character_id": "xxx",
    "character_name": "Evie",
    "exp_name": "evie_f_xxx",
    "version": "v4",
    "status": "completed",
    "sovits_checkpoint_path": "s3://.../SoVITS_weights_v4/<file>.pth",
    "gpt_checkpoint_path": "s3://.../GPT_weights_v4/<file>.ckpt",
    "error_message": null
  }
}
```

### 4.3 Callback

If `callback_url` is provided, the worker POSTs on completion (success or
failure):

```json
{
  "job_id": "internal-uuid",
  "character_id": "xxx",
  "character_name": "Evie",
  "engine": "gpt-sovits",
  "status": "completed" | "failed",
  "result": {
    "sovits_checkpoint_path": "s3://...",
    "gpt_checkpoint_path": "s3://..."
  },
  "error_message": null
}
```

`voice_service`'s `/training/callback` handler matches by `character_id`
when the internal `job_id` does not correlate to a tracked parallel job.

## 5. Environment variables (handlers read these)

| Key | Default | Purpose |
|---|---|---|
| `GPTSOVITS_BASE_DIR` | `/srv/gpt-sovits` | Source repo root |
| `GPTSOVITS_VOLUME_PATH` | `/runpod-volume/gpt-sovits` | Network Volume mount |
| `GPTSOVITS_WORK_ROOT` | `/tmp/work` | Ephemeral working root |
| `GPTSOVITS_ASR_PATH` | `$GPTSOVITS_VOLUME_PATH/asr/faster-whisper-large-v3` | Whisper weights |
| `GPTSOVITS_MODEL_REGISTRY_S3` | `s3://shiftup-enterprise-ai-service/voice/model_registry/GPT-SoVITS` | Where train uploads checkpoints |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | — | S3 access |
| `RUNPOD_API_KEY` | — | Not read by handler; needed by clients |

## 6. Cold start

First worker spinup takes 60–120s (image pull + base model load). Keep
`workersMin >= 1` on the inference endpoint to avoid this on every request.
Training always runs with `workersMin = 0` since the cold start is dwarfed
by a 5–30 minute training run.
