# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPT-SoVITS is a powerful few-shot voice conversion and text-to-speech (TTS) system. It supports:
- Zero-shot TTS with 5-second vocal samples
- Few-shot TTS with 1-minute training data
- Cross-lingual support (English, Japanese, Korean, Cantonese, Chinese)
- Multiple model versions (v1, v2, v3, v4, v2Pro, v2ProPlus)

## Environment Setup

### Installation

**Linux (including WSL):**
```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
bash install.sh --device <CU126|CU128|ROCM|CPU> --source <HF|HF-Mirror|ModelScope> [--download-uvr5]
```

**Manual Installation:**
```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
pip install -r extra-req.txt --no-deps
pip install -r requirements.txt
conda install ffmpeg  # or use system package manager
```

### Tested Environments
- Python 3.10-3.12
- PyTorch 2.5.1+ with CUDA 12.4+
- Apple Silicon (MPS) and CPU also supported

## Running the Application

### WebUI (Main Application)

**Start full WebUI:**
```bash
python webui.py [language]
```
Language options: `Auto`, `zh_CN`, `en_US`, `ja_JP`, `ko_KR`

**Start v1 WebUI:**
```bash
python webui.py v1 [language]
```

**Inference WebUI only:**
```bash
python GPT_SoVITS/inference_webui.py [language]
```

**Fast inference WebUI:**
```bash
python GPT_SoVITS/inference_webui_fast.py [language]
```

### API Server

**Start API server:**
```bash
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

Parameters:
- `-a`: Bind address (default: 127.0.0.1)
- `-p`: Port (default: 9880)
- `-c`: TTS config file path

**API Endpoints:**
- `/tts` - Text-to-speech inference (GET/POST)
- `/set_gpt_weights` - Switch GPT model
- `/set_sovits_weights` - Switch SoVITS model
- `/control` - Control commands (restart/exit)

## Training

### Two-Stage Training Process

**Stage 1: GPT Model Training (Text-to-Semantic)**
```bash
python GPT_SoVITS/s1_train.py --config_file <path_to_config.yaml>
```
Config files in `GPT_SoVITS/configs/`:
- `s1.yaml` - Base configuration
- `s1longer.yaml` - Longer training
- `s1longer-v2.yaml` - v2 longer training

**Stage 2: SoVITS Model Training (VITS-based vocoder)**

For v1/v2:
```bash
python GPT_SoVITS/s2_train.py
```

For v3/v4:
```bash
python GPT_SoVITS/s2_train.py
```

For v3 with LoRA:
```bash
python GPT_SoVITS/s2_train_v3_lora.py
```

Training relies on `config.py` and dataset .list files.

## Dataset Preparation

### Dataset Format

TTS annotation `.list` file format:
```
vocal_path|speaker_name|language|text
```

Example:
```
/path/to/audio.wav|speaker_name|en|This is the transcription text.
```

Language codes:
- `zh` - Chinese
- `ja` - Japanese
- `en` - English
- `ko` - Korean
- `yue` - Cantonese

### Audio Processing Tools

**Audio slicing:**
```bash
python audio_slicer.py \
    --input_path <path_to_audio> \
    --output_root <output_directory> \
    --threshold <volume_threshold> \
    --min_length <minimum_duration> \
    --min_interval <gap_between_clips> \
    --hop_size <step_size>
```

**ASR (Automatic Speech Recognition):**

Chinese (FunASR):
```bash
python tools/asr/funasr_asr.py -i <input> -o <output>
```

Other languages (Faster-Whisper):
```bash
python tools/asr/fasterwhisper_asr.py -i <input> -o <output> -l <language> -p <precision>
```

**UVR5 (Vocal/Accompaniment Separation):**
```bash
python tools/uvr5/webui.py <infer_device> <is_half> <webui_port>
```

**Denoising:**
```bash
python tools/cmd-denoise.py
```

## Architecture

### Core Components

**Main Modules:**
- `GPT_SoVITS/AR/` - Autoregressive text-to-semantic model (GPT-based)
  - `models/t2s_lightning_module.py` - PyTorch Lightning training module
  - `data/data_module.py` - Data loading for text-to-semantic
- `GPT_SoVITS/TTS_infer_pack/` - TTS inference pipeline
  - `TTS.py` - Main TTS class (70KB+ file with inference logic)
  - `TextPreprocessor.py` - Text preprocessing and language handling
  - `text_segmentation_method.py` - Text segmentation strategies
- `GPT_SoVITS/module/` - VITS model components (generators, discriminators)
- `GPT_SoVITS/feature_extractor/` - Audio feature extraction
- `GPT_SoVITS/text/` - Text processing for different languages
- `GPT_SoVITS/BigVGAN/` - Vocoder implementation

**Supporting Tools:**
- `tools/asr/` - ASR model managers (FunASR, Faster-Whisper)
- `tools/uvr5/` - Audio source separation
- `tools/i18n/` - Internationalization

### Model Versions

The system supports multiple model versions with different capabilities:
- **v1**: Original model (2k hours training)
- **v2**: Extended to 5k hours, supports Korean/Cantonese
- **v3**: Higher timbre similarity, fewer repetitions
- **v4**: Fixes v3 artifacts, native 48kHz output
- **v2Pro/v2ProPlus**: Enhanced performance at v2 speed

Each version has:
- GPT weights (s1*.ckpt) in `GPT_weights_v*` directories
- SoVITS weights (s2*.pth) in `SoVITS_weights_v*` directories
- Pretrained models in `GPT_SoVITS/pretrained_models/`

### Configuration

**Main config:** `config.py`
- Device selection and GPU info
- Model path mappings (`pretrained_sovits_name`, `pretrained_gpt_name`)
- WebUI ports (main: 9874, uvr5: 9873, infer_tts: 9872, subfix: 9871)
- API port (9880)

**Inference config:** `GPT_SoVITS/configs/tts_infer.yaml`
- Version-specific model paths
- Device settings (cuda/cpu/mps)
- Half-precision flag

## Model Weights

### Directory Structure
```
GPT_weights/          # v1 GPT models
GPT_weights_v2/       # v2 GPT models
GPT_weights_v3/       # v3 GPT models
GPT_weights_v4/       # v4 GPT models
GPT_weights_v2Pro/    # v2Pro GPT models
SoVITS_weights/       # v1 SoVITS models
SoVITS_weights_v2/    # v2 SoVITS models
SoVITS_weights_v3/    # v3 SoVITS models
SoVITS_weights_v4/    # v4 SoVITS models
SoVITS_weights_v2Pro/ # v2Pro SoVITS models
GPT_SoVITS/pretrained_models/  # Pretrained base models
```

### Pretrained Models Location
- Chinese-Hubert: `GPT_SoVITS/pretrained_models/chinese-hubert-base`
- Chinese-Roberta: `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`
- G2PW (Chinese TTS): `GPT_SoVITS/text/G2PWModel`
- UVR5 weights: `tools/uvr5/uvr5_weights`
- ASR models: `tools/asr/models`

## Working with the Code

### Two-Model Architecture

The system uses a two-stage approach:
1. **GPT (s1)**: Text → Semantic tokens (autoregressive transformer)
2. **SoVITS (s2)**: Semantic tokens → Audio waveform (VITS-based)

When modifying inference or training:
- Stage 1 code is in `GPT_SoVITS/AR/`
- Stage 2 code is in `GPT_SoVITS/module/` and uses `s2_train.py`
- Inference combines both in `GPT_SoVITS/TTS_infer_pack/TTS.py`

### Text Processing Pipeline

Text goes through:
1. `TextPreprocessor.py` - Language detection and normalization
2. Language-specific processing in `GPT_SoVITS/text/`
3. Segmentation via `text_segmentation_method.py` (cut0-cut5 methods)
4. Conversion to phonemes/tokens for GPT model

### Important Code Locations

- Main WebUI: `webui.py` (sets up all tabs and Gradio interface)
- Training entry points: `s1_train.py`, `s2_train.py`, `s2_train_v3.py`
- Inference logic: `GPT_SoVITS/TTS_infer_pack/TTS.py` (~70KB, handles full TTS pipeline)
- Checkpoint processing: `process_ckpt.py` (model weight saving utilities)
- Model exports: `export_torch_script.py`, `export_torch_script_v3v4.py`, `onnx_export.py`

### Environment Variables

Key environment variables used:
- `version` - Model version (v1/v2/v3/v4/v2Pro/v2ProPlus)
- `is_half` - Enable FP16 precision
- `is_share` - Enable Gradio sharing
- `language` - UI language
- `CUDA_VISIBLE_DEVICES` - GPU selection
- `HF_ENDPOINT` - Hugging Face mirror endpoint

### ASR Model Management

The codebase includes a lazy-loaded singleton pattern for STT (Speech-To-Text) models with manual eviction:
- See `tools/asr/model_manager.py` for the implementation
- Models are loaded on demand and can be evicted to free memory
- Recent commit indicates STT enhancements: "STT Lazy-loaded Singleton with Manual Eviction"

## Docker Support

Build Docker image:
```bash
bash docker_build.sh --cuda <12.6|12.8> [--lite]
```

Run with Docker Compose:
```bash
docker compose run --service-ports <GPT-SoVITS-CU126|GPT-SoVITS-CU128|GPT-SoVITS-CU126-Lite|GPT-SoVITS-CU128-Lite>
```

## Common Issues

- **Memory**: v3/v4 require more VRAM than v1/v2. Use `is_half=true` to reduce usage.
- **Audio Quality**: v3/v4 are sensitive to reference audio quality. v1/v2/v2Pro work better with average quality datasets.
- **Mac Training**: GPU training on Mac produces lower quality; use CPU instead.
- **Language Support**: G2PW models required for Chinese. Faster-Whisper for non-Chinese ASR.
