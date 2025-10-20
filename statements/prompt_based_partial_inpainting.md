# Prompt-based Partial Inpainting Implementation Plan

## 목표

CFM의 prompt conditioning 메커니즘을 활용하여 생성된 오디오의 일부분만 재생성(inpainting)하는 기능 구현.

---

## 문제 정의

### 현재 상태

**전체 재생성만 가능**:
- `TTS.run()` 메서드는 전체 텍스트를 처음부터 끝까지 생성
- 사용자가 일부분만 수정하고 싶어도 전체를 다시 생성해야 함
- 예: "안녕하세요 반갑습니다" → "반갑습니다"만 "만나서 기쁩니다"로 바꾸고 싶어도 전체 재생성

**기존 TTS의 한계**:
- **VITS (v1/v2/v2ProPlus)**: Autoregressive decoder → 한 번 생성하면 수정 불가
- **Diffusion 기반**: Inpainting 가능하지만 GPT-SoVITS는 diffusion 아님
- **일반 Flow**: 전역적 변환 → 부분 수정 어려움

### 해결 방향

**CFM의 Prompt Conditioning 활용**:
- v4의 CFM은 **prefix를 고정**하는 메커니즘 내장
- 코드 증거: `models.py:1032-1034, 1084`
  ```python
  prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
  x[:, :, :prompt_len] = 0  # Prefix는 ODE에서 업데이트 안 됨
  ```
- 이를 inpainting으로 확장: **재생성하지 않을 부분을 prompt로 고정**

---

## 아키텍처 설계

### 핵심 아이디어

**Prompt = 유지할 원본 오디오 구간**

```
원본 오디오:     [===== A =====][===== B =====][===== C =====]
재생성 요청:     [   유지 A   ][  재생성 B'  ][   유지 C   ]
                        ↓
Prompt 설정:     [   Prompt   ][   Generate  ][  Post-fill  ]
```

1. **A 구간**: Prompt로 설정 → CFM이 자동으로 유지
2. **B 구간**: CFM이 새로 생성
3. **C 구간**: 생성 후 원본으로 교체 (cross-fade)

**왜 가능한가**:
- CFM은 `x[:, :, :prompt_len] = 0`로 매 ODE step마다 prefix를 강제로 고정
- Prompt 뒷부분은 자연스럽게 이어지도록 DiT가 학습되어 있음 (autoregressive 특성)

---

## Phase 1: Character-to-Frame Alignment

### 문제

**문자 인덱스 → Mel frame 인덱스 매핑 필요**:
- 사용자 입력: "안녕하세요 반갑습니다"의 6번째 문자부터 재생성
- 필요한 정보: 6번째 문자가 mel-spectrogram의 몇 번째 frame인지

### 해결책 1: Linear Approximation (단순, 빠름)

**새 파일**: `GPT_SoVITS/TTS_infer_pack/alignment.py`

```python
class CharToFrameAligner:
    """문자 인덱스를 mel frame 인덱스로 매핑"""

    def __init__(self, mel_frame_rate: float = 150.0):
        """
        Args:
            mel_frame_rate: v4는 48kHz / 320 hop = 150 fps
        """
        self.mel_frame_rate = mel_frame_rate

    def align_linear(
        self,
        text: str,
        total_frames: int
    ) -> Dict[int, int]:
        """
        선형 근사: 문자 길이에 비례하여 frame 분배

        Args:
            text: 원본 텍스트 "안녕하세요 반갑습니다"
            total_frames: Mel-spectrogram의 전체 frame 수

        Returns:
            {char_idx: frame_idx, ...}
            예: {0: 0, 1: 10, 2: 20, ...}
        """
        total_chars = len(text)
        if total_chars == 0:
            return {}

        char_to_frame = {}
        for i in range(total_chars + 1):  # +1 for end position
            frame_idx = int(i * total_frames / total_chars)
            char_to_frame[i] = frame_idx

        return char_to_frame

    def get_frame_range(
        self,
        text: str,
        start_char_idx: int,
        end_char_idx: int,
        total_frames: int
    ) -> Tuple[int, int]:
        """
        문자 범위 → frame 범위 변환

        Returns:
            (start_frame, end_frame)
        """
        char_to_frame = self.align_linear(text, total_frames)
        start_frame = char_to_frame[start_char_idx]
        end_frame = char_to_frame[end_char_idx]
        return start_frame, end_frame
```

**한계**:
- 모든 문자가 동일한 발화 시간을 가진다고 가정
- 실제로는 음절마다 duration이 다름 (예: "ㅏ"는 짧고 "ㅆ"는 김)

**왜 이 방법**:
- ✅ 구현 단순 → 빠른 프로토타이핑
- ✅ 청각적으로 ±100ms 오차는 인지 어려움
- ✅ 향후 정교한 alignment로 교체 가능 (인터페이스 유지)

### 해결책 2: Phoneme-based Alignment (정교함)

```python
class CharToFrameAligner:
    def align_phoneme_based(
        self,
        text: str,
        phones: List[str],
        phone_durations: List[float],  # Duration predictor 출력
        mel_frame_rate: float = 150.0
    ) -> Dict[int, int]:
        """
        Phoneme duration 기반 정밀 매핑

        Args:
            phones: ["ㅇ", "ㅏ", "ㄴ", "ㄴ", "ㅕ", "ㅇ", ...]
            phone_durations: [0.05, 0.08, 0.06, ...] (초 단위)

        Returns:
            문자별 frame 시작 위치
        """
        # 1. 문자 → phoneme 매핑 (G2P 역과정)
        char_to_phones = self._map_char_to_phones(text, phones)

        # 2. Phoneme duration → frame 누적
        char_to_frame = {}
        cumulative_frame = 0

        for char_idx, phone_list in char_to_phones.items():
            char_to_frame[char_idx] = cumulative_frame

            # 해당 문자의 phoneme duration 합산
            for phone in phone_list:
                phone_idx = phones.index(phone)
                duration = phone_durations[phone_idx]
                cumulative_frame += int(duration * mel_frame_rate)

        return char_to_frame

    def _map_char_to_phones(
        self,
        text: str,
        phones: List[str]
    ) -> Dict[int, List[str]]:
        """
        문자 → phoneme 리스트 매핑

        예: "안녕" → {0: ["ㅇ", "ㅏ", "ㄴ"], 1: ["ㄴ", "ㅕ", "ㅇ"]}
        """
        # G2P 출력과 대조하여 매핑 (한국어 G2P는 문자 순서 유지)
        # 구현 복잡도 높음 → Phase 2로 연기 가능
        pass
```

**구현 우선순위**:
- Phase 1: `align_linear()` 사용 (빠른 구현)
- Phase 2: `align_phoneme_based()` 추가 (정확도 개선)

---

## Phase 2: Audio-to-Mel Conversion

### 필요성

**원본 오디오 → Mel-spectrogram 변환**:
- Inpainting을 위해서는 원본 오디오의 mel 표현 필요
- 기존 `TTS.py`에는 inference만 있고 audio → mel 변환 없음

### 구현

**파일**: `GPT_SoVITS/TTS_infer_pack/TTS.py`

```python
class TTS:
    # ... 기존 코드

    def _audio_to_mel(
        self,
        audio_path: str
    ) -> torch.Tensor:
        """
        오디오 파일 → Mel-spectrogram 변환

        Returns:
            mel: (1, 100, T) tensor
        """
        import torchaudio

        # 1. Load audio
        audio, sr = torchaudio.load(audio_path)
        audio = audio.to(self.configs.device).float()

        # Mono conversion
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)

        # 2. Resample to 48kHz (v4 native)
        if sr != 48000:
            from TTS_infer_pack.TTS import resample
            audio = resample(audio, sr, 48000, self.configs.device)

        # 3. Mel extraction (v4는 mel_fn_v4 사용)
        from TTS_infer_pack.TTS import mel_fn_v4, norm_spec
        mel = mel_fn_v4(audio)  # (1, 100, T)
        mel = norm_spec(mel)    # Normalize [-1, 1]

        return mel

    def _mel_to_audio(
        self,
        mel: torch.Tensor
    ) -> torch.Tensor:
        """
        Mel-spectrogram → 오디오 변환

        Args:
            mel: (1, 100, T) normalized mel

        Returns:
            audio: (T_audio,) waveform
        """
        from TTS_infer_pack.TTS import denorm_spec

        mel = denorm_spec(mel)  # Denormalize

        # Vocoder
        with torch.inference_mode():
            audio = self.vocoder(mel)  # (1, 1, T_audio)
            audio = audio[0, 0]  # (T_audio,)

        return audio
```

**재사용**:
- `using_vocoder_synthesis()`의 mel 생성 로직과 유사
- Vocoder 호출 부분 재사용

---

## Phase 3: Cross-fade 처리

### 문제

**생성 구간과 원본 구간의 경계 처리**:
- Hard cut은 click/pop 노이즈 발생
- 특히 mel-spectrogram에서 갑작스러운 불연속은 청각적으로 거슬림

### 해결: Cross-fade

**파일**: `GPT_SoVITS/TTS_infer_pack/TTS.py`

```python
class TTS:
    def _crossfade_mel(
        self,
        mel1: torch.Tensor,
        mel2: torch.Tensor,
        fade_frames: int = 50
    ) -> torch.Tensor:
        """
        두 mel-spectrogram을 부드럽게 연결

        Args:
            mel1: (B, C, T1) - 앞부분
            mel2: (B, C, T2) - 뒷부분
            fade_frames: Cross-fade 구간 길이 (default: 50 frames ≈ 330ms)

        Returns:
            merged: (B, C, T1 + T2 - fade_frames)
        """
        # 1. Overlap 영역 추출
        mel1_end = mel1[:, :, -fade_frames:]    # (B, C, fade_frames)
        mel2_start = mel2[:, :, :fade_frames]   # (B, C, fade_frames)

        # 2. Fade curve 생성
        fade_out = torch.linspace(1, 0, fade_frames, device=mel1.device)  # 1 → 0
        fade_in = torch.linspace(0, 1, fade_frames, device=mel1.device)   # 0 → 1

        # (fade_frames,) → (1, 1, fade_frames) for broadcasting
        fade_out = fade_out.view(1, 1, -1)
        fade_in = fade_in.view(1, 1, -1)

        # 3. Cross-fade
        crossfade_region = mel1_end * fade_out + mel2_start * fade_in

        # 4. 결합
        result = torch.cat([
            mel1[:, :, :-fade_frames],  # mel1의 앞부분
            crossfade_region,           # Cross-fade 영역
            mel2[:, :, fade_frames:]    # mel2의 뒷부분
        ], dim=2)

        return result
```

**Fade curve 선택**:
- Linear fade: 간단하지만 중간 지점에서 에너지 감소
- Equal-power fade: `sqrt(fade_out)` + `sqrt(fade_in)` → 에너지 일정 유지
- 현재는 linear 사용 (구현 단순), 필요시 교체

**Fade length 결정**:
- 50 frames ≈ 330ms (150 fps 기준)
- 너무 짧으면: 여전히 불연속 느껴짐
- 너무 길면: 재생성 구간이 침범받음
- 경험적으로 300-500ms가 적절

---

## Phase 4: Main Inpainting Logic

### 새 API 메서드

**파일**: `GPT_SoVITS/TTS_infer_pack/TTS.py`

```python
class TTS:
    @torch.no_grad()
    def regenerate_partial(
        self,
        original_audio_path: str,
        original_text: str,
        start_char_idx: int,
        end_char_idx: int,
        new_text: str,
        ref_audio_path: str,
        prompt_text: str = "",
        prompt_lang: str = "ko",
        text_lang: str = "ko",
        sample_steps: int = 32,
        speed_factor: float = 1.0,
    ) -> Tuple[int, np.ndarray]:
        """
        부분 재생성 (v4 전용)

        Args:
            original_audio_path: 원본 오디오 파일 경로
            original_text: 원본 텍스트 "안녕하세요 반갑습니다"
            start_char_idx: 재생성 시작 문자 인덱스 (예: 6)
            end_char_idx: 재생성 종료 문자 인덱스 (예: 11)
            new_text: 새로운 텍스트 (예: "만나서 기쁩니다")
            ref_audio_path: Reference audio (speaker identity)
            prompt_text: Reference text
            prompt_lang: Reference text language
            text_lang: Target text language
            sample_steps: CFM ODE steps
            speed_factor: Speech speed

        Returns:
            (sample_rate, audio_array)

        Example:
            원본: "안녕하세요 반갑습니다" (0-11)
            재생성: start=6, end=11 (공백 포함)
            new_text: "만나서 기쁩니다"
            결과: "안녕하세요 만나서 기쁩니다"
        """
        assert self.configs.version == "v4", "Partial regeneration only supports v4!"

        # ===== Step 1: 원본 mel 추출 =====
        original_mel = self._audio_to_mel(original_audio_path)  # (1, 100, T_orig)

        # ===== Step 2: 문자 → Frame 매핑 =====
        from TTS_infer_pack.alignment import CharToFrameAligner
        aligner = CharToFrameAligner(mel_frame_rate=150.0)

        start_frame, end_frame = aligner.get_frame_range(
            original_text,
            start_char_idx,
            end_char_idx,
            total_frames=original_mel.size(2)
        )

        # ===== Step 3: Prompt 설정 (재생성 이전 부분) =====
        prompt_mel = original_mel[:, :, :start_frame]  # (1, 100, start_frame)
        prompt_len = start_frame

        # ===== Step 4: 새 텍스트 합성 =====
        # 전체 텍스트 재구성
        before_text = original_text[:start_char_idx]
        after_text = original_text[end_char_idx:]
        full_new_text = before_text + new_text + after_text

        # Reference audio 설정
        self.set_ref_audio(ref_audio_path)
        if prompt_text:
            # Prompt text preprocessing
            phones, bert_features, norm_text = \
                self.text_preprocessor.segment_and_extract_feature_for_text(
                    prompt_text, prompt_lang, self.configs.version
                )
            self.prompt_cache["prompt_text"] = prompt_text
            self.prompt_cache["phones"] = phones
            self.prompt_cache["bert_features"] = bert_features
            self.prompt_cache["norm_text"] = norm_text

        # Text preprocessing (전체)
        data = self.text_preprocessor.preprocess(
            full_new_text, text_lang, "cut0", self.configs.version
        )

        # ===== Step 5: GPT inference (semantic tokens) =====
        # Batch 생성
        batch, _ = self.to_batch(
            data,
            prompt_data=self.prompt_cache,
            batch_size=1,
            threshold=0.75,
            split_bucket=False,
            device=self.configs.device,
            precision=self.precision,
        )

        item = batch[0]
        batch_phones = item["phones"]
        all_phoneme_ids = item["all_phones"]
        all_phoneme_lens = item["all_phones_len"]
        all_bert_features = item["all_bert_features"]

        prompt_semantic = self.prompt_cache["prompt_semantic"].expand(
            len(all_phoneme_ids), -1
        ).to(self.configs.device)

        # GPT inference
        pred_semantic_list, idx_list = self.t2s_model.model.infer_panel(
            all_phoneme_ids,
            all_phoneme_lens,
            prompt_semantic,
            all_bert_features,
            top_k=5,
            top_p=1.0,
            temperature=1.0,
            early_stop_num=self.configs.hz * self.configs.max_sec,
        )

        # ===== Step 6: SoVITS feature extraction =====
        pred_semantic = pred_semantic_list[0][-idx_list[0]:]
        pred_semantic = pred_semantic.unsqueeze(0).unsqueeze(0)
        phones = batch_phones[0].unsqueeze(0).to(self.configs.device)

        refer_spec = self.prompt_cache["refer_spec"][0][0]
        refer_spec = refer_spec.to(dtype=self.precision, device=self.configs.device)

        # Feature extraction (전체)
        fea_full, ge = self.vits_model.decode_encp(
            pred_semantic, phones, refer_spec, speed=speed_factor
        )

        # ===== Step 7: CFM inference with prompt =====
        # Reference mel for prompt
        ref_audio_tensor = self.prompt_cache["raw_audio"].to(self.configs.device).float()
        if ref_audio_tensor.shape[0] == 2:
            ref_audio_tensor = ref_audio_tensor.mean(0).unsqueeze(0)

        from TTS_infer_pack.TTS import mel_fn_v4, norm_spec, denorm_spec
        ref_mel = mel_fn_v4(ref_audio_tensor)
        ref_mel = norm_spec(ref_mel).to(self.precision)

        # T_ref 맞추기
        T_ref = min(ref_mel.size(2), prompt_mel.size(2), 500)  # vocoder T_ref
        ref_mel = ref_mel[:, :, :T_ref]
        prompt_mel_for_cfm = prompt_mel[:, :, :T_ref]

        # CFM inference (chunked)
        fea_full = fea_full.transpose(2, 1)  # (B, T, C)

        generated_mel = self.vits_model.cfm.inference(
            mu=fea_full,
            x_lens=torch.LongTensor([fea_full.size(1)]).to(self.configs.device),
            prompt=prompt_mel_for_cfm,  # ★ Prompt로 앞부분 고정
            n_timesteps=sample_steps,
            temperature=1.0,
            inference_cfg_rate=0
        )  # (B, 100, T_new)

        # ===== Step 8: 뒷부분 원본으로 교체 (필요시) =====
        if end_char_idx < len(original_text):
            # 생성된 mel에서 재생성 구간 끝 위치 추정
            # 원본 대비 길이 변화 비율 계산
            original_duration_frames = end_frame - start_frame
            new_text_duration_estimate = int(
                len(new_text) / len(original_text[start_char_idx:end_char_idx])
                * original_duration_frames
            )
            end_frame_new = start_frame + new_text_duration_estimate

            # 원본 뒷부분
            original_tail = original_mel[:, :, end_frame:]

            # Cross-fade로 연결
            generated_mel = self._crossfade_mel(
                generated_mel[:, :, :end_frame_new],
                original_tail,
                fade_frames=50
            )

        # ===== Step 9: Vocoder =====
        generated_mel = denorm_spec(generated_mel)
        audio = self._mel_to_audio(generated_mel)

        # ===== Step 10: Postprocessing =====
        audio = audio.cpu().numpy()
        audio = (audio * 32768).astype(np.int16)

        return 48000, audio
```

---

## Phase 5: API Integration

### REST API 추가

**파일**: `api_v2.py`

```python
class PartialRegenerationRequest(BaseModel):
    original_audio_path: str
    original_text: str
    start_char_idx: int
    end_char_idx: int
    new_text: str
    ref_audio_path: str
    prompt_text: str = ""
    prompt_lang: str = "ko"
    text_lang: str = "ko"
    sample_steps: int = 32
    speed_factor: float = 1.0
    media_type: str = "wav"

@APP.post("/tts/partial_regenerate")
async def partial_regenerate_endpoint(request: PartialRegenerationRequest):
    """
    부분 재생성 API (v4 전용)

    Example Request:
    {
        "original_audio_path": "outputs/original.wav",
        "original_text": "안녕하세요 반갑습니다",
        "start_char_idx": 6,
        "end_char_idx": 11,
        "new_text": "만나서 기쁩니다",
        "ref_audio_path": "samples/speaker1.wav",
        "prompt_text": "안녕하세요",
        "sample_steps": 32
    }
    """
    try:
        sr, audio = tts_pipeline.regenerate_partial(
            original_audio_path=request.original_audio_path,
            original_text=request.original_text,
            start_char_idx=request.start_char_idx,
            end_char_idx=request.end_char_idx,
            new_text=request.new_text,
            ref_audio_path=request.ref_audio_path,
            prompt_text=request.prompt_text,
            prompt_lang=request.prompt_lang,
            text_lang=request.text_lang,
            sample_steps=request.sample_steps,
            speed_factor=request.speed_factor,
        )

        # Audio encoding
        io_buffer = BytesIO()
        if request.media_type == "wav":
            pack_wav(io_buffer, audio, sr)
        # ... (기타 포맷)

        return StreamingResponse(io_buffer, media_type=f"audio/{request.media_type}")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
```

---

## 구현 순서

### Step 1: Alignment Module
1. `TTS_infer_pack/alignment.py` 신규 작성
2. `CharToFrameAligner.align_linear()` 구현
3. 단위 테스트: 문자 범위 → frame 범위 변환 검증

### Step 2: Audio ↔ Mel Conversion
1. `TTS.py`에 `_audio_to_mel()` 추가
2. `_mel_to_audio()` 추가 (기존 vocoder 재사용)
3. 왕복 변환 테스트: audio → mel → audio 품질 확인

### Step 3: Cross-fade
1. `_crossfade_mel()` 구현
2. 단위 테스트: 두 mel을 부드럽게 연결
3. 청각 테스트: Click/pop 노이즈 없는지 확인

### Step 4: Inpainting Logic
1. `regenerate_partial()` 메서드 구현
2. 간단한 케이스 테스트: "안녕 반가워" → "안녕 좋아"
3. Edge case 처리:
   - `start_char_idx = 0` (처음부터 재생성)
   - `end_char_idx = len(text)` (끝까지 재생성)
   - Cross-fade 영역이 부족한 경우

### Step 5: API & Integration
1. `api_v2.py`에 `/tts/partial_regenerate` 엔드포인트 추가
2. End-to-end 테스트
3. 다양한 재생성 패턴 검증

## 한계점 및 향후 개선

### 현재 방법의 한계

1. **Alignment 정확도**:
   - Linear approximation은 음절 길이 차이를 반영 못 함
   - 오차가 클 경우 재생성 구간이 의도와 다를 수 있음
   - **개선**: Phoneme-level alignment 또는 Forced Aligner (Montreal Forced Aligner 등)

2. **Suffix 고정 불가**:
   - CFM은 autoregressive 특성상 뒷부분은 앞부분에 영향받음
   - "A [재생성] C" 요청 시 C 부분이 재생성 영향을 받음
   - **개선**: Bidirectional CFM (연구 필요) 또는 더 긴 cross-fade

3. **길이 변화 처리**:
   - 새 텍스트가 원본보다 길거나 짧으면 뒷부분 정렬이 어려움
   - 현재는 선형 비례로 추정 → 부정확할 수 있음
   - **개선**: 생성된 mel의 실제 길이로 dynamic alignment

4. **Speaker Consistency**:
   - 재생성 부분과 원본의 화자가 다를 수 있음 (특히 긴 구간)
   - **개선**: Reference audio를 원본 audio에서 추출

### 왜 이 방법을 선택했나

**CFM의 구조적 장점**:
- ✅ **Prompt conditioning이 내장**: 추가 학습 없이 prefix 고정 가능
- ✅ **코드 증거 명확**: `x[:, :, :prompt_len] = 0` (매 ODE step)
- ✅ **Inference-time만 수정**: 기존 학습된 모델 그대로 사용

**다른 방법 대비**:
- **VITS inpainting**: Duration predictor 재학습 + Flow inversion 필요 → 복잡
- **Diffusion inpainting**: GPT-SoVITS는 diffusion 아님 → 아키텍처 변경 필요
- **Sentence-level concat**: 경계가 부자연스러움 → Cross-fade로도 한계

**실용성 우선**:
- Alignment 오차 ±200ms는 대부분의 use case에서 허용 가능
- Suffix 고정 불가는 cross-fade로 완화
- Speaker consistency는 reference audio 선택으로 해결

---

## 참고 자료

### 코드 위치
- **CFM Prompt Conditioning**: `GPT_SoVITS/module/models.py:1032-1034, 1084`
- **CFM Inference**: `GPT_SoVITS/module/models.py:1027-1085`
- **TTS Pipeline**: `GPT_SoVITS/TTS_infer_pack/TTS.py`
- **Vocoder (v4)**: `GPT_SoVITS/module/models.py:631-654` (Generator)

### 관련 기술
- **Conditional Flow Matching**: Inference 시 condition을 바꿔도 일관성 유지
- **Autoregressive Property**: 앞부분 고정 시 뒷부분이 자연스럽게 이어짐
- **Mel-Spectrogram Cross-fade**: 주파수 영역에서의 부드러운 전환

### 비슷한 연구
- **VoiceBox** (Meta): Flow-based model with in-filling capability
- **AudioLM** (Google): Semantic token inpainting for audio generation
- **SoundStorm**: Parallel audio generation with masking (GPT-SoVITS의 GPT 부분과 유사)

---

## 사용 예시

### CLI Example

```bash
# 원본 오디오: "안녕하세요 반갑습니다"
# 목표: "반갑습니다"를 "만나서 기쁩니다"로 변경

curl -X POST http://localhost:9880/tts/partial_regenerate \
  -H "Content-Type: application/json" \
  -d '{
    "original_audio_path": "outputs/original.wav",
    "original_text": "안녕하세요 반갑습니다",
    "start_char_idx": 6,
    "end_char_idx": 11,
    "new_text": "만나서 기쁩니다",
    "ref_audio_path": "samples/speaker1.wav",
    "prompt_text": "안녕하세요",
    "sample_steps": 32
  }' \
  --output result.wav
```

### Python Example

```python
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

# 모델 로드
config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts = TTS(config)

# 부분 재생성
sr, audio = tts.regenerate_partial(
    original_audio_path="outputs/original.wav",
    original_text="안녕하세요 반갑습니다",
    start_char_idx=6,
    end_char_idx=11,
    new_text="만나서 기쁩니다",
    ref_audio_path="samples/speaker1.wav",
    prompt_text="안녕하세요",
    sample_steps=32
)

# 저장
import soundfile as sf
sf.write("result.wav", audio, sr)
```
