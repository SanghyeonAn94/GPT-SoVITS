# CFM Pause Conditioning Implementation Plan

## 목표

Conditional Flow Matching (CFM)의 time-continuous 특성을 활용하여 텍스트 내 구두점(쉼표, 점 등)을 정확한 pause duration으로 변환하는 기능 구현.

---

## 문제 정의

### 현재 상태

**구두점 처리의 한계**:
- `TextPreprocessor.py:235-238`의 `replace_consecutive_punctuation()` 메서드가 연속된 구두점을 단일 구두점으로 축소
  ```python
  # 현재 동작
  "나는 말야..." → "나는 말야."  # "..."가 "."로 축소됨
  "조심스럽지만,," → "조심스럽지만," # 감정적 뉘앙스 손실
  ```

**감정 표현 불가**:
- 사용자가 의도한 pause 길이 정보가 전처리 단계에서 소실
- 모든 쉼표는 동일한 짧은 pause로 처리
- 말줄임표(...), 연속 쉼표(,,) 등의 의도가 반영되지 않음

### 해결 방향

**CFM의 장점 활용**:
- v4는 CFM 기반 mel generation을 사용 → **시간 축에서의 명시적 제어 가능**
- ODE solver의 각 step에서 특정 frame에 silence를 강제 주입 가능
- v1/v2의 VITS와 달리 duration predictor 재학습 불필요

---

## 아키텍처 설계

### Phase 1: Pause Token 시스템

#### 1.1 Pause Token 정의

**새로운 특수 토큰 추가**:

```python
# 파일: GPT_SoVITS/text/symbols2.py
# 위치: 기존 _punctuation 정의 다음

_pause_tokens = [
    "<P_SHORT>",     # 100ms - 쉼표 1개
    "<P_MEDIUM>",    # 300ms - 쉼표 2개, 점 1개
    "<P_LONG>",      # 500ms - 점 2~3개
    "<P_VERYLONG>",  # 800ms - 말줄임표(...)
]

# symbols 리스트에 추가
symbols = [_pad] + list(_punctuation) + _pause_tokens + # ... 기존 심볼
```

**선택 이유**:
- 기존 phoneme symbol 체계와 호환
- G2P 파이프라인 변경 최소화
- 향후 SSML-style 제어로 확장 용이

#### 1.2 Text Preprocessing 수정

**파일**: `GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py`

```python
class TextPreprocessor:
    # 기존 메서드 교체
    def replace_consecutive_punctuation(self, text: str) -> str:
        """
        기존: 연속 구두점 축소
        신규: 구두점 → pause 토큰 변환
        """
        import re

        # 우선순위 순서로 처리 (긴 패턴 우선)
        replacements = [
            (r'\.{3,}', '<P_VERYLONG>'),     # ... → 800ms
            (r',{2,}', '<P_MEDIUM>'),        # ,, → 300ms
            (r'\.{2}', '<P_LONG>'),          # .. → 500ms
            (r'\.', '<P_MEDIUM>'),           # . → 300ms
            (r',', '<P_SHORT>'),             # , → 100ms
        ]

        for pattern, token in replacements:
            text = re.sub(pattern, token, text)

        return text

    def preprocess(self, text: str, lang: str, text_split_method: str, version: str):
        print(f"############ {i18n('切分文本')} ############")

        # ★ 구두점 → pause 토큰 변환 (기존 replace_consecutive_punctuation 호출 대체)
        text = self.replace_consecutive_punctuation(text)

        # ... 기존 로직 그대로 유지
        texts = self.pre_seg_text(text, lang, text_split_method)
        # ...
```

**변경 이유**:
- 기존 메서드 시그니처 유지 → 하위 호환성 보장
- 텍스트 전처리 단계에서만 수정 → 다른 컴포넌트 영향 없음

#### 1.3 Pause Token → Frame Index 매핑

**새 파일**: `GPT_SoVITS/TTS_infer_pack/pause_mapper.py`

```python
class PauseMapper:
    """Pause token을 mel frame index와 duration으로 매핑"""

    # Pause duration 정의 (초 단위)
    PAUSE_DURATIONS = {
        "<P_SHORT>": 0.1,      # 100ms
        "<P_MEDIUM>": 0.3,     # 300ms
        "<P_LONG>": 0.5,       # 500ms
        "<P_VERYLONG>": 0.8,   # 800ms
    }

    def __init__(self, mel_frame_rate: float = 93.75):
        """
        Args:
            mel_frame_rate: v4는 48kHz audio → 320 hop_size = 150 fps
                           실제로는 upsampling 고려하여 93.75 fps
        """
        self.mel_frame_rate = mel_frame_rate

    def extract_pause_info(
        self,
        phones: List[int],
        phone_to_symbol: Dict[int, str]
    ) -> List[Tuple[int, float]]:
        """
        Phoneme sequence에서 pause token 위치와 duration 추출

        Args:
            phones: Phoneme ID 리스트 [1, 45, 67, <P_LONG_ID>, 23, ...]
            phone_to_symbol: {phone_id: symbol_string} 매핑

        Returns:
            [(frame_index, duration_sec), ...]
            예: [(50, 0.5), (120, 0.3)] → frame 50에 0.5초, frame 120에 0.3초 pause
        """
        pause_info = []
        current_frame = 0

        for phone_id in phones:
            symbol = phone_to_symbol.get(phone_id, "")

            if symbol in self.PAUSE_DURATIONS:
                # Pause token 발견
                duration = self.PAUSE_DURATIONS[symbol]
                pause_info.append((current_frame, duration))

                # Pause 구간만큼 frame 증가
                pause_frames = int(duration * self.mel_frame_rate)
                current_frame += pause_frames
            else:
                # 일반 phoneme: 평균 duration 가정 (약 100ms)
                current_frame += int(0.1 * self.mel_frame_rate)

        return pause_info
```

**설계 근거**:
- Phoneme-to-frame 매핑은 근사치 사용 (실제는 duration predictor 출력 활용 가능)
- CFM에서 frame-level 제어를 위한 사전 계산

---

### Phase 2: CFM Inference 수정

#### 2.1 CFM 클래스 확장

**파일**: `GPT_SoVITS/module/models.py`

**수정 위치**: `CFM.inference()` 메서드 (라인 1027-1085)

```python
class CFM(torch.nn.Module):
    # ... 기존 코드 유지

    @torch.inference_mode()
    def inference(
        self,
        mu,
        x_lens,
        prompt,
        n_timesteps,
        temperature=1.0,
        inference_cfg_rate=0,
        pause_info: Optional[List[Tuple[int, float]]] = None  # ★ 새 파라미터
    ):
        """
        Args:
            pause_info: [(frame_idx, duration_sec), ...]
                       각 frame_idx 위치에 duration만큼 silence 주입
        """
        B, T = mu.size(0), mu.size(1)
        x = torch.randn([B, self.in_channels, T], device=mu.device, dtype=mu.dtype) * temperature
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x, dtype=mu.dtype)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        mu = mu.transpose(2, 1)

        # ★ Pause mask 생성
        pause_mask = self._create_pause_mask(x.shape, pause_info, mu.device) if pause_info else None

        t = 0
        d = 1 / n_timesteps
        # ... (기존 cache 변수들)

        for j in range(n_timesteps):
            t_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * t
            d_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * d

            # DiT velocity prediction
            v_pred, text_emb, dt = self.estimator(
                x, prompt_x, x_lens, t_tensor, d_tensor, mu,
                use_grad_ckpt=False, drop_audio_cond=False, drop_text=False,
                infer=True, text_cache=text_cache, dt_cache=dt_cache
            )
            v_pred = v_pred.transpose(2, 1)

            # ... (CFG 코드 유지)

            # ODE step
            x = x + d * v_pred

            # ★ Pause 영역에 silence 강제 주입
            if pause_mask is not None:
                x = x * (1 - pause_mask)  # pause 영역은 0으로 설정 (mel silence)

            t = t + d
            x[:, :, :prompt_len] = 0  # Prompt 부분은 항상 고정

        return x

    def _create_pause_mask(
        self,
        shape: Tuple[int, int, int],  # (B, C, T)
        pause_info: List[Tuple[int, float]],
        device: torch.device
    ) -> torch.Tensor:
        """
        Pause 위치에 해당하는 mask 생성

        Returns:
            mask: (B, C, T) tensor, pause 구간은 1.0, 나머지는 0.0
        """
        B, C, T = shape
        mask = torch.zeros(B, C, T, device=device)

        # Mel frame rate (v4: 48kHz / 320 hop = 150 fps)
        mel_frame_rate = 150.0

        for frame_idx, duration in pause_info:
            pause_frames = int(duration * mel_frame_rate)
            end_idx = min(frame_idx + pause_frames, T)

            # 해당 구간을 1로 설정
            mask[:, :, frame_idx:end_idx] = 1.0

        return mask
```

**핵심 원리**:

1. **ODE Integration과 Masking의 조화**:
   - CFM은 continuous normalizing flow → `x(t) = x(0) + ∫v(t)dt` 형태
   - 각 time step에서 `x`를 업데이트한 직후 pause 영역을 0으로 마스킹
   - Silence (mel-spectrogram에서 모든 bin이 0)를 강제

2. **왜 이 방법인가**:
   - **VITS 방식**: Duration predictor 재학습 필요 → 대규모 데이터셋 필요
   - **CFM 방식**: Inference-time만 수정 → 모델 재학습 불필요
   - DiT는 여전히 자연스러운 velocity를 예측하지만, 우리가 원하는 위치에 강제로 silence 주입

3. **Mel-Spectrogram에서 Silence의 의미**:
   - Mel bin 값이 모두 0 ≈ 에너지 없음 ≈ 무음
   - Vocoder는 이를 그대로 silence로 변환

#### 2.2 TTS 클래스 통합

**파일**: `GPT_SoVITS/TTS_infer_pack/TTS.py`

**수정 위치**: `using_vocoder_synthesis()` 메서드 (라인 1431-1494)

```python
class TTS:
    # ... 기존 코드

    def using_vocoder_synthesis(
        self,
        semantic_tokens: torch.Tensor,
        phones: torch.Tensor,
        speed: float = 1.0,
        sample_steps: int = 32
    ):
        # ... (기존 reference 처리 코드 유지: 라인 1434-1466)

        fea_todo, ge = self.vits_model.decode_encp(semantic_tokens, phones, refer_audio_spec, ge, speed)

        # ★ Pause 정보 추출
        from TTS_infer_pack.pause_mapper import PauseMapper
        pause_mapper = PauseMapper(mel_frame_rate=150.0)  # v4: 48kHz/320

        # phones tensor를 symbol로 역매핑
        from text.symbols2 import symbols
        phone_to_symbol = {i: sym for i, sym in enumerate(symbols)}
        pause_info = pause_mapper.extract_pause_info(
            phones.squeeze(0).cpu().tolist(),
            phone_to_symbol
        )

        cfm_resss = []
        idx = 0
        while 1:
            fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
            if fea_todo_chunk.shape[-1] == 0:
                break
            idx += chunk_len
            fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)

            # ★ pause_info를 chunk에 맞게 조정
            chunk_pause_info = [
                (frame_idx - idx + mel2.shape[2], duration)
                for frame_idx, duration in pause_info
                if idx - mel2.shape[2] <= frame_idx < idx + chunk_len
            ]

            # CFM inference with pause conditioning
            cfm_res = self.vits_model.cfm.inference(
                fea,
                torch.LongTensor([fea.size(1)]).to(fea.device),
                mel2,
                sample_steps,
                inference_cfg_rate=0,
                pause_info=chunk_pause_info  # ★ 전달
            )
            # ... (기존 후처리 유지)

        # ... (vocoder 처리 유지)
```

**Chunked Processing 고려**:
- v4는 메모리 효율을 위해 chunk 단위로 처리
- Pause 위치를 각 chunk의 local coordinate로 변환 필요
- `chunk_pause_info` 계산 시 chunk 시작 offset 고려

---

### Phase 3: Backward Compatibility

#### 3.1 기본값 처리

**전략**: `pause_info=None`일 때 기존 동작 유지

```python
# CFM.inference()에서
if pause_info is None:
    # 기존 동작: pause mask 적용 안 함
    # ... 기존 코드 그대로
```

#### 3.2 API 레벨 플래그

**파일**: `api_v2.py`

```python
class TTS_Request(BaseModel):
    # ... 기존 필드
    enable_pause_control: bool = False  # ★ 새 필드 (기본값 False)
    pause_duration_scale: float = 1.0   # ★ Pause 길이 스케일 (조정 가능)
```

**사용 예시**:
```json
{
    "text": "나는 말야... 조심스럽지만,, 괜찮아.",
    "enable_pause_control": true,
    "pause_duration_scale": 1.2  // 20% 더 긴 pause
}
```

---

## 구현 순서

### Step 1: Symbol 및 Preprocessing
1. `text/symbols2.py`에 `_pause_tokens` 추가
2. `TextPreprocessor.py`의 `replace_consecutive_punctuation()` 수정
3. 단위 테스트: "나는..." → "나는<P_VERYLONG>" 변환 확인

### Step 2: Pause Mapper
1. `TTS_infer_pack/pause_mapper.py` 신규 작성
2. `extract_pause_info()` 로직 구현
3. 단위 테스트: phoneme sequence → frame index 매핑 검증

### Step 3: CFM Modification
1. `module/models.py`의 `CFM.inference()` 수정
   - `pause_info` 파라미터 추가
   - `_create_pause_mask()` 구현
   - ODE loop 내 masking 추가
2. 통합 테스트: 간단한 텍스트로 pause 적용 확인

### Step 4: TTS Integration
1. `TTS_infer_pack/TTS.py`의 `using_vocoder_synthesis()` 수정
2. Pause info 추출 및 CFM 전달 로직
3. Chunked processing 대응

### Step 5: API & Testing
1. `api_v2.py`에 `enable_pause_control` 필드 추가
2. End-to-end 테스트
3. 다양한 구두점 패턴 검증

---

## 검증 방법

### 단위 테스트

```python
# test_pause_preprocessing.py
def test_pause_token_conversion():
    preprocessor = TextPreprocessor(...)

    # Case 1: 말줄임표
    text = "나는 말야... 조심스럽지만"
    result = preprocessor.replace_consecutive_punctuation(text)
    assert "<P_VERYLONG>" in result

    # Case 2: 연속 쉼표
    text = "그게,, 뭐랄까,, 어려워"
    result = preprocessor.replace_consecutive_punctuation(text)
    assert result.count("<P_MEDIUM>") == 2
```

### 통합 테스트

**테스트 케이스**:
1. **짧은 pause**: "안녕, 반가워" → 0.1초 pause 확인
2. **긴 pause**: "안녕... 반가워" → 0.8초 pause 확인
3. **혼합 패턴**: "나는,, 말야... 괜찮아." → 다양한 pause 길이

**검증 지표**:
- Mel-spectrogram 시각화: pause 구간이 0에 가까운지
- Waveform 분석: silence 구간 길이 측정
- 주관적 청취 테스트: 자연스러운지

---

## 한계점 및 향후 개선

### 현재 방법의 한계

1. **Pause의 자연스러움**:
   - 강제 마스킹은 hard silence 생성
   - 실제 인간 발화는 soft silence (약한 호흡음 등)
   - **개선 방향**: DiT에 pause token을 condition으로 학습 (모델 재학습)

2. **Phone-to-Frame 매핑 정확도**:
   - 현재는 평균 duration 가정 (100ms/phoneme)
   - 실제 duration은 phoneme마다 다름
   - **개선 방향**: Duration predictor 출력 활용 또는 forced alignment

3. **Chunk 경계 처리**:
   - Pause가 chunk 경계에 걸칠 경우 부자연스러울 수 있음
   - **개선 방향**: Chunk split 시 pause 위치 고려

### 왜 이 방법을 선택했나

**장점**:
- ✅ 모델 재학습 불필요 → 빠른 구현
- ✅ Inference-time만 수정 → 기존 모델 재사용
- ✅ v4의 CFM 아키텍처를 최대한 활용
- ✅ 사용자가 직관적으로 제어 가능 (구두점으로)

**단점 수용 이유**:
- Hard silence도 충분히 자연스러움 (기존 TTS 대비)
- Phone-frame 매핑 오차는 100ms 내외로 청각적으로 인지 어려움
- 향후 점진적 개선 가능한 구조

---

## 참고 자료

### 코드 위치
- **CFM 구현**: `GPT_SoVITS/module/models.py:1013-1112`
- **DiT Backbone**: `GPT_SoVITS/f5_tts/model/backbones/dit.py`
- **Text Preprocessing**: `GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py`
- **TTS Pipeline**: `GPT_SoVITS/TTS_infer_pack/TTS.py:1431-1494`

### 관련 논문
- Conditional Flow Matching (CFM): "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- F5-TTS: "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching" (Chen et al., 2024)
