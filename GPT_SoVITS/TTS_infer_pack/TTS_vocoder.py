import torch
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from tools.i18n.i18n import i18n
from tqdm import tqdm
import torch.nn.functional as F

class TTSVocoder:
    def __init__(self, parent_tts):
        self.parent = parent_tts

    def synthesis_from_semantic(self, semantic_tokens: dict, super_sampling: bool = False) -> List[Tuple[int, np.ndarray]]:
        if semantic_tokens.get("empty_result"):
            return [(16000, np.zeros(int(16000), dtype=np.int16))]
        semantic_results = semantic_tokens.get("semantic_data") or semantic_tokens.get("semantic_tokens")
        if semantic_results is None:
            raise ValueError("Neither 'semantic_data' nor 'semantic_tokens' found in semantic_tokens dict")

        metadata = semantic_tokens["metadata"]

        batch_index_list = metadata.get("batch_index_list")
        split_bucket = metadata.get("split_bucket")
        return_fragment = metadata.get("return_fragment")
        fragment_interval = metadata.get("fragment_interval")
        speed_factor = metadata.get("speed_factor")
        sample_steps = metadata.get("sample_steps")
        n_samples = metadata.get("n_samples")
        parallel_infer = metadata.get("parallel_infer")

        audio = []
        output_sr = self.parent.configs.sampling_rate if not self.parent.configs.use_vocoder else self.parent.vocoder_configs["sr"]

        t_34 = 0.0
        t_45 = 0.0

        for semantic_result in semantic_results:
            t3 = time.perf_counter()

            pred_semantic_list = semantic_result["pred_semantic_list"]
            idx_list = semantic_result["idx_list"]
            batch_phones = semantic_result["batch_phones"]

            t4 = time.perf_counter()
            t_34 += t4 - t3

            refer_audio_spec = []
            if self.parent.is_v2pro:
                sv_emb = []
            for spec, audio_tensor in self.parent.prompt_cache["refer_spec"]:
                spec = spec.to(dtype=self.parent.precision, device=self.parent.configs.device)
                refer_audio_spec.append(spec)
                if self.parent.is_v2pro:
                    sv_emb.append(self.parent.sv_model.compute_embedding3(audio_tensor))

            batch_audio_fragment = []

            print(f"############ {i18n('合成音频')} ############")
            if not self.parent.configs.use_vocoder:
                if speed_factor == 1.0:
                    print(f"{i18n('并行合成中')}...")
                    pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
                    upsample_rate = math.prod(self.parent.vits_model.upsample_rates)
                    audio_frag_idx = [
                        pred_semantic_list[i].shape[0] * 2 * upsample_rate
                        for i in range(0, len(pred_semantic_list))
                    ]
                    audio_frag_end_idx = [sum(audio_frag_idx[: i + 1]) for i in range(0, len(audio_frag_idx))]
                    all_pred_semantic = (
                        torch.cat(pred_semantic_list).unsqueeze(0).unsqueeze(0).to(self.parent.configs.device)
                    )
                    _batch_phones = torch.cat(batch_phones).unsqueeze(0).to(self.parent.configs.device)
                    if self.parent.is_v2pro != True:
                        _batch_audio_fragment = self.parent.vits_model.decode(
                            all_pred_semantic, _batch_phones, refer_audio_spec, speed=speed_factor
                        ).detach()[0, 0, :]
                    else:
                        _batch_audio_fragment = self.parent.vits_model.decode(
                            all_pred_semantic, _batch_phones, refer_audio_spec, speed=speed_factor, sv_emb=sv_emb
                        ).detach()[0, 0, :]
                    audio_frag_end_idx.insert(0, 0)
                    batch_audio_fragment = [
                        _batch_audio_fragment[audio_frag_end_idx[i - 1] : audio_frag_end_idx[i]]
                        for i in range(1, len(audio_frag_end_idx))
                    ]

                    if n_samples > 1:
                        num_text_segments = len(batch_audio_fragment) // n_samples
                        all_samples_audio_fragments = []
                        for sample_idx in range(n_samples):
                            sample_fragments = []
                            for text_idx in range(num_text_segments):
                                idx = text_idx * n_samples + sample_idx
                                sample_fragments.append(batch_audio_fragment[idx])
                            all_samples_audio_fragments.append(sample_fragments)
                    else:
                        all_samples_audio_fragments = [batch_audio_fragment]
                else:
                    for i, idx in enumerate(tqdm(idx_list)):
                        phones = batch_phones[i].unsqueeze(0).to(self.parent.configs.device)
                        _pred_semantic = (
                            pred_semantic_list[i][-idx:].unsqueeze(0).unsqueeze(0)
                        )
                        if self.parent.is_v2pro != True:
                            audio_fragment = self.parent.vits_model.decode(
                                _pred_semantic, phones, refer_audio_spec, speed=speed_factor
                            ).detach()[0, 0, :]
                        else:
                            audio_fragment = self.parent.vits_model.decode(
                                _pred_semantic, phones, refer_audio_spec, speed=speed_factor, sv_emb=sv_emb
                            ).detach()[0, 0, :]
                        batch_audio_fragment.append(audio_fragment)

                    if n_samples > 1:
                        num_text_segments = len(batch_audio_fragment) // n_samples
                        all_samples_audio_fragments = []
                        for sample_idx in range(n_samples):
                            sample_fragments = []
                            for text_idx in range(num_text_segments):
                                idx = text_idx * n_samples + sample_idx
                                sample_fragments.append(batch_audio_fragment[idx])
                            all_samples_audio_fragments.append(sample_fragments)
                    else:
                        all_samples_audio_fragments = [batch_audio_fragment]
            else:
                if parallel_infer:
                    print(f"{i18n('并行合成中')}...")
                    audio_fragments = self.parent.using_vocoder_synthesis_batched_infer(
                        idx_list, pred_semantic_list, batch_phones, speed=speed_factor, sample_steps=sample_steps
                    )
                    batch_audio_fragment.extend(audio_fragments)

                    if n_samples > 1:
                        num_text_segments = len(batch_audio_fragment) // n_samples
                        all_samples_audio_fragments = []
                        for sample_idx in range(n_samples):
                            sample_fragments = []
                            for text_idx in range(num_text_segments):
                                idx = text_idx * n_samples + sample_idx
                                sample_fragments.append(batch_audio_fragment[idx])
                            all_samples_audio_fragments.append(sample_fragments)
                    else:
                        all_samples_audio_fragments = [batch_audio_fragment]
                else:
                    for i, idx in enumerate(tqdm(idx_list)):
                        phones = batch_phones[i].unsqueeze(0).to(self.parent.configs.device)
                        _pred_semantic = (
                            pred_semantic_list[i][-idx:].unsqueeze(0).unsqueeze(0)
                        )
                        audio_fragment = self.parent.using_vocoder_synthesis(
                            _pred_semantic, phones, speed=speed_factor, sample_steps=sample_steps
                        )
                        batch_audio_fragment.append(audio_fragment)

                    if n_samples > 1:
                        num_text_segments = len(batch_audio_fragment) // n_samples
                        all_samples_audio_fragments = []
                        for sample_idx in range(n_samples):
                            sample_fragments = []
                            for text_idx in range(num_text_segments):
                                idx = text_idx * n_samples + sample_idx
                                sample_fragments.append(batch_audio_fragment[idx])
                            all_samples_audio_fragments.append(sample_fragments)
                    else:
                        all_samples_audio_fragments = [batch_audio_fragment]

            t5 = time.perf_counter()
            t_45 += t5 - t4

            if return_fragment:
                t0 = metadata.get("t0")
                t1 = metadata.get("t1")
                t2 = metadata.get("t2")
                print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t4 - t3, t5 - t4))
                results = []
                for sample_audio_fragments in all_samples_audio_fragments:
                    result = self.parent.audio_postprocess(
                        [sample_audio_fragments],
                        output_sr,
                        None,
                        speed_factor,
                        False,
                        fragment_interval,
                        super_sampling if self.parent.configs.use_vocoder and self.parent.configs.version == "v3" else False,
                    )
                    results.append(result)
                return results
            else:
                audio.append(all_samples_audio_fragments)

            if self.parent.stop_flag:
                return [(16000, np.zeros(int(16000), dtype=np.int16))]

        if not return_fragment:
            t0 = metadata.get("t0")
            t1 = metadata.get("t1")
            t2 = metadata.get("t2")
            print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t_34, t_45))
            if len(audio) == 0:
                return [(16000, np.zeros(int(16000), dtype=np.int16))]

            all_samples = [sample_fragments for item in audio for sample_fragments in item]

            results = []
            for sample_audio_fragments in all_samples:
                result = self.parent.audio_postprocess(
                    [sample_audio_fragments],
                    output_sr,
                    batch_index_list,
                    speed_factor,
                    split_bucket,
                    fragment_interval,
                    super_sampling if self.parent.configs.use_vocoder and self.parent.configs.version == "v3" else False,
                )
                results.append(result)
            return results

    def synthesis_from_mel(self, mel_spectrogram: torch.Tensor) -> Tuple[int, np.ndarray]:
        with torch.inference_mode():
            wav_gen = self.parent.vocoder(mel_spectrogram)
            audio = wav_gen[0][0]

        output_sr = self.parent.vocoder_configs["sr"] if self.parent.configs.use_vocoder else self.parent.configs.sampling_rate
        audio_np = audio.cpu().numpy()
        audio_np = (audio_np * 32768).astype(np.int16)

        return output_sr, audio_np