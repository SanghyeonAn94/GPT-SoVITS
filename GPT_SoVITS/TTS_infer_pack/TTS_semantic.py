import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from tools.i18n.i18n import i18n
from tqdm import tqdm

class TTSSemantic:
    def __init__(self, parent_tts):
        self.parent = parent_tts
        self.last_generated_semantic_data = None

    def generate_semantic_tokens(self, inputs: dict) -> dict:
        self.parent.stop_flag = False
        text = inputs.get("text", "")
        text_lang = inputs.get("text_lang", "")
        ref_audio_path = inputs.get("ref_audio_path", "")
        aux_ref_audio_paths = inputs.get("aux_ref_audio_paths", [])
        prompt_text = inputs.get("prompt_text", "")
        prompt_lang = inputs.get("prompt_lang", "")
        top_k = inputs.get("top_k", 5)
        top_p = inputs.get("top_p", 1)
        temperature = inputs.get("temperature", 1)
        text_split_method = inputs.get("text_split_method", "cut0")
        auto_adjust_sampling = inputs.get("auto_adjust_sampling", True)
        batch_size = inputs.get("batch_size", 1)
        batch_threshold = inputs.get("batch_threshold", 0.75)
        speed_factor = inputs.get("speed_factor", 1.0)
        split_bucket = inputs.get("split_bucket", True)
        return_fragment = inputs.get("return_fragment", False)
        fragment_interval = inputs.get("fragment_interval", 0.3)
        seed = inputs.get("seed", -1)
        seed = -1 if seed in ["", None] else seed
        from GPT_SoVITS.TTS_infer_pack.TTS import set_seed
        actual_seed = set_seed(seed)
        parallel_infer = inputs.get("parallel_infer", True)
        repetition_penalty = inputs.get("repetition_penalty", 1.35)
        sample_steps = inputs.get("sample_steps", 32)
        n_samples = inputs.get("n_samples", 1)

        if parallel_infer:
            print(i18n("并行推理模式已开启"))
            self.parent.t2s_model.model.infer_panel = self.parent.t2s_model.model.infer_panel_batch_infer
        else:
            print(i18n("并行推理模式已关闭"))
            self.parent.t2s_model.model.infer_panel = self.parent.t2s_model.model.infer_panel_naive_batched

        if return_fragment:
            print(i18n("分段返回模式已开启"))
            if split_bucket:
                split_bucket = False
                print(i18n("分段返回模式不支持分桶处理，已自动关闭分桶处理"))

        if split_bucket and speed_factor == 1.0 and not (self.parent.configs.use_vocoder and parallel_infer):
            print(i18n("分桶处理模式已开启"))
        elif speed_factor != 1.0:
            print(i18n("语速调节不支持分桶处理，已自动关闭分桶处理"))
            split_bucket = False
        elif self.parent.configs.use_vocoder and parallel_infer:
            print(i18n("当开启并行推理模式时，SoVits V3/4模型不支持分桶处理，已自动关闭分桶处理"))
            split_bucket = False
        else:
            print(i18n("分桶处理模式已关闭"))

        if fragment_interval < 0.01:
            fragment_interval = 0.01
            print(i18n("分段间隔过小，已自动设置为0.01"))

        no_prompt_text = False
        if prompt_text in [None, ""]:
            no_prompt_text = True

        from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import splits
        assert text_lang in self.parent.configs.languages
        if not no_prompt_text:
            assert prompt_lang in self.parent.configs.languages

        if no_prompt_text and self.parent.configs.use_vocoder:
            from GPT_SoVITS.TTS_infer_pack.TTS import NO_PROMPT_ERROR
            raise NO_PROMPT_ERROR("prompt_text cannot be empty when using SoVITS_V3")

        if ref_audio_path in [None, ""] and (
            (self.parent.prompt_cache["prompt_semantic"] is None) or (self.parent.prompt_cache["refer_spec"] in [None, []])
        ):
            raise ValueError(
                "ref_audio_path cannot be empty, when the reference audio is not set using set_ref_audio()"
            )

        import os
        t0 = time.perf_counter()
        if (ref_audio_path is not None) and (
            ref_audio_path != self.parent.prompt_cache["ref_audio_path"]
            or (self.parent.is_v2pro and self.parent.prompt_cache["refer_spec"][0][1] is None)
        ):
            if not os.path.exists(ref_audio_path):
                raise ValueError(f"{ref_audio_path} not exists")
            self.parent.set_ref_audio(ref_audio_path)

        aux_ref_audio_paths = aux_ref_audio_paths if aux_ref_audio_paths is not None else []
        paths = set(aux_ref_audio_paths) & set(self.parent.prompt_cache["aux_ref_audio_paths"])
        if not (len(list(paths)) == len(aux_ref_audio_paths) == len(self.parent.prompt_cache["aux_ref_audio_paths"])):
            self.parent.prompt_cache["aux_ref_audio_paths"] = aux_ref_audio_paths
            self.parent.prompt_cache["refer_spec"] = [self.parent.prompt_cache["refer_spec"][0]]
            for path in aux_ref_audio_paths:
                if path in [None, ""]:
                    continue
                if not os.path.exists(path):
                    print(i18n("音频文件不存在，跳过："), path)
                    continue
                self.parent.prompt_cache["refer_spec"].append(self.parent._get_ref_spec(path))

        if not no_prompt_text:
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in splits:
                prompt_text += "。" if prompt_lang != "en" else "."
            print(i18n("实际输入的参考文本:"), prompt_text)
            if self.parent.prompt_cache["prompt_text"] != prompt_text:
                phones, bert_features, norm_text = self.parent.text_preprocessor.segment_and_extract_feature_for_text(
                    prompt_text, prompt_lang, self.parent.configs.version
                )
                self.parent.prompt_cache["prompt_text"] = prompt_text
                self.parent.prompt_cache["prompt_lang"] = prompt_lang
                self.parent.prompt_cache["phones"] = phones
                self.parent.prompt_cache["bert_features"] = bert_features
                self.parent.prompt_cache["norm_text"] = norm_text

        t1 = time.perf_counter()
        data = None
        if not return_fragment:
            data = self.parent.text_preprocessor.preprocess(text, text_lang, text_split_method, self.parent.configs.version)
            if len(data) == 0:
                return {
                    "semantic_data": None,
                    "empty_result": True
                }

            batch_index_list = None
            data, batch_index_list = self.parent.to_batch(
                data,
                prompt_data=self.parent.prompt_cache if not no_prompt_text else None,
                batch_size=batch_size,
                threshold=batch_threshold,
                split_bucket=split_bucket,
                device=self.parent.configs.device,
                precision=self.parent.precision,
            )
        else:
            print(f"############ {i18n('切分文本')} ############")
            texts = self.parent.text_preprocessor.pre_seg_text(text, text_lang, text_split_method)
            data = []
            for i in range(len(texts)):
                if i % batch_size == 0:
                    data.append([])
                data[-1].append(texts[i])

        t2 = time.perf_counter()

        semantic_results = []

        for item in data:
            if return_fragment:
                batch_data = []
                print(f"############ {i18n('提取文本Bert特征')} ############")
                for text in tqdm(item):
                    phones, bert_features, norm_text = self.parent.text_preprocessor.segment_and_extract_feature_for_text(
                        text, text_lang, self.parent.configs.version
                    )
                    if phones is None:
                        continue
                    res = {
                        "phones": phones,
                        "bert_features": bert_features,
                        "norm_text": norm_text,
                    }
                    batch_data.append(res)
                if len(batch_data) == 0:
                    continue
                batch, _ = self.parent.to_batch(
                    batch_data,
                    prompt_data=self.parent.prompt_cache if not no_prompt_text else None,
                    batch_size=batch_size,
                    threshold=batch_threshold,
                    split_bucket=False,
                    device=self.parent.configs.device,
                    precision=self.parent.precision,
                )
                item = batch[0]

            batch_phones = item["phones"]
            batch_phones_len = item["phones_len"]
            all_phoneme_ids = item["all_phones"]
            all_phoneme_lens = item["all_phones_len"]
            all_bert_features = item["all_bert_features"]
            norm_text = item["norm_text"]
            max_len = item["max_len"]

            print(i18n("前端处理后的文本(每句):"), norm_text)
            if no_prompt_text:
                prompt = None
            else:
                prompt = (
                    self.parent.prompt_cache["prompt_semantic"].expand(len(all_phoneme_ids), -1).to(self.parent.configs.device)
                )

            adjusted_top_k = top_k
            adjusted_top_p = top_p
            adjusted_temperature = temperature

            if auto_adjust_sampling:
                text_len = len(norm_text) if isinstance(norm_text, str) else len(norm_text[0])
                if text_len <= 5:
                    adjusted_top_k = max(top_k, 20)
                    adjusted_top_p = max(top_p, 1.0)
                    adjusted_temperature = max(temperature, 1.0)

            print(f"############ {i18n('预测语义Token')} (n_samples={n_samples}) ############")
            all_pred_semantic_list = []
            all_idx_list = []
            all_batch_phones = []

            for sample_idx in range(n_samples):
                print(f"  Sample {sample_idx + 1}/{n_samples}...")
                pred_semantic_list, idx_list = self.parent.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_lens,
                    prompt,
                    all_bert_features,
                    top_k=adjusted_top_k,
                    top_p=adjusted_top_p,
                    temperature=adjusted_temperature,
                    early_stop_num=self.parent.configs.hz * self.parent.configs.max_sec,
                    max_len=max_len,
                    repetition_penalty=repetition_penalty,
                )
                all_pred_semantic_list.extend(pred_semantic_list)
                all_idx_list.extend(idx_list)
                all_batch_phones.extend(batch_phones)

            semantic_results.append({
                "pred_semantic_list": all_pred_semantic_list,
                "idx_list": all_idx_list,
                "batch_phones": all_batch_phones
            })

        result = {
            "semantic_data": semantic_results,
            "metadata": {
                "batch_index_list": batch_index_list if not return_fragment else None,
                "split_bucket": split_bucket,
                "return_fragment": return_fragment,
                "fragment_interval": fragment_interval,
                "speed_factor": speed_factor,
                "sample_steps": sample_steps,
                "n_samples": n_samples,
                "no_prompt_text": no_prompt_text,
                "parallel_infer": parallel_infer,
                "t0": t0,
                "t1": t1,
                "t2": t2
            },
            "empty_result": False
        }

        self.last_generated_semantic_data = result

        return result