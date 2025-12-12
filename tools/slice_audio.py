import os
import sys
import shutil
import numpy as np
import traceback
from scipy.io import wavfile

# parent_directory = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(parent_directory)
from tools.my_utils import load_audio
from slicer2 import Slicer

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma', '.aac'}


def find_matching_lab_file(audio_path: str) -> str | None:
    """Find matching .lab file for audio file."""
    base_path = os.path.splitext(audio_path)[0]
    for ext in ['.lab', '.LAB', '.Lab']:
        lab_path = f"{base_path}{ext}"
        if os.path.exists(lab_path):
            return lab_path
    return None


def is_audio_file(filename: str) -> bool:
    """Check if file is an audio file by extension."""
    _, ext = os.path.splitext(filename.lower())
    return ext in AUDIO_EXTENSIONS


def slice(inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, i_part, all_part):
    os.makedirs(opt_root, exist_ok=True)
    if os.path.isfile(inp):
        input = [inp]
    elif os.path.isdir(inp):
        # Filter: only audio files
        input = [
            os.path.join(inp, name)
            for name in sorted(os.listdir(inp))
            if is_audio_file(name)
        ]
    else:
        return "输入路径存在但既不是文件也不是文件夹"

    slicer = Slicer(
        sr=32000,
        threshold=int(threshold),
        min_length=int(min_length),
        min_interval=int(min_interval),
        hop_size=int(hop_size),
        max_sil_kept=int(max_sil_kept),
    )
    _max = float(_max)
    alpha = float(alpha)

    skipped_count = 0
    sliced_count = 0

    for inp_path in input[int(i_part) :: int(all_part)]:
        try:
            name = os.path.basename(inp_path)

            # Check for matching .lab file - skip slicing if exists
            lab_path = find_matching_lab_file(inp_path)
            if lab_path:
                # Copy audio and lab file directly (already segmented)
                shutil.copy2(inp_path, os.path.join(opt_root, name))
                shutil.copy2(lab_path, os.path.join(opt_root, os.path.basename(lab_path)))
                skipped_count += 1
                continue

            # No .lab file - run normal slicing
            audio = load_audio(inp_path, 32000)
            for chunk, start, end in slicer.slice(audio):
                tmp_max = np.abs(chunk).max()
                if tmp_max > 1:
                    chunk /= tmp_max
                chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
                wavfile.write(
                    "%s/%s_%010d_%010d.wav" % (opt_root, name, start, end),
                    32000,
                    (chunk * 32767).astype(np.int16),
                )
            sliced_count += 1
        except:
            print(inp_path, "->fail->", traceback.format_exc())

    if skipped_count > 0:
        print(f"[LAB] {skipped_count}개 파일 슬라이싱 스킵 (.lab 파일 존재)")
    if sliced_count > 0:
        print(f"[SLICE] {sliced_count}개 파일 슬라이싱 완료")

    return "执行完毕，请检查输出文件"


print(slice(*sys.argv[1:]))
