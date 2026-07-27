"""Small helpers: bool parsing, newest-checkpoint selection, and non-empty text-file check."""

import re


def str2bool(str):
    return True if str.lower() == "true" else False


def get_newest_ckpt(string_list):
    pattern = r"epoch=(\d+)-step=(\d+)\.ckpt"

    extracted_info = []
    for string in string_list:
        match = re.match(pattern, string)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            extracted_info.append((epoch, step, string))
    sorted_info = sorted(extracted_info, key=lambda x: (x[0], x[1]), reverse=True)
    newest_ckpt = sorted_info[0][2]
    return newest_ckpt


def check_txt_file(file_path):
    try:
        with open(file_path, "r") as file:
            text = file.readline().strip()
        assert text.strip() != ""
        return text
    except Exception:
        return False
    return False
