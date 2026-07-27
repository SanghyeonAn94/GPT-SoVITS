"""Japanese g2p. Modified from https://github.com/CjangCjengh/vits/blob/main/text/japanese.py

pyopenjtalk_g2p_prosody and _numeric_feature_by_regex are copied from espnet
https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
"""

import re
import os
import hashlib

try:
    import pyopenjtalk

    current_file_path = os.path.dirname(__file__)

    if os.name == "nt":
        python_dir = os.getcwd()
        OPEN_JTALK_DICT_DIR = pyopenjtalk.OPEN_JTALK_DICT_DIR.decode("utf-8")
        if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", OPEN_JTALK_DICT_DIR)):
            if OPEN_JTALK_DICT_DIR[: len(python_dir)].upper() == python_dir.upper():
                OPEN_JTALK_DICT_DIR = os.path.join(os.path.relpath(OPEN_JTALK_DICT_DIR, python_dir))
            else:
                import shutil

                if not os.path.exists("TEMP"):
                    os.mkdir("TEMP")
                if not os.path.exists(os.path.join("TEMP", "ja")):
                    os.mkdir(os.path.join("TEMP", "ja"))
                if os.path.exists(os.path.join("TEMP", "ja", "open_jtalk_dic")):
                    shutil.rmtree(os.path.join("TEMP", "ja", "open_jtalk_dic"))
                shutil.copytree(
                    pyopenjtalk.OPEN_JTALK_DICT_DIR.decode("utf-8"),
                    os.path.join("TEMP", "ja", "open_jtalk_dic"),
                )
                OPEN_JTALK_DICT_DIR = os.path.join("TEMP", "ja", "open_jtalk_dic")
            pyopenjtalk.OPEN_JTALK_DICT_DIR = OPEN_JTALK_DICT_DIR.encode("utf-8")

        if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", current_file_path)):
            if current_file_path[: len(python_dir)].upper() == python_dir.upper():
                current_file_path = os.path.join(os.path.relpath(current_file_path, python_dir))
            else:
                if not os.path.exists("TEMP"):
                    os.mkdir("TEMP")
                if not os.path.exists(os.path.join("TEMP", "ja")):
                    os.mkdir(os.path.join("TEMP", "ja"))
                if not os.path.exists(os.path.join("TEMP", "ja", "ja_userdic")):
                    os.mkdir(os.path.join("TEMP", "ja", "ja_userdic"))
                    shutil.copyfile(
                        os.path.join(current_file_path, "ja_userdic", "userdict.csv"),
                        os.path.join("TEMP", "ja", "ja_userdic", "userdict.csv"),
                    )
                current_file_path = os.path.join("TEMP", "ja")

    def get_hash(fp: str) -> str:
        hash_md5 = hashlib.md5()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    USERDIC_CSV_PATH = os.path.join(current_file_path, "ja_userdic", "userdict.csv")
    USERDIC_BIN_PATH = os.path.join(current_file_path, "ja_userdic", "user.dict")
    USERDIC_HASH_PATH = os.path.join(current_file_path, "ja_userdic", "userdict.md5")
    if os.path.exists(USERDIC_CSV_PATH):
        if (
            not os.path.exists(USERDIC_BIN_PATH)
            or get_hash(USERDIC_CSV_PATH) != open(USERDIC_HASH_PATH, "r", encoding="utf-8").read()
        ):
            pyopenjtalk.mecab_dict_index(USERDIC_CSV_PATH, USERDIC_BIN_PATH)
            with open(USERDIC_HASH_PATH, "w", encoding="utf-8") as f:
                f.write(get_hash(USERDIC_CSV_PATH))

    if os.path.exists(USERDIC_BIN_PATH):
        pyopenjtalk.update_global_jtalk_with_user_dict(USERDIC_BIN_PATH)
except Exception:
    import pyopenjtalk

    pass


from text.symbols import punctuation

_japanese_characters = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

_japanese_marks = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

_symbols_to_japanese = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]


_real_sokuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"Q([↑↓]*[kg])", r"k#\1"),
        (r"Q([↑↓]*[tdjʧ])", r"t#\1"),
        (r"Q([↑↓]*[sʃ])", r"s\1"),
        (r"Q([↑↓]*[pb])", r"p#\1"),
    ]
]

_real_hatsuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"N([↑↓]*[pbm])", r"m\1"),
        (r"N([↑↓]*[ʧʥj])", r"n^\1"),
        (r"N([↑↓]*[tdn])", r"n\1"),
        (r"N([↑↓]*[kg])", r"ŋ\1"),
    ]
]


def post_replace_ph(ph):
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
    }

    if ph in rep_map.keys():
        ph = rep_map[ph]
    return ph


def replace_consecutive_punctuation(text):
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def symbols_to_japanese(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text


def preprocess_jap(text, with_prosody=False):
    text = symbols_to_japanese(text)
    text = text.lower()
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)
    text = []
    for i, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            if with_prosody:
                text += pyopenjtalk_g2p_prosody(sentence)[1:-1]
            else:
                p = pyopenjtalk.g2p(sentence)
                text += p.split(" ")

        if i < len(marks):
            if marks[i] == " ":
                continue
            text += [marks[i].replace(" ", "")]
    return text


def text_normalize(text):
    text = replace_consecutive_punctuation(text)
    return text


def pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True):
    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones


def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def g2p(norm_text, with_prosody=True):
    phones = preprocess_jap(norm_text, with_prosody)
    phones = [post_replace_ph(i) for i in phones]
    return phones


if __name__ == "__main__":
    phones = g2p("Hello.こんにちは！今日もNiCe天気ですね！tokyotowerに行きましょう！")
    print(phones)
