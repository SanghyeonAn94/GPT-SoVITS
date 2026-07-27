import pickle
import os
import re
import wordsegment
from g2p_en import G2p

from text.symbols import punctuation

from text.symbols2 import symbols

from builtins import str as unicode
from text.en_normalization.expend import normalize
from nltk.tokenize import TweetTokenizer

word_tokenize = TweetTokenizer().tokenize
from nltk import pos_tag

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CMU_DICT_FAST_PATH = os.path.join(current_file_path, "cmudict-fast.rep")
CMU_DICT_HOT_PATH = os.path.join(current_file_path, "engdict-hot.rep")
CACHE_PATH = os.path.join(current_file_path, "engdict_cache.pickle")
NAMECACHE_PATH = os.path.join(current_file_path, "namedict_cache.pickle")


rep_map = {
    "[;:：，；]": ",",
    '["’]': "'",
    "。": ".",
    "！": "!",
    "？": "?",
}


arpa = {
    "AH0",
    "S",
    "AH1",
    "EY2",
    "AE2",
    "EH0",
    "OW2",
    "UH0",
    "NG",
    "B",
    "G",
    "AY0",
    "M",
    "AA0",
    "F",
    "AO0",
    "ER2",
    "UH1",
    "IY1",
    "AH2",
    "DH",
    "IY0",
    "EY1",
    "IH0",
    "K",
    "N",
    "W",
    "IY2",
    "T",
    "AA1",
    "ER1",
    "EH2",
    "OY0",
    "UH2",
    "UW1",
    "Z",
    "AW2",
    "AW1",
    "V",
    "UW2",
    "AA2",
    "ER",
    "AW0",
    "UW0",
    "R",
    "OW1",
    "EH1",
    "ZH",
    "AE0",
    "IH2",
    "IH",
    "Y",
    "JH",
    "P",
    "AY1",
    "EY0",
    "OY2",
    "TH",
    "HH",
    "D",
    "ER0",
    "CH",
    "AO1",
    "AE1",
    "AO2",
    "OY1",
    "AY2",
    "IH1",
    "OW0",
    "L",
    "SH",
}


def replace_phs(phs):
    rep_map = {"'": "-"}
    phs_new = []
    for ph in phs:
        if ph in symbols:
            phs_new.append(ph)
        elif ph in rep_map.keys():
            phs_new.append(rep_map[ph])
        else:
            print("ph not in symbols: ", ph)
    return phs_new


def replace_consecutive_punctuation(text):
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}\s])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0].lower()

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def read_dict_new():
    g2p_dict = {}
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 57:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0].lower()
                g2p_dict[word] = [word_split[1].split(" ")]

            line_index = line_index + 1
            line = f.readline()

    with open(CMU_DICT_FAST_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0].lower()
                if word not in g2p_dict:
                    g2p_dict[word] = [word_split[1:]]

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def hot_reload_hot(g2p_dict):
    with open(CMU_DICT_HOT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0].lower()
                g2p_dict[word] = [word_split[1:]]

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict_new()
        cache_dict(g2p_dict, CACHE_PATH)

    g2p_dict = hot_reload_hot(g2p_dict)

    return g2p_dict


def get_namedict():
    if os.path.exists(NAMECACHE_PATH):
        with open(NAMECACHE_PATH, "rb") as pickle_file:
            name_dict = pickle.load(pickle_file)
    else:
        name_dict = {}

    return name_dict


def text_normalize(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    text = pattern.sub(lambda x: rep_map[x.group()], text)

    text = unicode(text)
    text = normalize(text)

    text = replace_consecutive_punctuation(text)
    return text


class en_G2p(G2p):
    def __init__(self):
        super().__init__()
        wordsegment.load()

        self.cmu = get_dict()
        self.namedict = get_namedict()

        for word in ["AE", "AI", "AR", "IOS", "HUD", "OS"]:
            del self.cmu[word.lower()]

        self.homograph2features["read"] = (["R", "IY1", "D"], ["R", "EH1", "D"], "VBP")
        self.homograph2features["complex"] = (
            ["K", "AH0", "M", "P", "L", "EH1", "K", "S"],
            ["K", "AA1", "M", "P", "L", "EH0", "K", "S"],
            "JJ",
        )

    def __call__(self, text):
        words = word_tokenize(text)
        tokens = pos_tag(words)

        prons = []
        for o_word, pos in tokens:
            word = o_word.lower()

            if re.search("[a-z]", word) is None:
                pron = [word]
            elif len(word) == 1:
                if o_word == "A":
                    pron = ["EY1"]
                else:
                    pron = self.cmu[word][0]
            elif word in self.homograph2features:
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                elif len(pos) < len(pos1) and pos == pos1[: len(pos)]:
                    pron = pron1
                else:
                    pron = pron2
            else:
                pron = self.qryword(o_word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]

    def qryword(self, o_word):
        word = o_word.lower()

        if len(word) > 1 and word in self.cmu:
            return self.cmu[word][0]

        if o_word.istitle() and word in self.namedict:
            return self.namedict[word][0]

        if len(word) <= 3:
            phones = []
            for w in word:
                if w == "a":
                    phones.extend(["EY1"])
                elif not w.isalpha():
                    phones.extend([w])
                else:
                    phones.extend(self.cmu[w][0])
            return phones

        if re.match(r"^([a-z]+)('s)$", word):
            phones = self.qryword(word[:-2])[:]
            if phones[-1] in ["P", "T", "K", "F", "TH", "HH"]:
                phones.extend(["S"])
            elif phones[-1] in ["S", "Z", "SH", "ZH", "CH", "JH"]:
                phones.extend(["AH0", "Z"])
            else:
                phones.extend(["Z"])
            return phones

        comps = wordsegment.segment(word.lower())

        if len(comps) == 1:
            return self.predict(word)

        return [phone for comp in comps for phone in self.qryword(comp)]


_g2p = en_G2p()


def g2p(text):
    phone_list = _g2p(text)
    phones = [ph if ph != "<unk>" else "UNK" for ph in phone_list if ph not in [" ", "<pad>", "UW", "</s>", "<s>"]]

    return replace_phs(phones)


if __name__ == "__main__":
    print(g2p("hello"))
    print(g2p(text_normalize("e.g. I used openai's AI tool to draw a picture.")))
    print(g2p(text_normalize("In this; paper, we propose 1 DSPGAN, a GAN-based universal vocoder.")))
