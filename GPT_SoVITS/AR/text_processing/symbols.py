"""Symbol table (pad, punctuation, letters, IPA) and id mappings for the phonemizer.

Modified from https://github.com/yangdongchao/SoundStorm and https://github.com/lifeiteng/vall-e.
"""

PAD = "_"
PUNCTUATION = ';:,.!?¡¿—…"«»“” '
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
IPA_LETTERS = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)
SYMBOLS = [PAD] + list(PUNCTUATION) + list(LETTERS) + list(IPA_LETTERS)
SPACE_ID = SYMBOLS.index(" ")
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}
