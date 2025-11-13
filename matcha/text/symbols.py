""" from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""
_pad = "_"
# _punctuation = ';:,.!?¡¿—…"«»“” ' # default punctuations
_punctuation = ';:,.!?¡¿—…"«»“”`\'1234567890- ' # new
# _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters = "abcdefghijklmnopqrstuvwxyz"
# _letters_ipa = (
#     "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
# )


# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters)

# print(len(symbols))

# Special symbol ids
SPACE_ID = symbols.index(" ")
