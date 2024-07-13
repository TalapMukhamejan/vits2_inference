_pad = '_'
_punctuation = ' !+,-.:;?«»—'

def get_letters(lang):
    if lang == 'kz':
        return 'аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя'
    elif lang == 'ru':
        return 'абвгдежзийклмнопрстуфхцчшщъыьэюяё'
    elif lang == 'en':
        global _punctuation
        _punctuation = ';:,.!?¡¿—…"«»“” '
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    else:
        raise ValueError(f"Unsupported language: {lang}")

_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

def get_symbols(lang):
    return [_pad] + list(_punctuation) + list(get_letters(lang)) + list(_letters_ipa)
