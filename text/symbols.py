_pad = '_'
_punctuation_ru_kz = ' !+,-.:;?«»—'
_punctuation_en = ';:,.!?¡¿—…"«»"" '

def get_letters(lang):
    if lang == 'kz':
        return 'аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя'
    elif lang == 'ru':
        return 'абвгдежзийклмнопрстуфхцчшщъыьэюяё'
    elif lang == 'en':
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    else:
        raise ValueError(f"Unsupported language: {lang}")

_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

def get_symbols(lang):
    if lang == 'en':
        return [_pad] + list(_punctuation_en) + list(get_letters(lang)) + list(_letters_ipa)
    else:
        return [_pad] + list(_punctuation_ru_kz) + list(get_letters(lang)) + list(_letters_ipa)