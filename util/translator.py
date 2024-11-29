from deep_translator import GoogleTranslator


def korean_to_english(text):
    return GoogleTranslator(source='ko', target='en').translate(
        text
    )


def english_to_korean(text):
    return GoogleTranslator(source='en', target='ko').translate(
        text
    )
