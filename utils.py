import re

def preprocess(text):

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text