import re
import unicodedata


def _basic_english_normalize(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    return text


_PATTERNS = [
    (re.compile(r"\'"), " '  "),
    (re.compile(r"\""), ""),
    (re.compile(r"\."), " . "),
    (re.compile(r"<br \/>"), " "),
    (re.compile(r","), " , "),
    (re.compile(r"\("), " ( "),
    (re.compile(r"\)"), " ) "),
    (re.compile(r"\!"), " ! "),
    (re.compile(r"\?"), " ? "),
    (re.compile(r"\;"), " "),
    (re.compile(r"\:"), " "),
    (re.compile(r"\s+"), " "),
]


def basic_english(text):
    text = _basic_english_normalize(text)
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text.strip().split()


def whitespace_tokenizer(text):
    return text.strip().split()


_TOKENIZERS = {
    "basic_english": basic_english,
    "whitespace": whitespace_tokenizer,
}


def get_tokenizer(name, language="en"):
    if callable(name):
        return name
    if name in _TOKENIZERS:
        return _TOKENIZERS[name]
    raise ValueError("unknown tokenizer %r; available: %s" % (name, sorted(_TOKENIZERS)))


def ngrams_iterator(token_list, ngrams):
    for x in token_list:
        yield x
    for n in range(2, ngrams + 1):
        for i in range(len(token_list) - n + 1):
            yield " ".join(token_list[i:i + n])
