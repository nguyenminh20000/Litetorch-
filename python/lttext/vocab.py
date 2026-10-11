from collections import Counter, OrderedDict


class Vocab:
    def __init__(self, stoi, itos=None, default_index=None):
        self.stoi = dict(stoi)
        if itos is None:
            itos = [None] * len(self.stoi)
            for tok, idx in self.stoi.items():
                itos[idx] = tok
        self.itos = list(itos)
        self.default_index = default_index

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        if isinstance(token, (list, tuple)):
            return [self[t] for t in token]
        if token in self.stoi:
            return self.stoi[token]
        if self.default_index is not None:
            return self.default_index
        raise KeyError("token %r not in vocab and no default index set" % (token,))

    def lookup_token(self, index):
        return self.itos[index]

    def lookup_tokens(self, indices):
        return [self.itos[i] for i in indices]

    def lookup_index(self, token):
        return self[token]

    def lookup_indices(self, tokens):
        return [self[t] for t in tokens]

    def get_stoi(self):
        return dict(self.stoi)

    def get_itos(self):
        return list(self.itos)

    def set_default_index(self, index):
        self.default_index = index

    def __contains__(self, token):
        return token in self.stoi


def build_vocab_from_iterator(iterator, min_freq=1, specials=None, special_first=True,
                              max_tokens=None):
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    specials = list(specials) if specials else []
    stoi = OrderedDict()
    if special_first:
        for s in specials:
            if s not in stoi:
                stoi[s] = len(stoi)
    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if max_tokens is not None and len(stoi) >= max_tokens:
            break
        if tok not in stoi:
            stoi[tok] = len(stoi)
    if not special_first:
        for s in specials:
            if s not in stoi:
                stoi[s] = len(stoi)
    return Vocab(stoi)
