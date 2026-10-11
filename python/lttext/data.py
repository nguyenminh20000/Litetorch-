import math
import random

import litetorch as lt


class Field:
    def __init__(self, sequential=True, tokenize=None, lower=False, include_lengths=False,
                 batch_first=False, pad_token="<pad>", unk_token="<unk>",
                 init_token=None, eos_token=None, dtype=None):
        self.sequential = sequential
        self.tokenize = tokenize or (lambda s: s.split())
        self.lower = lower
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.init_token = init_token
        self.eos_token = eos_token
        self.dtype = dtype
        self.vocab = None

    def preprocess(self, text):
        if self.lower:
            text = text.lower()
        return self.tokenize(text)

    def build_vocab(self, texts, min_freq=1, max_size=None):
        from .vocab import build_vocab_from_iterator
        specials = [self.pad_token, self.unk_token]
        if self.init_token:
            specials.append(self.init_token)
        if self.eos_token:
            specials.append(self.eos_token)
        vocab = build_vocab_from_iterator(
            (self.preprocess(t) for t in texts),
            min_freq=min_freq, specials=specials, max_tokens=max_size)
        vocab.set_default_index(vocab[self.unk_token])
        self.vocab = vocab
        return vocab

    def numericalize(self, texts):
        if self.vocab is None:
            raise RuntimeError("build_vocab must be called before numericalize")
        seqs = []
        for t in texts:
            toks = self.preprocess(t)
            if self.init_token:
                toks = [self.init_token] + toks
            if self.eos_token:
                toks = toks + [self.eos_token]
            seqs.append([self.vocab[x] for x in toks])
        return seqs

    def pad(self, minibatch):
        pad_idx = self.vocab[self.pad_token]
        lengths = [len(x) for x in minibatch]
        max_len = max(lengths) if lengths else 0
        padded = [x + [pad_idx] * (max_len - len(x)) for x in minibatch]
        if self.batch_first:
            arr = padded
        else:
            arr = [list(col) for col in zip(*padded)] if padded else []
        flat = [v for row in arr for v in row]
        shape = [len(arr), len(arr[0])] if arr else [0, 0]
        tensor = lt.Tensor.from_vector([float(v) for v in flat], shape, lt.Device("cpu"))
        if self.include_lengths:
            return tensor, lengths
        return tensor


class Batch:
    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


def batch_iterator(dataset, batch_size, field, shuffle=True, sort_key=None):
    idx = list(range(len(dataset)))
    if sort_key is not None:
        idx.sort(key=lambda i: sort_key(dataset[i]))
    elif shuffle:
        random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        texts = [dataset[i] for i in chunk]
        yield field.pad(field.numericalize(texts))


def bucket_iterator(dataset, batch_size, field, sort_key=None, shuffle=True):
    if sort_key is None:
        sort_key = lambda x: len(field.preprocess(x))
    return batch_iterator(dataset, batch_size, field, shuffle=shuffle, sort_key=sort_key)
