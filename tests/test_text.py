import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import litetorch as lt
import lttext
from lttext import (Vocab, build_vocab_from_iterator, get_tokenizer,
                    basic_english, ngrams_iterator, Field, Batch,
                    batch_iterator, bucket_iterator)

print("=== vocab ===")
it = [["hello", "world"], ["hello", "there"], ["foo"]]
v = build_vocab_from_iterator(it, min_freq=1, specials=["<unk>", "<pad>"])
assert v["<unk>"] == 0 and v["<pad>"] == 1, (v["<unk>"], v["<pad>"])
assert v["hello"] == 2, v["hello"]
assert v.lookup_token(2) == "hello"
assert v.lookup_indices(["hello", "world"]) == [2, 3]
assert len(v) == 6, len(v)
v.set_default_index(0)
assert v["missing"] == 0
print("build_vocab_from_iterator OK", v.get_itos())

v2 = build_vocab_from_iterator(it, min_freq=2, specials=["<unk>"])
assert "foo" not in v2 and "hello" in v2
print("min_freq OK")

print("\n=== tokenizer ===")
tok = get_tokenizer("basic_english")
toks = tok("Hello, World! It's great.")
assert toks == ["hello", ",", "world", "!", "it", "'", "s", "great", "."], toks
print("basic_english OK", toks)
wtok = get_tokenizer("whitespace")
assert wtok("a  b c") == ["a", "b", "c"]
grams = list(ngrams_iterator(["a", "b", "c"], 2))
assert grams == ["a", "b", "c", "a b", "b c"], grams
print("ngrams OK")

print("\n=== Field ===")
texts = ["hello world", "foo bar baz", "hi"]
f = Field(sequential=True, tokenize=basic_english, batch_first=True,
          include_lengths=True, init_token="<bos>", eos_token="<eos>")
f.build_vocab(texts, min_freq=1)
assert "<bos>" in f.vocab and "<eos>" in f.vocab
num = f.numericalize(texts)
assert len(num) == 3 and all(len(s) >= 3 for s in num)
padded, lengths = f.pad(num)
assert list(padded.shape) == [3, 5], padded.shape
assert lengths == [4, 5, 3], lengths
pv = padded.to_vector()
assert pv[0] == float(f.vocab["<bos>"])
print("Field OK", list(padded.shape), lengths)

f2 = Field(tokenize=basic_english, batch_first=False)
f2.build_vocab(texts)
p2 = f2.pad(f2.numericalize(texts[:2]))
assert list(p2.shape) == [3, 2], p2.shape
print("Field time-first OK", list(p2.shape))

print("\n=== iterators ===")
data = ["hello world foo", "a b", "x y z w", "p q"]
f3 = Field(tokenize=basic_english, batch_first=True)
f3.build_vocab(data)
batches = list(batch_iterator(data, 2, f3, shuffle=False))
assert len(batches) == 2
assert list(batches[0].shape) == [2, 3], batches[0].shape
bb = list(bucket_iterator(data, 2, f3, shuffle=False))
assert len(bb) == 2
print("iterators OK")

print("\n=== ALL TEXT TESTS PASSED ===")
