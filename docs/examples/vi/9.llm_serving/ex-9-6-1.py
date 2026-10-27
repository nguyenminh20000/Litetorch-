import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

vocab = ["{", "}", '"name"', ':', '"John"', '"age"', '25', ","]
decoder = lt.nn.GuidedDecoder(vocab)

logits = lt.Tensor.from_vector([1.0] * len(vocab), [1, len(vocab)], lt.auto_device(), False)
masked_logits = decoder.apply_mask(logits, prefix="{", target_pattern='\\{"name": "[a-zA-Z]+"\\}')
