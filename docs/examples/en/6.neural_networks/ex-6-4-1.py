import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

embedding = lt.nn.Embedding(num_embeddings=32000, embedding_dim=4096)
norm = lt.nn.LayerNorm(normalized_shape=[4096], eps=1e-5)
dropout = lt.nn.Dropout(p=0.1)

relu = lt.nn.ReLU()
leaky_relu = lt.nn.LeakyReLU(negative_slope=0.02)
sigmoid = lt.nn.Sigmoid()
tanh = lt.nn.Tanh()
gelu = lt.nn.GELU()
softmax = lt.nn.Softmax(dim=-1)
flatten = lt.nn.Flatten()
