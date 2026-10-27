import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
decoder_layer = lt.nn.TransformerDecoderLayer(embed_dim=512, num_heads=8, dim_feedforward=2048)
