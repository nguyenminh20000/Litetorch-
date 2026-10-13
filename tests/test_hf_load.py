import json
import os
import tempfile

import litetorch as lt

CPU = lt.Device("cpu")

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"[FAIL] {name}")

def t(shape):
    return lt.Tensor.zeros(shape, CPU)

def make_repo(d, model_type="llama", sharded=False):
    cfg = {
        "model_type": model_type,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "vocab_size": 100,
        "max_position_embeddings": 128,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    if model_type == "llama":
        names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
    elif model_type == "gpt2":
        names = [
            "transformer.wte.weight",
            "transformer.wpe.weight",
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.0.attn.c_proj.weight",
            "transformer.h.0.ln_1.weight",
            "transformer.h.0.ln_2.weight",
            "transformer.h.0.mlp.c_fc.weight",
            "transformer.h.0.mlp.c_proj.weight",
            "transformer.ln_f.weight",
            "lm_head.weight",
        ]
    else:
        names = [
            "bert.embeddings.word_embeddings.weight",
            "bert.encoder.layer.0.attention.self.query.weight",
            "bert.encoder.layer.0.attention.self.key.weight",
            "bert.encoder.layer.0.attention.self.value.weight",
            "bert.encoder.layer.0.attention.output.dense.weight",
            "bert.encoder.layer.0.attention.output.LayerNorm.weight",
            "bert.encoder.layer.0.intermediate.dense.weight",
            "bert.encoder.layer.0.output.dense.weight",
            "bert.encoder.layer.0.output.LayerNorm.weight",
            "bert.pooler.dense.weight",
        ]
    tensors = {n: t([4, 4]) for n in names}
    if sharded:
        half = len(names) // 2
        files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        lt.save_safetensors({k: tensors[k] for k in names[:half]}, os.path.join(d, files[0]))
        lt.save_safetensors({k: tensors[k] for k in names[half:]}, os.path.join(d, files[1]))
        wmap = {}
        for i, n in enumerate(names):
            wmap[n] = files[0] if i < half else files[1]
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": wmap}, f)
    else:
        lt.save_safetensors(tensors, os.path.join(d, "model.safetensors"))
    return names

with tempfile.TemporaryDirectory() as d:
    names = make_repo(d, "llama")

    cfg = lt.hf.load_config(d)
    check("config model_type", cfg.model_type == "llama")
    check("config hidden_size", cfg.hidden_size == 32)
    check("config num_hidden_layers", cfg.num_hidden_layers == 2)
    check("config num_attention_heads", cfg.num_attention_heads == 4)
    check("config num_key_value_heads", cfg.num_key_value_heads == 2)
    check("config intermediate_size", cfg.intermediate_size == 64)
    check("config vocab_size", cfg.vocab_size == 100)
    check("config hidden_act", cfg.hidden_act == "silu")

    bundle = lt.hf.from_pretrained(d)
    check("bundle config type", bundle.config.model_type == "llama")
    check("bundle state_dict count", len(bundle.state_dict) == len(names))
    check("bundle has q_proj", "model.layers.0.self_attn.q_proj.weight" in bundle.state_dict)

    remapped = lt.hf.remap_state_dict(bundle.state_dict, "llama")
    check("llama q_proj remapped", "layers.0.attn.q_proj.weight" in remapped)
    check("llama k_proj remapped", "layers.0.attn.k_proj.weight" in remapped)
    check("llama o_proj remapped", "layers.0.attn.o_proj.weight" in remapped)
    check("llama gate_proj remapped", "layers.0.mlp.gate_proj.weight" in remapped)
    check("llama input_layernorm remapped", "layers.0.attn_norm.weight" in remapped)
    check("llama post_attn remapped", "layers.0.ffn_norm.weight" in remapped)
    check("llama embed remapped", "embed.weight" in remapped)
    check("llama norm remapped", "final_norm.weight" in remapped)
    check("llama lm_head kept", "lm_head.weight" in remapped)
    check("llama layer1 remapped", "layers.1.attn.q_proj.weight" in remapped)

    wmap = lt.hf.weight_name_map("llama")
    check("weight_name_map embed", wmap.get("model.embed_tokens.weight") == "embed.weight")

with tempfile.TemporaryDirectory() as d:
    make_repo(d, "gpt2")
    bundle = lt.hf.from_pretrained(d)
    remapped = lt.hf.remap_state_dict(bundle.state_dict, "gpt2")
    check("gpt2 c_attn remapped", "layers.0.attn.c_attn.weight" in remapped)
    check("gpt2 ln_1 remapped", "layers.0.attn_norm.weight" in remapped)
    check("gpt2 ln_2 remapped", "layers.0.ffn_norm.weight" in remapped)
    check("gpt2 wte remapped", "embed.weight" in remapped)
    check("gpt2 wpe remapped", "pos_embed.weight" in remapped)
    check("gpt2 ln_f remapped", "final_norm.weight" in remapped)

with tempfile.TemporaryDirectory() as d:
    make_repo(d, "bert")
    bundle = lt.hf.from_pretrained(d)
    remapped = lt.hf.remap_state_dict(bundle.state_dict, "bert")
    check("bert query remapped", "layers.0.attn.query_proj.weight" in remapped)
    check("bert key remapped", "layers.0.attn.key_proj.weight" in remapped)
    check("bert dense remapped", "layers.0.attn.o_proj.weight" in remapped)
    check("bert intermediate remapped", "layers.0.mlp.up_proj.weight" in remapped)
    check("bert word_embeddings remapped", "embed.word.weight" in remapped)
    check("bert pooler remapped", "pooler.weight" in remapped)

with tempfile.TemporaryDirectory() as d:
    names = make_repo(d, "llama", sharded=True)
    bundle = lt.hf.from_pretrained(d)
    check("sharded count", len(bundle.state_dict) == len(names))
    check("sharded has shard2 tensor", "model.norm.weight" in bundle.state_dict)

check("hasattr hf", hasattr(lt, "hf"))
check("hasattr from_pretrained", hasattr(lt.hf, "from_pretrained"))

print(f"\nPASS={PASS} FAIL={FAIL}")
raise SystemExit(1 if FAIL else 0)
