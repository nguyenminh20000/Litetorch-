"""Pretrain GPT2B (2.074B params) on Kaggle 2x Tesla T4 with pipeline parallelism.

Launch:
    cd /tmp/lt-bench && PYTHONPATH=/tmp/lt-bench nohup python3 tests/train_2b.py > /tmp/train_2b.log 2>&1 &

Flags:
    --fp16      enable FP16 autocast + grad scaling (default: FP32, proven path)
    --resume    resume from latest checkpoint in CKPT_DIR

Requires:
    - litetorch built with CUDA, 2 GPUs
    - slimpajama_loader.py  (env LLMB_DIR or ~/workspace/llm-2b or ./llm-2b)
    - gpt2_tokenizer.json   (env TOKENIZER_PATH or same dirs; else HF download)

Checkpoints: CKPT_DIR/ckpt_step_<N>/  (sharded SafeTensors + trainer_state.json)
Resume requires the same model config and pipeline split as the saved run.
"""

import json
import os
import signal
import sys
import time

import litetorch as lt
from litetorch import nn


VOCAB = 50257
D_MODEL = 2560
N_LAYERS = 28
N_Q_HEADS = 20
N_KV_HEADS = 5
D_FF = 6912
SEQ_LEN = 2048

SPLIT = 14

MICRO_BS = 2
GRAD_ACCUM = 8
TOKENS_PER_STEP = MICRO_BS * GRAD_ACCUM * SEQ_LEN

TOTAL_TOKENS = 3_000_000_000
TOTAL_STEPS = int(TOTAL_TOKENS // TOKENS_PER_STEP)

PEAK_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 2000
WEIGHT_DECAY = 0.1

CKPT_DIR = "/tmp/checkpoints"
CKPT_EVERY = 1000
LOG_EVERY = 100

T4_PEAK_FP16 = 65.0
T4_PEAK_FP32 = 8.1

USE_FP16 = "--fp16" in sys.argv
DO_RESUME = "--resume" in sys.argv

D0 = lt.Device("cuda:0")
D1 = lt.Device("cuda:1")
CPU = lt.Device("cpu")
_sync = getattr(lt, "cuda_synchronize", lambda: None)

STOP = {"flag": False}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _request_stop(signum, frame):
    log(f"signal {signum} received, will stop after current step")
    STOP["flag"] = True


def find_loader():
    import importlib.util
    cands = []
    if os.environ.get("LLMB_DIR"):
        cands.append(os.environ["LLMB_DIR"])
    cands += [os.path.expanduser("~/workspace/llm-2b"), "./llm-2b",
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../llm-2b")]
    for d in cands:
        p = os.path.join(d, "slimpajama_loader.py")
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("slimpajama_loader", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            log(f"loader: {p}")
            return mod
    raise RuntimeError("slimpajama_loader.py not found; set LLMB_DIR")


def find_tokenizer():
    cands = []
    if os.environ.get("TOKENIZER_PATH"):
        cands.append(os.environ["TOKENIZER_PATH"])
    cands += [os.path.expanduser("~/workspace/llm-2b/gpt2_tokenizer.json"),
              "./llm-2b/gpt2_tokenizer.json"]
    for p in cands:
        if os.path.isfile(p):
            log(f"tokenizer: {p}")
            return p
    log("tokenizer file not found, loader will try HF download")
    return None


def latest_ckpt():
    ptr = os.path.join(CKPT_DIR, "latest.txt")
    if not os.path.isfile(ptr):
        return None
    with open(ptr) as f:
        d = f.read().strip()
    st = os.path.join(d, "trainer_state.json")
    if os.path.isdir(d) and os.path.isfile(st):
        return d
    return None


def save_checkpoint(model, step, tokens_seen, loss_val, lr, scaler, sched):
    d = os.path.join(CKPT_DIR, f"ckpt_step_{step}")
    os.makedirs(CKPT_DIR, exist_ok=True)
    log(f"saving checkpoint step {step} -> {d}")
    model.save_pretrained(d, num_shards=4)
    state = {
        "step": step,
        "tokens_seen": tokens_seen,
        "loss": loss_val,
        "lr": lr,
        "scaler_scale": scaler.get_scale(),
        "sched_last_step": sched.last_step,
        "rng_state": lt.get_rng_state(),
        "fp16": USE_FP16,
    }
    with open(os.path.join(d, "trainer_state.json"), "w") as f:
        json.dump(state, f)
    with open(os.path.join(CKPT_DIR, "latest.txt"), "w") as f:
        f.write(d)
    log(f"checkpoint saved (step {step})")


def load_checkpoint(model, opt, sched, scaler):
    d = latest_ckpt()
    if d is None:
        return 0, 0
    log(f"resuming from {d}")
    model.load_pretrained(d)
    with open(os.path.join(d, "trainer_state.json")) as f:
        state = json.load(f)
    step = int(state["step"])
    opt.set_lr(float(state["lr"]))
    sched.last_step = int(state["sched_last_step"])
    scaler.set_scale(float(state["scaler_scale"]))
    lt.set_rng_state(state["rng_state"])
    log(f"resumed at step {step}, tokens_seen={state['tokens_seen']}")
    return step, int(state["tokens_seen"])


def sync_tied_weights(model):
    gh = model.lm_head.weight.grad
    ge = model.tok_emb.weight.grad
    if gh is not None and ge is not None:
        model.tok_emb.weight.grad = lt.Ops.add(ge, gh.to(D0))
    model.lm_head.weight.copy_(model.tok_emb.weight.to(D1))


def make_block_fn(block):
    def fn(h):
        return block.forward(h)
    return fn


def main():
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    assert lt.is_gpu_available(), "CUDA not available"
    for dev, name in ((D0, "cuda:0"), (D1, "cuda:1")):
        lt.Tensor.zeros([1], dev, False)
        log(f"{name} ok")

    log(f"building GPT2B (fp16={USE_FP16})")
    lt.manual_seed(42)
    model = nn.GPT2B(vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS,
                     n_q_heads=N_Q_HEADS, n_kv_heads=N_KV_HEADS, d_ff=D_FF)
    model.pipeline_to(D0, D1, split=SPLIT)

    params = model.parameters()
    n_params = sum(p.numel() for p in params)
    log(f"params: {n_params/1e9:.3f}B in {len(params)} tensors")

    opt = lt.optim.AdamW8bit(params, lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    sched = lt.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TOTAL_STEPS - WARMUP_STEPS, eta_min=MIN_LR)
    scaler = lt.optim.GradScaler()
    ce = nn.CrossEntropyLoss()
    inv_accum = lt.Tensor.from_vector([1.0 / GRAD_ACCUM], [], D1, False)

    block_fns = [make_block_fn(b) for b in model.blocks]

    step, tokens_seen = 0, 0
    if DO_RESUME:
        step, tokens_seen = load_checkpoint(model, opt, sched, scaler)

    loader_mod = find_loader()
    tok_path = find_tokenizer()
    loader = loader_mod.SlimPajamaStreamLoader(
        seq_len=SEQ_LEN, batch_size=MICRO_BS, tokenizer_path=tok_path)
    log(f"total_steps={TOTAL_STEPS} tokens/step={TOKENS_PER_STEP} "
        f"warmup={WARMUP_STEPS} ckpt_every={CKPT_EVERY}")

    def to_toks(arr, dev):
        return lt.Tensor.from_vector([float(v) for row in arr for v in row],
                                     [MICRO_BS, SEQ_LEN], dev, False)

    def forward_ckpt(toks):
        h = model.tok_emb.forward(toks)
        for i in range(SPLIT):
            h = lt.checkpoint(block_fns[i], h)
        h = h.to(D1)
        for i in range(SPLIT, N_LAYERS):
            h = lt.checkpoint(block_fns[i], h)
        h = model.final_norm.forward(h)
        return model.lm_head.forward(h)

    def forward(toks):
        if not USE_FP16:
            return forward_ckpt(toks)
        guard = lt.amp.AutocastGuard(True, lt.DataType.FP16)
        try:
            return forward_ckpt(toks)
        finally:
            del guard

    peak = T4_PEAK_FP16 if USE_FP16 else T4_PEAK_FP32
    ema_loss = None

    try:
        while step < TOTAL_STEPS and not STOP["flag"]:
            opt.zero_grad()
            _sync()
            t0 = time.time()
            for _ in range(GRAD_ACCUM):
                x_np, y_np = loader.next_batch()
                toks = to_toks(x_np, D0)
                tgt = to_toks(y_np, D1)
                logits = forward(toks)
                lv = logits.view([MICRO_BS * SEQ_LEN, VOCAB])
                tv = tgt.view([MICRO_BS * SEQ_LEN])
                loss = ce.forward(lv, tv)
                scaler.scale_loss(lt.Ops.mul(loss, inv_accum)).backward()
                tokens_seen += MICRO_BS * SEQ_LEN
            sync_tied_weights(model)
            scaler.step(opt)
            scaler.update()

            step += 1
            if step <= WARMUP_STEPS:
                opt.set_lr(PEAK_LR * step / WARMUP_STEPS)
            else:
                if step == WARMUP_STEPS + 1:
                    opt.set_lr(PEAK_LR)
                sched.step()
            _sync()
            dt = time.time() - t0

            lv_now = loss.to(CPU).to_vector()[0]
            ema_loss = lv_now if ema_loss is None else 0.95 * ema_loss + 0.05 * lv_now

            if step % LOG_EVERY == 0:
                tflops = 6 * n_params * TOKENS_PER_STEP / dt / 1e12
                mfu = 100.0 * tflops / peak
                tps = TOKENS_PER_STEP / dt
                eta_h = (TOTAL_STEPS - step) * dt / 3600
                log(f"step {step}/{TOTAL_STEPS} loss={ema_loss:.4f} lr={opt.get_lr():.2e} "
                    f"{tps:.0f} tok/s {tflops:.2f} TFLOPS MFU={mfu:.1f}% eta={eta_h:.1f}h")

            if step % CKPT_EVERY == 0:
                save_checkpoint(model, step, tokens_seen, ema_loss,
                                opt.get_lr(), scaler, sched)
    except RuntimeError as e:
        if "memory" in str(e).lower() or "out of memory" in str(e).lower():
            log(f"OOM at step {step}: {e}")
            log("attempting emergency checkpoint save via finally block")
            sys.exit(2)
        raise
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        if step > 0 and step % CKPT_EVERY != 0:
            try:
                save_checkpoint(model, step, tokens_seen, ema_loss or float("nan"),
                                opt.get_lr(), scaler, sched)
            except Exception as e:
                log(f"final save failed: {e}")

    log(f"done: step={step} tokens_seen={tokens_seen}")


if __name__ == "__main__":
    main()
