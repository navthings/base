import importlib.util, subprocess, sys

need = [p for p in ("datasets", "safetensors", "transformers", "psutil", "matplotlib") if importlib.util.find_spec(p) is None]
if need:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)

import glob, json, math, os, queue, threading, time
from functools import partial

import datasets, transformers
import jax
import jax.numpy as jnp
import numpy as np
import psutil
from datasets import load_dataset
from safetensors.numpy import load_file, save_file
from transformers import AutoTokenizer

RUN_START = time.time()

# hf token avoids rate limits over a 9h stream
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from kaggle secrets")
except Exception as e:
    print(f"no HF_TOKEN secret ({type(e).__name__}), streaming unauthenticated")

DEVICES = jax.devices()
N_CHIPS = len(DEVICES)
DEV = DEVICES[0]
print(f"jax {jax.__version__}, {DEV.platform} {DEV.device_kind}, {N_CHIPS} device(s)")
print(f"host RAM {psutil.virtual_memory().total / 2**30:.0f} GB")

# kaggle sometimes hands out a degraded tpu with 1 chip, fail loudly instead of training on 1/8th
if DEV.platform != "tpu" and not os.environ.get("LILSTORY_ALLOW_CPU"):
    raise RuntimeError("jax can't see a tpu. set the accelerator to tpu, or pip install -U 'jax[tpu]' and restart")
if DEV.platform == "tpu" and N_CHIPS != 8:
    raise RuntimeError(f"expected 8 tpu chips (v5e-8), jax sees {N_CHIPS}. restart the session and try again")
if not hasattr(datasets.IterableDataset, "state_dict"):
    raise RuntimeError(f"datasets {datasets.__version__} can't save stream position; pip install -U datasets")

mesh = jax.sharding.Mesh(np.array(DEVICES), ("data",))
P = jax.sharding.PartitionSpec
replicated = jax.sharding.NamedSharding(mesh, P())
data_sharded = jax.sharding.NamedSharding(mesh, P("data"))

# ~297m params, 6.1b tokens is chinchilla-optimal; micro=16 oom'd hbm so 8x8 accum keeps the same tokens/step
VOCAB, D, N_LAYERS, N_HEADS, N_KV, D_FF = 32000, 1024, 24, 16, 4, 2730
HEAD_DIM = D // N_HEADS
ROPE_THETA, NORM_EPS = 10000.0, 1e-5
SEQ = 1024
MICRO, ACCUM = 8, 8
TOTAL_TOKENS = 6_100_000_000
CKPT_EVERY, LOG_EVERY, EVAL_EVERY = 500, 50, 500
PEAK_LR, WARMUP_STEPS, MIN_LR_RATIO = 4e-4, 1000, 0.1
BETA1, BETA2, ADAM_EPS, WEIGHT_DECAY, GRAD_CLIP = 0.9, 0.95, 1e-8, 0.1, 1.0
EVAL_SEQS, EVAL_DOCS = 256, 1500
SHUFFLE_BUFFER = 5000
MIN_FREE_RAM_GB = 2.0
SAMPLE_LEN = 256

TOKENS_PER_STEP = MICRO * ACCUM * SEQ * N_CHIPS
TOTAL_STEPS = TOTAL_TOKENS // TOKENS_PER_STEP

DATASET, DATASET_CFG = "HuggingFaceFW/fineweb-edu", "sample-10BT"
TOKENIZER = "hf-internal-testing/llama-tokenizer"
OUT = "/kaggle/working/lilstory"
# sessions die at 9h, so stop early and resume from a previous version's output added as input
RESUME_GLOB = "/kaggle/input/**/lilstory_step_*.json"
TIME_BUDGET_H = 8.4
os.makedirs(OUT, exist_ok=True)
DT = jnp.bfloat16 if DEV.platform == "tpu" else jnp.float32


class RamWatch:
    def __init__(self):
        self.phase, self.low, self.peak = "setup", False, 0.0
        threading.Thread(target=self._run, daemon=True).start()

    # samples every second so compile and eval spikes get caught too
    def _run(self):
        proc = psutil.Process()
        while True:
            rss = proc.memory_info().rss / 2**30
            free = psutil.virtual_memory().available / 2**30
            self.peak = max(self.peak, rss)
            low = free < MIN_FREE_RAM_GB
            if low and not self.low:
                print(f"[ram] only {free:.1f} GB free during '{self.phase}', this process holds {rss:.1f} GB", flush=True)
            self.low = low
            time.sleep(1)


def rms_norm(x, w):
    xf = x.astype(jnp.float32)
    return xf * jax.lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + NORM_EPS) * w


# bf16 matmul with fp32 accumulate; out=float32 for residual adds
def linear(x, w, out=None):
    return jnp.einsum("...i,oi->...o", x.astype(DT), w.astype(DT), preferred_element_type=out or DT)


def rope(x, cos, sin):
    half = HEAD_DIM // 2
    x1, x2 = x[..., :half].astype(jnp.float32), x[..., half:].astype(jnp.float32)
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def rope_tables(seq_len):
    inv = ROPE_THETA ** (-jnp.arange(0, HEAD_DIM, 2, dtype=jnp.float32) / HEAD_DIM)
    t = jnp.arange(seq_len, dtype=jnp.float32)[:, None] * inv[None]
    return jnp.cos(t), jnp.sin(t)


def block(p, x, cos, sin):
    B, L, _ = x.shape
    h = rms_norm(x, p["attn_norm.weight"])
    q = linear(h, p["attn.q_proj.weight"]).reshape(B, L, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = linear(h, p["attn.k_proj.weight"]).reshape(B, L, N_KV, HEAD_DIM).transpose(0, 2, 1, 3)
    v = linear(h, p["attn.v_proj.weight"]).reshape(B, L, N_KV, HEAD_DIM).transpose(0, 2, 1, 3)
    q, k = rope(q, cos, sin).astype(DT), rope(k, cos, sin).astype(DT)
    k = jnp.repeat(k, N_HEADS // N_KV, axis=1)
    v = jnp.repeat(v, N_HEADS // N_KV, axis=1)
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32) * HEAD_DIM ** -0.5
    s = jnp.where(jnp.tril(jnp.ones((L, L), dtype=bool)), s, -jnp.inf)
    a = jax.nn.softmax(s, axis=-1).astype(DT)
    o = jnp.einsum("bhqk,bhkd->bhqd", a, v).transpose(0, 2, 1, 3).reshape(B, L, -1)
    x = x + linear(o, p["attn.o_proj.weight"], jnp.float32)
    h = rms_norm(x, p["mlp_norm.weight"])
    m = jax.nn.silu(linear(h, p["mlp.gate_proj.weight"])) * linear(h, p["mlp.up_proj.weight"])
    return x + linear(m, p["mlp.down_proj.weight"], jnp.float32)


# scan traces block() once and reuses it for every layer
def forward(params, ids, remat=True):
    cos, sin = rope_tables(ids.shape[1])
    x = params["embed_tokens.weight"][ids].astype(jnp.float32)
    layers = {k[len("layers."):]: v for k, v in params.items() if k.startswith("layers.")}
    body = lambda x, p: (block(p, x, cos, sin), None)
    if remat:
        body = jax.checkpoint(body, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    x, _ = jax.lax.scan(body, x, layers)
    x = rms_norm(x, params["norm.weight"])
    return linear(x, params["embed_tokens.weight"], jnp.float32)


def loss_fn(params, seqs, remat=True):
    logits = forward(params, seqs[:, :-1], remat)
    tgt = seqs[:, 1:]
    nll = jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, tgt[..., None], -1)[..., 0]
    return nll.mean()


# global-norm clip, then decoupled weight decay on matrices only
def adamw_update(params, grads, opt, lr):
    step = opt["step"] + 1
    gnorm = jnp.sqrt(sum(jnp.sum(g * g) for g in grads.values()))
    scale = jnp.minimum(1.0, GRAD_CLIP / (gnorm + 1e-6))
    t = step.astype(jnp.float32)
    c1, c2 = 1 - BETA1 ** t, 1 - BETA2 ** t
    new_p, m, v = {}, {}, {}
    for k, p in params.items():
        g = grads[k] * scale
        m[k] = BETA1 * opt["m"][k] + (1 - BETA1) * g
        v[k] = BETA2 * opt["v"][k] + (1 - BETA2) * g * g
        upd = (m[k] / c1) / (jnp.sqrt(v[k] / c2) + ADAM_EPS)
        if not k.endswith("norm.weight"):
            upd = upd + WEIGHT_DECAY * p
        new_p[k] = p - lr * upd
    return new_p, {"step": step, "m": m, "v": v}, gnorm


# shard_map so each chip scans its own accum steps, plain jit would scan over chips instead
@partial(jax.jit, donate_argnums=(0, 1))
def train_step(params, opt, batch, lr):
    @partial(jax.shard_map, mesh=mesh, in_specs=(P(), P(), P("data"), P()), out_specs=(P(), P(), P(), P()))
    def _step(params, opt, batch, lr):
        local_batch = batch[0]
        def micro(g_acc, mb):
            l, g = jax.value_and_grad(loss_fn)(params, mb)
            return jax.tree.map(jnp.add, g_acc, g), l
        g, losses = jax.lax.scan(micro, jax.tree.map(jnp.zeros_like, params), local_batch)
        g = jax.tree.map(lambda x: jax.lax.pmean(x / local_batch.shape[0], "data"), g)
        loss = jax.lax.pmean(losses.mean(), "data")
        new_params, new_opt, gnorm = adamw_update(params, g, opt, lr)
        return new_params, new_opt, loss, gnorm
    return _step(params, opt, batch, lr)


@jax.jit
def eval_loss(params, seqs):
    return loss_fn(params, seqs, remat=False)


def lr_at_step(step):
    if step < WARMUP_STEPS:
        return PEAK_LR * (step + 1) / WARMUP_STEPS
    pr = min(1.0, (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS))
    floor = PEAK_LR * MIN_LR_RATIO
    return floor + (PEAK_LR - floor) * 0.5 * (1 + math.cos(math.pi * pr))


def param_shapes():
    kv, L = N_KV * HEAD_DIM, N_LAYERS
    return {
        "embed_tokens.weight": (VOCAB, D), "norm.weight": (D,),
        "layers.attn_norm.weight": (L, D), "layers.mlp_norm.weight": (L, D),
        "layers.attn.q_proj.weight": (L, D, D), "layers.attn.k_proj.weight": (L, kv, D),
        "layers.attn.v_proj.weight": (L, kv, D), "layers.attn.o_proj.weight": (L, D, D),
        "layers.mlp.gate_proj.weight": (L, D_FF, D), "layers.mlp.up_proj.weight": (L, D_FF, D),
        "layers.mlp.down_proj.weight": (L, D, D_FF),
    }


def init_params(seed=0):
    rng = np.random.default_rng(seed)
    out_std = 0.02 / math.sqrt(2 * N_LAYERS)
    params = {}
    for k, s in param_shapes().items():
        if k.endswith("norm.weight"):
            arr = np.ones(s, np.float32)
        else:
            std = out_std if k.endswith(("o_proj.weight", "down_proj.weight")) else 0.02
            arr = rng.standard_normal(s, dtype=np.float32) * std
        params[k] = jax.device_put(arr, replicated)
    return params


def init_opt(params):
    return {"step": jax.device_put(jnp.array(0, jnp.int32), replicated),
            "m": {k: jax.device_put(jnp.zeros_like(p), replicated) for k, p in params.items()},
            "v": {k: jax.device_put(jnp.zeros_like(p), replicated) for k, p in params.items()}}


tok = AutoTokenizer.from_pretrained(TOKENIZER)
tok.model_max_length = 10**9
EOS = tok.eos_token_id


def open_stream():
    return load_dataset(DATASET, DATASET_CFG, split="train", streaming=True)


def build_eval_set():
    need = EVAL_SEQS * (SEQ + 1)
    ids = []
    for ex in open_stream().take(EVAL_DOCS):
        ids.extend(tok(ex["text"], add_special_tokens=False)["input_ids"])
        ids.append(EOS)
        if len(ids) >= need:
            return np.array(ids[:need], np.int32).reshape(EVAL_SEQS, SEQ + 1)
    raise RuntimeError(f"only got {len(ids)} tokens from {EVAL_DOCS} docs, need {need}; raise EVAL_DOCS")


class Batches:
    def __init__(self, state=None):
        self.state = state
        self.q = queue.Queue(maxsize=32)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # first EVAL_DOCS docs are held out for eval
    def _open(self):
        ds = open_stream().skip(EVAL_DOCS).shuffle(buffer_size=SHUFFLE_BUFFER, seed=0)
        if self.state is not None:
            ds.load_state_dict(self.state)
        return ds

    # background tokenization; network errors reopen the stream at the last emitted batch
    def _run(self):
        need = N_CHIPS * ACCUM * MICRO * (SEQ + 1)
        failures = 0
        while not self.stop.is_set():
            buf = np.empty(0, np.int32)
            try:
                ds = self._open()
                it = iter(ds)
                while not self.stop.is_set():
                    while buf.size < need:
                        texts = [next(it)["text"] for _ in range(64)]
                        enc = tok(texts, add_special_tokens=False)["input_ids"]
                        buf = np.concatenate([buf] + [np.array(e + [EOS], np.int32) for e in enc])
                    self.state = ds.state_dict()
                    item = (buf[:need].reshape(N_CHIPS, ACCUM, MICRO, SEQ + 1).copy(), self.state)
                    buf = buf[need:]
                    failures = 0
                    while not self.stop.is_set():
                        try:
                            self.q.put(item, timeout=1)
                            break
                        except queue.Full:
                            continue
            except StopIteration:
                self.q.put(RuntimeError("training stream ran out of documents"))
                return
            except Exception as e:
                failures += 1
                if failures > 5:
                    self.q.put(e)
                    return
                print(f"[data] {type(e).__name__}: {e}; reopening stream in {30 * failures}s ({failures}/5)", flush=True)
                self.stop.wait(30 * failures)

    def get(self):
        item = self.q.get()
        if isinstance(item, BaseException):
            raise RuntimeError("data thread died") from item
        return item

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)


def ckpt_name(step):
    return os.path.join(OUT, f"lilstory_step_{step:08d}")


# zero-padded step in the name means max by basename is the newest
def latest_ckpt():
    found = glob.glob(os.path.join(OUT, "lilstory_step_*.json")) + glob.glob(RESUME_GLOB, recursive=True)
    return max(found, key=os.path.basename)[:-5] if found else None


# stacked (L, ...) tensors -> per-layer names that match the mlx checkpoint
def unstack(d, prefix):
    out = {}
    for k, v in d.items():
        v = np.asarray(v)
        if k.startswith("layers."):
            for i in range(N_LAYERS):
                out[f"{prefix}layers.{i}.{k[len('layers.'):]}"] = v[i]
        else:
            out[prefix + k] = v
    return out


def restack(d):
    out, per_layer = {}, {}
    for k, v in d.items():
        parts = k.split(".")
        if parts[0] == "layers":
            per_layer.setdefault(".".join(parts[2:]), {})[int(parts[1])] = v
        else:
            out[k] = jax.device_put(v, replicated)
    for k, layers in per_layer.items():
        if sorted(layers) != list(range(N_LAYERS)):
            raise KeyError(f"checkpoint has layers {sorted(layers)} for {k}, expected 0..{N_LAYERS - 1}")
        out["layers." + k] = jax.device_put(np.stack([layers[i] for i in range(N_LAYERS)]), replicated)
    missing = set(param_shapes()) - set(out)
    if missing:
        raise KeyError(f"checkpoint is missing {sorted(missing)[:3]}")
    return out


# tmp file + os.replace so a crash never leaves a half-written checkpoint
def write_pair(name, flat, meta):
    save_file(flat, name + ".safetensors.tmp")
    os.replace(name + ".safetensors.tmp", name + ".safetensors")
    with open(name + ".json", "w") as f:
        json.dump(meta, f)


# p: weights for tpu_convert.py, m:/v: adamw moments for resuming; only the newest pair is kept
def save_ckpt(params, opt, step, ds_state, history):
    flat = unstack(params, "p:")
    flat.update(unstack(opt["m"], "m:"))
    flat.update(unstack(opt["v"], "v:"))
    name = ckpt_name(step)
    write_pair(name, flat, {
        "step": step, "opt_step": int(opt["step"]), "dataset_state": ds_state,
        "datasets_version": datasets.__version__, "transformers_version": transformers.__version__,
        "dataset": f"{DATASET}/{DATASET_CFG}", "history": history,
        "model_config": {"D": D, "N_LAYERS": N_LAYERS, "N_HEADS": N_HEADS, "N_KV": N_KV, "D_FF": D_FF, "SEQ": SEQ},
    })
    del flat
    for old in glob.glob(os.path.join(OUT, "lilstory_step_*")):
        if not old.startswith(name):
            os.remove(old)
    print(f"[checkpoint] step {step} -> {name}.safetensors")


def load_ckpt(name):
    with open(name + ".json") as f:
        meta = json.load(f)
    flat = load_file(name + ".safetensors")
    params = restack({k[2:]: v for k, v in flat.items() if k.startswith("p:")})
    opt = {"step": jax.device_put(jnp.array(meta["opt_step"], jnp.int32), replicated),
           "m": restack({k[2:]: v for k, v in flat.items() if k.startswith("m:")}),
           "v": restack({k[2:]: v for k, v in flat.items() if k.startswith("v:")})}
    del flat
    return params, opt, meta


def ram_gb():
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / 2**30, vm.available / 2**30


def hbm_gb():
    ms = DEV.memory_stats() or {}
    return ms.get("peak_bytes_in_use", 0) / 2**30


def run_eval(params, eval_set):
    evals = jax.device_put(jnp.asarray(eval_set), replicated)
    return float(np.mean([float(eval_loss(params, evals[i:i + MICRO])) for i in range(0, len(evals), MICRO)]))


def train(watch, eval_set):
    watch.phase = "load/init weights"
    name = latest_ckpt()
    if name:
        params, opt, meta = load_ckpt(name)
        step, ds_state, history = meta["step"], meta["dataset_state"], meta["history"]
        print(f"resuming from step {step} ({name})")
    else:
        params = init_params(seed=0)
        opt = init_opt(params)
        step, ds_state, history = 0, None, {"train": [], "eval": []}
        print("starting from scratch")

    batches = Batches(ds_state)
    loss_sum, n_since = jnp.float32(0), 0
    out_of_time = False
    split = 1
    t_last = time.time()
    print(f"training steps {step} -> {TOTAL_STEPS} (first step compiles, give it a minute or two)")

    watch.phase = "compile + first step"
    try:
        while step < TOTAL_STEPS:
            if watch.low:
                raise MemoryError("host RAM nearly full")
            if time.time() - RUN_START > TIME_BUDGET_H * 3600:
                out_of_time = True
                break

            batch, ds_state = batches.get()
            # hbm overflow on first compile halves the micro-batch, same tokens per step
            while True:
                try:
                    dev_batch = jax.device_put(
                        jnp.asarray(batch.reshape(N_CHIPS, ACCUM * split, MICRO // split, SEQ + 1)), data_sharded)
                    params, opt, loss, gnorm = train_step(params, opt, dev_batch, jnp.float32(lr_at_step(step)))
                    break
                except jax.errors.JaxRuntimeError as e:
                    if "RESOURCE_EXHAUSTED" not in str(e) or watch.phase == "train" or (MICRO // split) % 2:
                        raise
                    split *= 2
                    print(f"hbm overflow while compiling, retrying with micro-batch {MICRO // split} x {ACCUM * split} accum")
            step += 1
            if watch.phase != "train":
                loss.block_until_ready()
                print(f"compiled, host process peak so far {watch.peak:.1f} GB")
                watch.phase, t_last = "train", time.time()
            else:
                loss_sum, n_since = loss_sum + loss, n_since + 1

            # q near 0 for long stretches means tokenization is the bottleneck, not the tpu
            if n_since and (step % LOG_EVERY == 0 or step == TOTAL_STEPS):
                avg = float(loss_sum) / n_since
                dt = (time.time() - t_last) / n_since
                eta = (TOTAL_STEPS - step) * dt / 3600
                used, _ = ram_gb()
                history["train"].append([step, avg])
                print(f"step {step:6d}/{TOTAL_STEPS}  loss={avg:.4f}  gnorm={float(gnorm):.2f}  lr={lr_at_step(step - 1):.2e}  "
                      f"{dt:.3f}s/step  {TOKENS_PER_STEP / dt:,.0f} tok/s  eta={eta:.1f}h  "
                      f"ram={used:.1f}GB  hbm_peak={hbm_gb():.1f}GB  q={batches.q.qsize()}")
                if not math.isfinite(avg):
                    raise FloatingPointError(f"loss went {avg} at step {step}; latest good checkpoint is {latest_ckpt()}")
                loss_sum, n_since, t_last = jnp.float32(0), 0, time.time()

            if step % EVAL_EVERY == 0 or step == TOTAL_STEPS:
                watch.phase = "eval"
                ev = run_eval(params, eval_set)
                history["eval"].append([step, ev])
                print(f"[eval] step {step}  held_out_loss={ev:.4f}  ppl={math.exp(ev):.1f}")
                watch.phase, t_last = "train", time.time()

            if step % CKPT_EVERY == 0 or step == TOTAL_STEPS:
                save_ckpt(params, opt, step, ds_state, history)
                t_last = time.time()
    # never save on nan, it would overwrite the last good checkpoint
    except FloatingPointError:
        raise
    except BaseException as e:
        if any(x.is_deleted() for x in jax.tree.leaves((params, opt))):
            print(f"{type(e).__name__} mid-step; latest saved checkpoint is {latest_ckpt()}")
        else:
            save_ckpt(params, opt, step, ds_state, history)
            print(f"{type(e).__name__} at step {step}; checkpoint saved")
        raise
    finally:
        batches.close()

    if out_of_time:
        save_ckpt(params, opt, step, ds_state, history)
        print(f"stopped at step {step}/{TOTAL_STEPS} after {TIME_BUDGET_H}h to beat the 9h limit. "
              "to continue: add this notebook's latest output as input, then save version again")
    return params, history


def plot_history(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr, ev = np.array(history["train"]), np.array(history["eval"])
    plt.figure(figsize=(8, 4))
    if len(tr):
        plt.plot(tr[:, 0], tr[:, 1], label=f"train ({LOG_EVERY}-step mean)", alpha=0.6)
    if len(ev):
        plt.plot(ev[:, 0], ev[:, 1], "o-", label="held-out")
    plt.xlabel("step"); plt.ylabel("loss"); plt.ylim(top=min(8, plt.ylim()[1])); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUT, "loss.png"), dpi=120, bbox_inches="tight")
    plt.close()


@jax.jit
def logits_at(params, ids, pos):
    return forward(params, ids, remat=False)[0, pos]


# fixed window so it compiles once; eos first because every training doc starts after one
def sample(params, prompt, n=80, temp=0.8, top_k=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = [EOS] + tok(prompt, add_special_tokens=False)["input_ids"][-(SAMPLE_LEN - n - 1):]
    buf = np.zeros((1, SAMPLE_LEN), np.int32)
    buf[0, :len(ids)] = ids
    end = len(ids)
    while end < min(len(ids) + n, SAMPLE_LEN):
        lg = np.asarray(logits_at(params, jnp.asarray(buf), end - 1)) / temp
        top = np.argpartition(lg, -top_k)[-top_k:]
        p = np.exp(lg[top] - lg[top].max())
        nxt = int(rng.choice(top, p=p / p.sum()))
        if nxt == EOS:
            break
        buf[0, end] = nxt
        end += 1
    return tok.decode(buf[0, 1:end])


# weights only (~1.7 gb); on the mac: python tpu_convert.py import <file>.safetensors --checkpoint-dir checkpoints_tpu
def export_weights():
    name = latest_ckpt()
    with open(name + ".json") as f:
        meta = json.load(f)
    export = os.path.join(OUT, f"lilstory_weights_step_{meta['step']:08d}")
    weights = {k: v for k, v in load_file(name + ".safetensors").items() if k.startswith("p:")}
    print(f"exporting {len(weights)} tensors, dtype {next(iter(weights.values())).dtype}")
    write_pair(export, weights, {**{k: meta[k] for k in ("step", "opt_step", "dataset_state")},
                                 "model_config": meta["model_config"]})


def main():
    n_params = sum(math.prod(s) for s in param_shapes().values())
    print(f"{n_params / 1e6:.1f}M params, {TOTAL_STEPS} steps, {TOTAL_STEPS * TOKENS_PER_STEP / 1e9:.2f}B training tokens, "
          f"{TOKENS_PER_STEP:,} tokens/step across {N_CHIPS} chips")

    watch = RamWatch()
    watch.phase = "build eval set"
    eval_set = build_eval_set()
    print(f"held-out: {eval_set.shape[0]} x {SEQ} tokens")

    params, history = train(watch, eval_set)
    plot_history(history)
    for prompt in ["The water cycle begins when", "In 1905, Albert Einstein", "The best way to learn a language is"]:
        print(sample(params, prompt), "\n---")
    export_weights()


if __name__ == "__main__":
    main()