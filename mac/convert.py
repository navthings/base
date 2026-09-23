import argparse
import json
from pathlib import Path

from safetensors.numpy import load_file, save_file
from transformers import AutoTokenizer, LlamaConfig

TOKENIZER = "hf-internal-testing/llama-tokenizer"
EOS = 2  # llama tokenizer's </s>; every training document starts right after it, so it doubles as bos

# training-code names -> transformers LlamaForCausalLM names; norms first so "attn." can't hit "attn_norm."
RENAMES = [("attn_norm.", "input_layernorm."), ("mlp_norm.", "post_attention_layernorm."), ("attn.", "self_attn.")]
LAYER_KEYS = ["input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj",
              "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def hf_name(key):
    key = key.removeprefix("p:")
    for old, new in RENAMES:
        key = key.replace(old, new)
    return "model." + key


def newest_weights():
    found = sorted((Path(__file__).parent / "weights").glob("*.safetensors"))
    if not found:
        raise SystemExit("no .safetensors in weights/; pass --weights explicitly")
    return found[-1]  # zero-padded step numbers sort correctly


def convert(weights_path, out_dir, with_tokenizer=True):
    weights_path, out_dir = Path(weights_path), Path(out_dir)
    cfg = json.loads(weights_path.with_suffix("").with_suffix(".json").read_text())["model_config"]
    tensors = {hf_name(k): v for k, v in load_file(weights_path).items() if k.startswith("p:")}

    expected = {"model.embed_tokens.weight", "model.norm.weight"} | {
        f"model.layers.{i}.{k}.weight" for i in range(cfg["N_LAYERS"]) for k in LAYER_KEYS}
    if set(tensors) != expected:
        raise KeyError(f"unexpected tensors: {sorted(set(tensors) ^ expected)[:5]}")

    config = LlamaConfig(
        vocab_size=tensors["model.embed_tokens.weight"].shape[0],
        hidden_size=cfg["D"], intermediate_size=cfg["D_FF"], num_hidden_layers=cfg["N_LAYERS"],
        num_attention_heads=cfg["N_HEADS"], num_key_value_heads=cfg["N_KV"],
        max_position_embeddings=cfg["SEQ"], rms_norm_eps=1e-5, rope_theta=10000.0, hidden_act="silu",
        tie_word_embeddings=True, bos_token_id=EOS, eos_token_id=EOS, architectures=["LlamaForCausalLM"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(out_dir)
    # no lm_head.weight: tie_word_embeddings makes transformers reuse embed_tokens for it on load
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    if with_tokenizer:
        AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(out_dir)
    return config


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="convert a TPU weights export into a transformers/PyTorch Llama folder")
    ap.add_argument("--weights", help="a *_weights_step_*.safetensors with its .json beside it; default: newest in weights/")
    ap.add_argument("--out", default=str(Path(__file__).parent / "hf"))
    ap.add_argument("--no-tokenizer", action="store_true", help="skip downloading and saving the tokenizer")
    args = ap.parse_args()

    src = args.weights or newest_weights()
    config = convert(src, args.out, with_tokenizer=not args.no_tokenizer)
    print(f"{src} -> {args.out} ({config.num_hidden_layers} layers, d={config.hidden_size})")
