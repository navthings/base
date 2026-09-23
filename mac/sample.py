import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="The water cycle begins when")
    ap.add_argument("--model", default=str(Path(__file__).parent / "hf"), help="folder written by convert.py")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--rep-penalty", type=float, default=1.1, help="1.0 = off; >1 discourages repetition loops")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained(args.model).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.model)

    # training documents start right after </s>, so prompts do too, not after the tokenizer's default <s>
    ids = torch.tensor([[tok.eos_token_id] + tok(args.prompt, add_special_tokens=False)["input_ids"]], device=device)
    torch.manual_seed(args.seed)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=args.temp,
                             top_k=args.top_k, repetition_penalty=args.rep_penalty,
                             max_new_tokens=args.max_tokens, pad_token_id=tok.eos_token_id)
    print(tok.decode(out[0, 1:], skip_special_tokens=True))
