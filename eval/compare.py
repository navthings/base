import json, math, os, re, time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
LILBASE = os.path.join(HERE, "..", "mac", "hf")
MODELS = {"gpt2 (124m)": "openai-community/gpt2", "lilbase (297m)": LILBASE, "gpt2-medium (355m)": "openai-community/gpt2-medium"}
N_HELLA, N_DOCS, CTX = 2000, 300, 1024
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def log(*a):
    print(*a, flush=True)


def hella_clean(t):
    t = t.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", t).replace("  ", " ")


def load_tasks():
    tasks = {}
    lam = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    tasks["lambada"] = [(t.rsplit(" ", 1)[0], " " + t.rsplit(" ", 1)[1]) for t in lam["text"]]

    hs = load_dataset("Rowan/hellaswag", split="validation", revision="refs/convert/parquet").select(range(N_HELLA))
    tasks["hellaswag"] = [(hella_clean(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()),
                           [" " + hella_clean(e) for e in r["endings"]], int(r["label"])) for r in hs]

    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    tasks["arc_easy"] = [("Question: " + r["question"] + "\nAnswer:", [" " + c for c in r["choices"]["text"]],
                          r["choices"]["label"].index(r["answerKey"])) for r in arc]

    # first 1500 docs of the stream were held out of training, so these were never seen
    fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    tasks["fineweb_edu"] = [ex["text"] for ex in fw.take(N_DOCS)]

    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    tasks["wikitext"] = ["".join(wt["text"])]
    return tasks


class Scorer:
    def __init__(self, path):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(dev).eval()
        self.eos = self.tok.eos_token_id

    def enc(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    # every doc in training started right after an eos, so prefix one for both models
    def pair(self, ctx, cont):
        whole, c = self.enc(ctx + cont), self.enc(ctx)
        cont_ids = whole[len(c):] if whole[:len(c)] == c else self.enc(cont)
        ids = ([self.eos] + c + cont_ids)[-CTX:]
        return ids, len(cont_ids)

    # gather on device, pulling full-vocab logits back to cpu was the slow part
    @torch.no_grad()
    def token_lls(self, seqs, spans):
        n = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), n, dtype=torch.long)
        mask = torch.zeros(len(seqs), n, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)], mask[i, :len(s)] = torch.tensor(s), 1
        x, mask = x.to(dev), mask.to(dev)
        lp = self.model(x, attention_mask=mask).logits.float().log_softmax(-1)
        tok_lp = lp[:, :-1].gather(-1, x[:, 1:, None])[..., 0]
        hit = lp[:, :-1].argmax(-1) == x[:, 1:]
        tok_lp, hit = tok_lp.cpu(), hit.cpu()
        return [(tok_lp[i, a - 1:b - 1].sum().item(), bool(hit[i, a - 1:b - 1].all())) for i, (a, b) in enumerate(spans)]

    def score(self, pairs):
        seqs, lens = zip(*[self.pair(c, t) for c, t in pairs])
        return self.token_lls(list(seqs), [(len(s) - k, len(s)) for s, k in zip(seqs, lens)])

    def choice_acc(self, items):
        hits = 0
        for ctx, choices, label in items:
            lls = [ll for ll, _ in self.score([(ctx, c) for c in choices])]
            norm = [ll / len(c) for ll, c in zip(lls, choices)]
            hits += int(max(range(len(norm)), key=norm.__getitem__) == label)
        return hits / len(items)

    def lambada(self, items, bs=16):
        hits, nll, ntok = 0, 0.0, 0
        for i in range(0, len(items), bs):
            for ll, g in self.score(items[i:i + bs]):
                hits += g
                nll -= ll
                ntok += 1
        return hits / len(items), math.exp(nll / ntok)

    def bpb(self, texts, bs=4):
        chunks, nbytes = [], 0
        for t in texts:
            ids = [self.eos] + self.enc(t)
            chunks += [ids[i:i + CTX] for i in range(0, len(ids) - 1, CTX - 1)]
            nbytes += len(t.encode())
        nll = 0.0
        for i in range(0, len(chunks), bs):
            batch = chunks[i:i + bs]
            nll -= sum(ll for ll, _ in self.token_lls(batch, [(1, len(c)) for c in batch]))
        return nll / math.log(2) / nbytes


def main():
    t0 = time.time()
    tasks = load_tasks()
    log(f"data loaded in {time.time() - t0:.0f}s: " + ", ".join(f"{k}={len(v)}" for k, v in tasks.items()))
    results = {}
    for name, path in MODELS.items():
        s, r = Scorer(path), {}
        t = time.time()
        r["lambada_acc"], r["lambada_ppl"] = s.lambada(tasks["lambada"])
        log(f"  lambada {time.time() - t:.0f}s")
        r["hellaswag_acc_norm"] = s.choice_acc(tasks["hellaswag"])
        log(f"  hellaswag {time.time() - t:.0f}s")
        r["arc_easy_acc_norm"] = s.choice_acc(tasks["arc_easy"])
        log(f"  arc {time.time() - t:.0f}s")
        r["fineweb_edu_bpb"] = s.bpb(tasks["fineweb_edu"])
        log(f"  fineweb {time.time() - t:.0f}s")
        r["wikitext_bpb"] = s.bpb(tasks["wikitext"])
        results[name] = r
        log(f"{name} ({time.time() - t:.0f}s): " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))
        del s
        if dev == "mps":
            torch.mps.empty_cache()
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump({"n_hellaswag": N_HELLA, "n_fineweb_docs": N_DOCS, "results": results}, f, indent=2)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
