# base (tpu)

a 297m param llama-style model trained from scratch on a kaggle tpu v5e-8.

## layout

`tpu/` is training: `base.py`, the kaggle notebook, and its requirements. `mac/` is running the trained model: `convert.py`, `sample.py`, and the `weights/` and `hf/` folders they use (both gitignored).

## what it is

gqa, rope, rmsnorm, swiglu. 24 layers, d=1024, 16 query heads, 4 kv heads. llama tokenizer, 32k vocab. 6.1b tokens of fineweb-edu (sample-10BT), which is roughly chinchilla-optimal for this size.

11,634 steps of 524,288 tokens each, data parallel over all 8 chips. bf16 matmuls, and fp32 sometimes

## running it

reccomend running on kaggle, just import the notebook (in the repo) and change the accelerator to tpu, then save and run all

or heres the link to the kaggle notebook

https://www.kaggle.com/code/navneetdagdiya/base-tpu-kaggle

or paste it into a script kernel and hit save version -> save & run all.

it refuses to start if jax sees anything other than 8 tpu chips. kaggle sometimes hands out a broken allocation with 1 chip. restart and try again rather than training at 1/8th speed without noticing.

## running it on a mac

the weights aren't in git (1.2gb). grab `base_weights_step_*.safetensors` and its `.json` from the kaggle output tab and put both in `mac/weights/`, then:

```
cd mac
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python convert.py
python sample.py "The water cycle begins when"
```

`convert.py` turns the newest file in `mac/weights/` into a standard hugging face llama folder at `mac/hf/` (config, weights, tokenizer). the architecture maps onto `LlamaForCausalLM` exactly, so after that it's a normal transformers model and loads anywhere pytorch does:

```python
from transformers import LlamaForCausalLM, AutoTokenizer
model = LlamaForCausalLM.from_pretrained("mac/hf")
```

its logits match a from-scratch fp32 reimplementation of the jax forward pass to ~1e-4. `--weights` picks a specific file, `--out` changes the folder, `--no-tokenizer` skips the tokenizer download.

`sample.py` runs it on mps (cpu if mps isn't there), fp32, about 1.2gb of ram. flags: `--temp` (0.8), `--top-k` (40), `--max-tokens` (120), `--seed`, `--model` for a different folder, and `--rep-penalty` (1.1, set 1.0 to turn it off). the penalty is on by default because without it the model happily writes things like "english is a foreign language because it is a foreign language".

prompts start after `</s>`, not the tokenizer's usual `<s>`, because every training document started after one. if you use the model from your own code, do the same or the first few tokens come out worse.

it's a base model, so it continues text instead of answering questions. grammar is solid, facts are confidently made up.

## results

held-out loss 2.608 (perplexity 13.6) at step 11,000. the 8.4h time budget stopped it at step 11,043 of 11,634, by which point the lr was already at its floor, so the missing ~5% wouldn't have moved much. ~193k tok/s across the 8 chips the whole way.

## knobs

all at the top of the file. the ones worth touching:

- `MICRO, ACCUM`: 8x8. 16x4 ran out of hbm by about 400mb. if it overflows on first compile anyway, it halves the micro-batch and retries on its own.
- `TIME_BUDGET_H`: 8.4. lower it if checkpoint saves start getting cut close.
- `PEAK_LR`: 4e-4. gpt-3 350m used 3e-4 at a similar batch size, so this is slightly optimistic.
- `CKPT_EVERY`: 500 steps. frequent because nobody has measured how well this scales across 8 chips yet.

also try changing the end print statements to test the end model with different prompts

## logs

every 50 steps (`LOSS_EVERY`) prints the mean loss over those steps. every 250 steps (`LOG_EVERY`) that line also shows grad norm, lr, tok/s, eta, host ram, hbm peak and `q`. if `q` sits near 0, tokenization is the bottleneck and the tpu is waiting on the cpu.

## requirements

if you somehow have a 8 core tpu, and are running from scratch, install `tpu/requirements.txt` and run `tpu/base.py`. otherwise kaggle has everything downloaded already

for running the trained model on a mac it's `mac/requirements.txt` instead (torch, transformers, safetensors, numpy).
