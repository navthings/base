# base (tpu)

a 297m param llama-style model trained from scratch on a kaggle tpu v5e-8.

## what it is

gqa, rope, rmsnorm, swiglu. 24 layers, d=1024, 16 query heads, 4 kv heads. llama tokenizer, 32k vocab. 6.1b tokens of fineweb-edu (sample-10BT), which is roughly chinchilla-optimal for this size.

11,634 steps of 524,288 tokens each, data parallel over all 8 chips. bf16 matmuls, and fp32 sometimes

## running it

reccomend running on kaggle, just import the notebook (in the repo) and change the accelerator to tpu, then save and run all

or heres the link to the kaggle notebook

https://www.kaggle.com/code/navneetdagdiya/base-tpu-kaggle

or paste it into a script kernel and hit save version -> save & run all.

it refuses to start if jax sees anything other than 8 tpu chips. kaggle sometimes hands out a broken allocation with 1 chip. restart and try again rather than training at 1/8th speed without noticing.

## knobs

all at the top of the file. the ones worth touching:

- `MICRO, ACCUM`: 8x8. 16x4 ran out of hbm by about 400mb. if it overflows on first compile anyway, it halves the micro-batch and retries on its own.
- `TIME_BUDGET_H`: 8.4. lower it if checkpoint saves start getting cut close.
- `PEAK_LR`: 4e-4. gpt-3 350m used 3e-4 at a similar batch size, so this is slightly optimistic.
- `CKPT_EVERY`: 500 steps. frequent because nobody has measured how well this scales across 8 chips yet.

also if you want to change the end print statements to test the end model with different prompts

## logs

every 50 steps (`LOSS_EVERY`) prints the mean loss over those steps. every 250 steps (`LOG_EVERY`) that line also shows grad norm, lr, tok/s, eta, host ram, hbm peak and `q`. if `q` sits near 0, tokenization is the bottleneck and the tpu is waiting on the cpu.

## requirements

if you somehow have a 8 core tpu, and are running from scratch, download the requirements and run base.py. otherwise kaggle has everything downloaded already
