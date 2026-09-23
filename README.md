# lilbase (tpu)

a 297m param llama-style model trained from scratch on a kaggle tpu v5e-8.

## try it

it's on ollama: https://ollama.com/navthings/lilbase

```
ollama run navthings/lilbase "The water cycle begins when"
```

tags are `latest` (q8_0, 379mb, no measurable loss vs f16), `q4_k_m` (273mb, ~0.8% worse perplexity) and `f16` (594mb). it's a base model, so give it the start of a sentence, not a question.

## examples

what it does best is textbook-style openings and lists, since that's most of fineweb-edu. it keeps a format going on its own. facts are where it slips.

the bold part is the prompt, the rest is the model.

**Here are some tips for studying effectively:**

**1.** Keep your schedule

If you're not sure how to study effectively, remember that studying is a mental exercise. You need to spend some time working on the material until you can get it done. Don't worry if you're in a hurry. It will not be finished. Take some time off, relax and be more organized.

2\. Use flash cards

It is important for you to know how to study well, but you should also learn some tips on how to study effectively. Using flash cards can help you to study better.

3\. Study early in the day

Most of us spend a lot of time during the day studying. So you should try to study before you have any breakfast.

---

**The heart is a muscular organ that** has two lobes, one on the right side and one on the left. It pumps blood into the body by way of the arteries. The right side is connected to the lungs and the left side connects with the heart.

The heart contains two main chambers, the ventricles (smaller) and the mitral valve (large). The mitral valve allows oxygenated blood to pass from the left atrium into the right ventricle.

## what it is

gqa, rope, rmsnorm, swiglu. 24 layers, d=1024, 16 query heads, 4 kv heads. llama tokenizer, 32k vocab. 6.1b tokens of fineweb-edu (sample-10BT), which is roughly chinchilla-optimal for this size.

11,634 steps of 524,288 tokens each, data parallel over all 8 chips. bf16 matmuls, and fp32 sometimes

## running it

reccomend running on kaggle, just import the notebook (in the repo) and change the accelerator to tpu, then save and run all

or heres the link to the kaggle notebook

https://www.kaggle.com/code/navneetdagdiya/base-tpu-kaggle

or paste it into a script kernel and hit save version -> save & run all.

it refuses to start if jax sees anything other than 8 tpu chips. kaggle sometimes hands out a broken allocation with 1 chip. restart and try again rather than training at 1/8th speed without noticing.

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

if you somehow have a 8 core tpu, and are running from scratch, install `tpu/requirements.txt` and run `tpu/lilbase.py`. otherwise kaggle has everything downloaded already
