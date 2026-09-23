# lilbase

a 297m param llama-style base model, trained from scratch on a kaggle tpu v5e-8.

it's a base model, so it continues text instead of answering questions. give it the start of a sentence, not a question. grammar is solid, facts are confidently made up.

```
ollama run navthings/lilbase "The water cycle begins when"
```

## tags

| tag | size | notes |
|---|---|---|
| `latest` / `q8_0` | 379mb | same perplexity as f16 |
| `q4_k_m` | 273mb | ~0.8% higher perplexity, smallest |
| `f16` | 594mb | unquantized |

## the model

gqa, rope, rmsnorm, swiglu. 24 layers, d=1024, 16 query heads, 4 kv heads, 1024 context. llama tokenizer, 32k vocab.

trained on 6.1b tokens of fineweb-edu (sample-10BT), roughly chinchilla-optimal for this size. 11,043 steps of 524k tokens, data parallel over 8 tpu chips, about 8.4 hours. held-out loss 2.608 (perplexity 13.6).

## defaults

temperature 0.8, top-k 40, repeat penalty 1.1, 200 tokens. the repeat penalty is there because without it the model loops ("english is a foreign language because it is a foreign language"). set it to 1.0 if you want the raw model.

there's no chat template. whatever you type is the prompt, exactly.

## code

https://github.com/navthings/lilbase

kaggle notebook: https://www.kaggle.com/code/navneetdagdiya/base-tpu-kaggle
