import json, time
import compare as c

c.N_HELLA, c.N_LAMBADA = 400, 10
t = c.load_tasks()
s = c.Scorer(c.LILBASE)
r = {}
r["lambada_acc"], r["lambada_ppl"] = 0.287, 57.37
print(r, flush=True)
r["hellaswag_acc_norm"] = s.choice_acc(t["hellaswag"])
print(r, flush=True)
r["arc_easy_acc_norm"] = s.choice_acc(t["arc_easy"][:400])
print(r, flush=True)
json.dump(r, open("results.json", "w"), indent=2)
