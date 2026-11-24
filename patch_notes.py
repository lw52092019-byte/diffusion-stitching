"""One-off: the 13 kappa=1 configs saved on 2026-11-24 02:22-02:36 ran with the GPU to
themselves, but carry the old default note "shared the GPU". Correct that note."""
import json

p = "results/raw_kappa1.json"
raw = json.load(open(p))
n = 0
for k, v in raw.items():
    if isinstance(v, dict) and "shared the GPU" in str(v.get("note", "")):
        v["note"] = "GPU used exclusively by this run (serial, 2026-11-24 02:22-02:36)"
        n += 1
json.dump(raw, open(p, "w"), indent=1)
print("patched", n, "entries")
