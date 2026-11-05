"""Per-identity variation and duplicate check."""
import csv, hashlib, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr

SPLITS, GATE, TAU = "image_splits.csv", "gate_test_results.csv", 0.30

def rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

splits = rows(SPLITS)
ntrain = defaultdict(int)
for r in splits:
    if r["split"] == "train":
        ntrain[r["cow_id"]] += 1

conf, passed = defaultdict(list), defaultdict(list)
for r in rows(GATE):
    if r["split"] != "known_test":
        continue
    c = float(r["confidence"])
    conf[r["actual"]].append(c)
    passed[r["actual"]].append(r["actual"] == r["predicted"] and c >= TAU)

ids = sorted(conf)
mean_conf = [sum(conf[i]) / len(conf[i]) for i in ids]
pass_rate = [sum(passed[i]) / len(passed[i]) for i in ids]
n_train = [ntrain[i] for i in ids]
print("identities:", len(ids))
print("Spearman train-count vs mean confidence: rho=%.2f p=%.1e" % spearmanr(n_train, mean_conf))
print("Spearman train-count vs pass rate:       rho=%.2f p=%.1e" % spearmanr(n_train, pass_rate))

if len(sys.argv) > 1 and sys.argv[1] == "--hash":   # run where data/ exists
    seen = defaultdict(list)
    for r in splits:
        p = Path(r["path"].replace("\\", "/"))
        seen[hashlib.md5(p.read_bytes()).hexdigest()].append((r["split"], r["path"]))
    dups = [v for v in seen.values() if len(v) > 1]
    print("duplicate groups:", len(dups))
    for g in dups:
        print(g)