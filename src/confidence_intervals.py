"""Exact (Clopper-Pearson) 95% intervals for the reported gate results."""
import csv
from pathlib import Path
from scipy.stats import beta

GATE_FILE = Path("experiments/gate_test_results.csv")
AUTHORISED = {"cow_1", "cow_2"}

def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi

def report(label, k, n):
    lo, hi = clopper_pearson(k, n)
    print(f"{label}: {k}/{n} ({k / n:.1%}); 95% CI {lo:.1%} to {hi:.1%}")

with GATE_FILE.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
opened = lambda r: r["gate_open"] == "True"
auth = [r for r in rows if r["actual"] in AUTHORISED]
enr = [r for r in rows if r["split"] == "known_test" and r["actual"] not in AUTHORISED]
unseen = [r for r in rows if r["split"] == "unknown_test"]
report("Authorised accepted", sum(map(opened, auth)), len(auth))
report("False accepts, enrolled unauthorised", sum(map(opened, enr)), len(enr))
report("False accepts, unseen", sum(map(opened, unseen)), len(unseen))
report("False accepts, all non-authorised", sum(map(opened, enr + unseen)), len(enr + unseen))