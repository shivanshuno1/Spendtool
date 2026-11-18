"""Score an output file against labels_A.csv (ERP rows only: that is where labels exist).
   python eval/eval_A.py --out outputs/x__A.csv --labels data/A/labels_A.csv [--hidden-ids outputs/x__A_hidden_ids.txt]"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from eval.metrics import pair_prf

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--labels", required=True)
ap.add_argument("--hidden-ids"); a = ap.parse_args()
o = pd.read_csv(a.out, dtype={"l1": str, "l2": str, "l3": str, "l4": str}, keep_default_na=False)
lab = pd.read_csv(a.labels)
j = o.merge(lab, on="record_id")
if a.hidden_ids:
    hid = set(open(a.hidden_ids).read().split()); j = j[j.record_id.isin(hid)]
    print(f"scoring only the {len(j)} records whose labels were hidden from the tool")
j = j[j.include == 1]
print(f"labelled included ERP rows: {len(j)}")
p, r, f = pair_prf(j.supplier_key_x.values, j.supplier_key_y.values); print(f"supplier dedup  P={p:.3f} R={r:.3f} F1={f:.3f}")
v = j.dropna(subset=["item_key_y"]); p, r, f = pair_prf(v.item_key_x.values, v.item_key_y.values); print(f"item dedup      P={p:.3f} R={r:.3f} F1={f:.3f}")
w = j.spend.abs(); 
for k, c in enumerate(["category_l1", "category_l2", "category_l3"], 1):
    ok = (j[f"l{k}"] == j[c]).values
    print(f"L{k} accuracy   lines={ok.mean():.3f}  spend-weighted={(ok * w).sum() / w.sum():.3f}")
print("mean confidence", round(j.confidence.mean(), 3))
