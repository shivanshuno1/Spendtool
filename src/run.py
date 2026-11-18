"""ONE COMMAND:  python -m src.run --config configs/A.yaml --out outputs/yourname__A.csv"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
from .config import load_config
from .ingest import ingest
from .clean import clean
from .exclude import apply_exclusions
from .match import run_match
from .dedup_supplier import supplier_keys
from .dedup_item import item_keys
from .taxonomy import load_tree
from .labels import load_guide
from .classify import classify
from .output import build_output, build_audit

def run(config, out, use_labels=True, holdout=0.0, seed=0, llm=None, log=print):
    t0 = time.time()
    cfg = load_config(config)
    cfg["_out_dir"] = os.path.dirname(os.path.abspath(out))
    if llm is not None:
        cfg["classify"]["llm"]["enabled"] = llm
    step = lambda m: log(f"[{time.time()-t0:6.1f}s] {m}")
    df = ingest(cfg);                                  step(f"ingested {len(df)} records from {df.source.nunique()} source(s)")
    df = clean(df, cfg);                               step("amounts/dates cleaned")
    df = apply_exclusions(df, cfg);                    step("status / non-procurement exclusions")
    df = run_match(df, cfg);                           step("cross-system matching + duplicates")
    guide = load_guide(cfg, df, use_labels) if use_labels else None
    hidden = set()
    if guide is not None and holdout > 0:               # evaluation helper: hide a share of the labels
        rng = np.random.RandomState(seed)
        ids = guide.cats["record_id"].values
        hidden = set(rng.choice(ids, int(len(ids) * holdout), replace=False))
        open(out.replace('.csv', '_hidden_ids.txt'), 'w').write('\n'.join(sorted(hidden)))
        df_g = df[~df.record_id.isin(hidden)]
        guide = load_guide(cfg, df_g, True)
    df["supplier_key"] = supplier_keys(df, cfg, guide)
    df["item_key"] = item_keys(df, cfg, guide)
    rid = pd.Series(df.index, index=df.record_id)       # matched record inherits its partner's keys
    m = df["dup_of"] != ""
    partner = rid.reindex(df.loc[m, "dup_of"]).values
    ok = ~pd.isna(partner)
    idx = df.index[m][ok]
    df.loc[idx, "supplier_key"] = df.loc[partner[ok].astype(int), "supplier_key"].values
    df.loc[idx, "item_key"] = df.loc[partner[ok].astype(int), "item_key"].values
    step(f"supplier_keys={df.supplier_key.nunique()} item_keys={df.item_key.nunique()}")
    tree = load_tree(cfg);                             step(f"tree: {len(tree)} leaf categories, {tree.depth} levels")
    cls = classify(df, tree, cfg, guide, log=log);     step("classified")
    res = build_output(df, cls, tree)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    res.iloc[:, :11].to_csv(out, index=False)
    build_audit(df, cls).to_csv(out.replace(".csv", "_audit.csv"), index=False)
    inc = res[res.include == 1]
    step(f"wrote {out}: {len(res)} rows, {len(inc)} included, total spend {inc.spend.sum():,.2f} {cfg['reporting_currency']}")
    print(res.exclusion_reason.replace("", "INCLUDED").value_counts().to_string())
    return res, df, cls, tree, hidden

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-labels", action="store_true", help="ignore labels file (zero-label mode)")
    ap.add_argument("--holdout", type=float, default=0.0, help="hide this share of labels (evaluation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", action="store_true", help="enable LLM labelling (needs ANTHROPIC_API_KEY)")
    a = ap.parse_args()
    run(a.config, a.out, not a.no_labels, a.holdout, a.seed, True if a.llm else None)

if __name__ == "__main__":
    main()
