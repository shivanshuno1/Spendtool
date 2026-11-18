"""Optional labelled examples (the client's golden keys / categories). The pipeline works without them;
when present they (a) pin supplier/item keys to the client's own keys and (b) train the classifier."""
import os
import pandas as pd
from .config import resolve
from .dedup_supplier import compact_name
from .dedup_item import norm_item

class Guide:
    def __init__(self):
        self.sup_by_name, self.sup_by_vendor = {}, {}
        self.item_by_code, self.item_by_desc = {}, {}
        self.cats = None            # DataFrame: record_id, l1..lN (labelled rows)
        self.n = 0

def _top(df, key, val):
    d = df[[key, val]].dropna()
    d = d[d[key].astype(str) != ""]
    g = d.groupby(key)[val].agg(lambda s: s.value_counts().index[0])
    uniq = d.groupby(key)[val].nunique()
    return g.to_dict(), set(uniq[uniq > 1].index)

def load_guide(cfg, df, enabled=True):
    lc = cfg.get("labels")
    if not enabled or not lc:
        return None
    path = resolve(cfg, lc["file"])
    if not os.path.exists(path):
        return None
    lab = pd.read_csv(path, dtype=str)
    lab = lab.rename(columns={lc.get("id_col", "record_id"): "record_id"})
    levels = lc.get("level_cols", [])
    j = df.merge(lab, on="record_id", how="inner")
    if j.empty:
        return None
    g = Guide()
    g.n = len(j)
    suff = set(cfg["legal_suffixes"]); stop = set(cfg["item_stopwords"])
    if lc.get("supplier_col") in j:
        j["_cn"] = j["supplier"].map(lambda x: compact_name(x, suff))
        g.sup_by_name, _ = _top(j[j["_cn"] != ""], "_cn", lc["supplier_col"])
        g.sup_by_vendor, _ = _top(j, "vendor_id", lc["supplier_col"])
    if lc.get("item_col") in j:
        j["_ic"] = j["item_code"].fillna("").astype(str).str.strip().str.lower()
        g.item_by_code, _ = _top(j[j["_ic"] != ""], "_ic", lc["item_col"])
        j["_nd"] = j["desc"].map(lambda x: norm_item(x, stop))
        m, amb = _top(j[j["_nd"] != ""], "_nd", lc["item_col"])
        g.item_by_desc = {k: v for k, v in m.items() if k not in amb}
    if levels:
        keep = j[["record_id"] + [c for c in levels if c in j]].dropna(subset=[levels[0]])
        keep.columns = ["record_id"] + [f"l{i+1}" for i in range(len(keep.columns) - 1)]
        g.cats = keep
    return g
