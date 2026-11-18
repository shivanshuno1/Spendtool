import numpy as np
import pandas as pd

COLS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
        "l1", "l2", "l3", "l4", "confidence"]
_EXCL_CONF = {"CANCELLED": 1.0, "NOT_APPROVED": 1.0, "NON_PROCUREMENT": 0.7}
_RULE_CONF = {"po+amount": 0.95, "amount+text": 0.75, "reexport": 0.95}

def build_output(df, cls, tree):
    out = pd.DataFrame({"record_id": df["record_id"]})
    inc = df["exclusion_reason"] == ""
    out["include"] = inc.astype(int)
    out["exclusion_reason"] = df["exclusion_reason"]

    out["supplier_key"] = df["supplier_key"]
    out["item_key"] = df["item_key"]
    out["spend"] = np.where(inc, df["spend_raw"].fillna(0.0), 0.0).round(2)
    nlev = max(4, tree.depth)
    for i in range(nlev):
        out[f"l{i+1}"] = ""
    out["confidence"] = 0.0
    idx = cls.index
    names = [tree.names(int(l)) for l in cls["leaf"].values]
    for i in range(nlev):
        out.loc[idx, f"l{i+1}"] = [n[i] if i < len(n) else "" for n in names]
    out.loc[idx, "confidence"] = cls["confidence"].astype(float).values
    ex = ~inc
    conf = df.loc[ex, "exclusion_reason"].map(_EXCL_CONF)
    conf = conf.fillna(df.loc[ex, "match_rule"].map(_RULE_CONF))

    # Ensure every confidence value is numeric
    conf = pd.to_numeric(conf, errors="coerce").fillna(0.8)

    out.loc[ex, "confidence"] = conf.astype(float).values
    return out[COLS + [c for c in out.columns if c not in COLS]]

def build_audit(df, cls):
    a = df[["record_id", "source", "exclusion_reason", "exclusion_note", "dup_of", "match_rule",
            "amount_num", "vat_used", "net_local", "cur", "fx_rate", "fx_source", "spend_raw",
            "date_parsed", "date_flag", "amount_flag"]].copy()
    a["class_method"] = cls["method"].reindex(df.index)
    return a
