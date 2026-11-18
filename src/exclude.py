"""Decide what counts as spend. Reason codes: CANCELLED, NOT_APPROVED, NON_PROCUREMENT (DUPLICATE is set in match.py)."""
import re
import warnings
import numpy as np
import pandas as pd

def _compile(pats):
    return [re.compile(p, re.I) for p in pats]

def apply_status(df, cfg):
    canc_re = re.compile(cfg["cancelled_regex"], re.I)
    reason = pd.Series("", index=df.index)
    for src in cfg["sources"]:
        st = src.get("status") or {}
        m = df.source == src["name"]
        if not m.any() or not st.get("approved"):
            continue                      # source has no status concept: everything passes
        approved = {str(x).strip().lower() for x in st["approved"]}
        cancelled = {str(x).strip().lower() for x in st.get("cancelled", [])}
        raw = df.loc[m, "status"].fillna("").astype(str).str.strip().str.lower()
        r = pd.Series("NOT_APPROVED", index=raw.index)
        r[raw.isin(approved)] = ""
        r[(raw == "")] = "" if cfg["missing_status"] == "approved" else "NOT_APPROVED"
        is_c = raw.isin(cancelled) | raw.map(lambda x: bool(canc_re.search(x)))
        r[is_c & ~raw.isin(approved)] = "CANCELLED"
        reason[m] = r
    return reason

def apply_nonprocurement(df, cfg):
    pats = {k: _compile(v) for k, v in cfg["nonproc_patterns"].items()}
    for rule in cfg.get("nonproc_rules", []):      # company-specific: {field, regex, name}
        pats.setdefault(rule.get("name", "custom"), []).append(re.compile(rule["regex"], re.I))
    fields = {"supplier": df["supplier"].fillna(""), "desc": df["desc"], "ctx": df["ctx"]}
    reason = pd.Series("", index=df.index)
    why = pd.Series("", index=df.index)
    for name, regs in pats.items():
        for rg in regs:
            for fname, ser in fields.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hit = ser.str.contains(rg, regex=True, na=False) & (reason == "")
                reason[hit] = "NON_PROCUREMENT"
                why[hit] = f"{name}:{fname}"
    for rule in cfg.get("nonproc_rules", []):      # restricted to one field, e.g. account code
        if rule.get("field") in df.columns:
            hit = df[rule["field"]].fillna("").astype(str).str.contains(rule["regex"], regex=True, flags=re.I) & (reason == "")
            reason[hit] = "NON_PROCUREMENT"
            why[hit] = rule.get("name", "rule") + ":" + rule["field"]
    if cfg.get("negative_amounts", "keep") == "exclude":
        hit = (df["spend_raw"] < 0) & (reason == "")
        reason[hit] = "NON_PROCUREMENT"; why[hit] = "negative_amount"
    return reason, why

def apply_exclusions(df, cfg):
    st = apply_status(df, cfg)
    npr, why = apply_nonprocurement(df, cfg)
    df["exclusion_reason"] = np.where(st != "", st, npr)       # CANCELLED / NOT_APPROVED win
    df["exclusion_note"] = np.where((st == "") & (npr != ""), why, "")
    return df
