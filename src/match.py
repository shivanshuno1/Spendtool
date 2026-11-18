"""Count each real purchase once.
 1) exact re-exports inside the secondary system (same invoice + text + net amount) -> DUPLICATE
 2) secondary record matched to a primary (e.g. ERP) record -> DUPLICATE (primary is counted)
    route A: same PO reference + amount within tolerance (best amount/text score, one-to-one)
    route B: no usable PO: very close amount AND similar text or supplier
"""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from .utils import norm_text, po_norm

def _prep(df, cfg):
    m = cfg["match"]
    rg = m.get("po_regex", r"(\d{6,})")
    df["po_norm"] = df["po"].map(lambda x: po_norm(x, rg))
    df["desc_norm"] = df["desc"].map(norm_text)
    df["sup_norm"] = df["supplier"].map(norm_text).str.replace(" ", "", regex=False)
    return df

def run_match(df, cfg):
    df["dup_of"] = ""
    df["match_rule"] = ""
    m = cfg["match"]
    df = _prep(df, cfg)
    if not m.get("enabled"):
        return df
    prim, sec = m["primary"], m["secondary"]

    # ---- 1) re-exports inside the secondary system -------------------------------
    S = df[(df.source == sec) & (df.exclusion_reason == "")].copy()
    S["_k_inv"] = S["invoice_no"].fillna("").astype(str).str.strip().str.lower()
    S["_k_amt"] = S["spend_raw"].round(2)
    order = {str(x).lower(): i for i, x in enumerate(
        next((s for s in cfg["sources"] if s["name"] == sec), {}).get("status", {}).get("approved", []) [::-1])}
    S["_rank"] = S["status"].fillna("").astype(str).str.lower().map(order).fillna(99)
    S = S.sort_values(["_rank", "record_id"])
    key = ["_k_inv", "desc_norm", "_k_amt"]
    first = S.drop_duplicates(key, keep="first")
    first_id = first.set_index(key)["record_id"]
    dups = S[S.duplicated(key, keep="first")]
    for i, r in dups.iterrows():
        df.at[i, "exclusion_reason"] = "DUPLICATE"
        df.at[i, "dup_of"] = first_id[(r["_k_inv"], r["desc_norm"], r["_k_amt"])]
        df.at[i, "match_rule"] = "reexport"

    # ---- 2) secondary vs primary -----------------------------------------------------
    P = df[(df.source == prim) & (df.exclusion_reason == "")]
    S = df[(df.source == sec) & (df.exclusion_reason == "")]
    by_po = {}
    for j, po in P["po_norm"].items():
        if po:
            by_po.setdefault(po, []).append(j)
    p_amt = P["spend_raw"].to_dict()
    p_desc = P["desc_norm"].to_dict()
    p_sup = P["sup_norm"].to_dict()
    tol_r, tol_a = m["amount_tol_rel"], m["amount_tol_abs"]
    pairs = []
    # route B index: amounts rounded to whole units
    amt_idx = {}
    for j, a in p_amt.items():
        if pd.notna(a):
            amt_idx.setdefault(int(round(a)), []).append(j)
    for i, r in S.iterrows():
        a = r["spend_raw"]
        if pd.isna(a):
            continue
        sd = r["desc_norm"]
        found = False
        for j in by_po.get(r["po_norm"], []) if r["po_norm"] else []:
            pa = p_amt[j]
            d = abs(pa - a)
            tol = max(tol_a, tol_r * abs(pa))
            if d <= tol:
                sim = fuzz.token_set_ratio(sd, p_desc[j]) / 100.0 if sd and p_desc[j] else 0.0
                score = 1.0 + (1 - d / (tol + 1e-9)) * 0.5 + sim
                pairs.append((score, i, j, "po+amount"))
                found = True
        if found:
            continue
        for k in (int(round(a)) - 1, int(round(a)), int(round(a)) + 1):    # route B
            for j in amt_idx.get(k, []):
                pa = p_amt[j]
                if abs(pa - a) <= max(tol_a, m["nopo_tol_rel"] * abs(pa)):
                    sim = fuzz.token_set_ratio(sd, p_desc[j]) / 100.0 if sd and p_desc[j] else 0.0
                    ssim = fuzz.ratio(r["sup_norm"], p_sup[j]) / 100.0 if r["sup_norm"] and p_sup[j] else 0.0
                    if sim >= m["desc_min_sim"] or ssim >= 0.9:
                        pairs.append((sim + ssim * 0.3, i, j, "amount+text"))
    pairs.sort(key=lambda t: -t[0])
    used_s, used_p = set(), set()
    for score, i, j, rule in pairs:
        if i in used_s or j in used_p:
            continue
        used_s.add(i); used_p.add(j)
        df.at[i, "exclusion_reason"] = "DUPLICATE"
        df.at[i, "dup_of"] = df.at[j, "record_id"]
        df.at[i, "match_rule"] = rule
    return df
