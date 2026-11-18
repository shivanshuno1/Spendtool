"""One supplier_key per real supplier, however the name is typed.
Signals (merged with union-find): same vendor id, same compacted name (legal suffixes, punctuation and
spacing removed), near-identical compacted names (typos), and - when labels exist - the client's own key."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from .utils import norm_text, UnionFind

def compact_name(name, suffixes):
    toks = [t for t in norm_text(name).split() if t not in suffixes]
    if not toks:
        toks = norm_text(name).split()
    return "".join(toks)

def supplier_keys(df, cfg, guide=None):
    suff = set(cfg["legal_suffixes"])
    comp = df["supplier"].map(lambda x: compact_name(x, suff))
    vend = df["vendor_id"].fillna("").astype(str)
    node = np.where(comp != "", "n:" + comp, np.where(vend != "", "v:" + vend, "u:" + df["record_id"]))
    uniq = pd.Index(pd.unique(node))
    idx = {k: i for i, k in enumerate(uniq)}
    extra = {}
    def gnode(k):
        return extra.setdefault(k, len(uniq) + len(extra))
    pairs = []
    nid = pd.Series(node).map(idx).values
    first_by_vendor = {}
    for n, v in zip(nid, vend):
        if v:
            if v in first_by_vendor:
                pairs.append((first_by_vendor[v], n))
            else:
                first_by_vendor[v] = n
    thr = cfg.get("supplier_fuzzy_threshold", 92)
    blocks = {}
    for k in uniq:
        if k.startswith("n:") and len(k) - 2 >= 5:
            blocks.setdefault(k[2:5], []).append(k)
    for ks in blocks.values():
        for a in range(len(ks)):
            for c in range(a + 1, len(ks)):
                if fuzz.ratio(ks[a][2:], ks[c][2:]) >= thr:
                    pairs.append((idx[ks[a]], idx[ks[c]]))
    glabel = {}
    if guide is not None:
        for k in uniq:
            g = guide.sup_by_name.get(k[2:]) if k.startswith("n:") else None
            if g:
                pairs.append((idx[k], gnode("g:" + g))); glabel[gnode("g:" + g)] = g
        for v, n in first_by_vendor.items():
            g = guide.sup_by_vendor.get(v)
            if g:
                pairs.append((n, gnode("g:" + g))); glabel[gnode("g:" + g)] = g
    uf = UnionFind(len(uniq) + len(extra))
    for a, b in pairs:
        uf.union(a, b)
    root = np.array([uf.find(n) for n in nid])
    root_label = {uf.find(i): g for i, g in glabel.items()}      # cluster pinned to a client key
    lab = df["supplier"].fillna(df["vendor_id"]).fillna(df["record_id"]).astype(str)
    name_of = pd.DataFrame({"r": root, "n": lab}).groupby("r")["n"].agg(lambda s: s.value_counts().index[0])
    out = pd.Series(root).map(name_of)
    for r, g in root_label.items():
        out[root == r] = g
    return out.values
