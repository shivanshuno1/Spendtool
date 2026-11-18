"""One item_key per real item.
 - item code present -> the code is the identity (same description with different codes stays separate)
 - no code -> normalised description; near-identical descriptions merge only if their numbers are identical
 - when labels exist the client's own item key is used for known codes/descriptions."""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from .utils import norm_text, UnionFind

def norm_item(desc, stop):
    return " ".join(t for t in norm_text(desc).split() if t not in stop)

def item_keys(df, cfg, guide=None):
    stop = set(cfg["item_stopwords"])
    d = df["desc"].map(lambda x: norm_item(x, stop))
    code = df["item_code"].fillna("").astype(str).str.strip().str.lower()
    node = np.where(code != "", "c:" + code, np.where(d != "", "d:" + d, "u:" + df["record_id"]))
    uniq = pd.Index(pd.unique(node))
    idx = {k: i for i, k in enumerate(uniq)}
    nid = pd.Series(node).map(idx).values
    pairs, extra, glabel = [], {}, {}
    def gnode(k):
        return extra.setdefault(k, len(uniq) + len(extra))
    # fuzzy merge of code-less descriptions (numbers must be identical)
    thr = cfg.get("item_fuzzy_threshold", 93)
    blocks = {}
    for k in uniq:
        if k.startswith("d:"):
            t = k[2:]
            nums = tuple(sorted(re.findall(r"\d+", t)))
            first = (re.findall(r"[a-z]+", t) or [""])[0]
            blocks.setdefault((nums, first), []).append(k)
    for ks in blocks.values():
        if 1 < len(ks) <= 400:
            for a in range(len(ks)):
                for c in range(a + 1, len(ks)):
                    if fuzz.token_sort_ratio(ks[a][2:], ks[c][2:]) >= thr:
                        pairs.append((idx[ks[a]], idx[ks[c]]))
    if guide is not None:
        for k in uniq:
            g = guide.item_by_code.get(k[2:]) if k.startswith("c:") else guide.item_by_desc.get(k[2:]) if k.startswith("d:") else None
            if g:
                pairs.append((idx[k], gnode("g:" + g))); glabel[gnode("g:" + g)] = g
    uf = UnionFind(len(uniq) + len(extra))
    for a, b in pairs:
        uf.union(a, b)
    root = np.array([uf.find(n) for n in nid])
    root_label = {uf.find(i): g for i, g in glabel.items()}
    lab = np.where(code != "", "code:" + code, np.where(d != "", d, df["record_id"]))
    name_of = pd.DataFrame({"r": root, "n": lab}).groupby("r")["n"].agg(lambda s: s.value_counts().index[0])
    out = pd.Series(root).map(name_of)
    for r, g in root_label.items():
        out[root == r] = g
    return out.values
