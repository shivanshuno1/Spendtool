"""Tree-driven classification of every included line (works with ZERO labels).

Layers (each optional, all driven by the tree file read at run time):
  1. zero-shot: TF-IDF (words + char n-grams) similarity between line text and category paths
     (+ sentence-embedding similarity if `classify.use_embeddings` and the package/model is available)
  2. LLM labels for the highest-spend unique items (only if enabled AND ANTHROPIC_API_KEY is set)
  3. self-training: pseudo-label the confident lines, learn text + supplier + spend-type -> category,
     re-predict everything (spreads the signal to lines whose own text says nothing)
  4. supervised model on the client's labelled lines, when a labels file is configured and present
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize, OneHotEncoder
from . import llm as llm_mod

warnings.filterwarnings("ignore")

def _softmax(S, t):
    z = S / t
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)

def _embed_sim(texts, cats, model_name):
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name)
        a = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        b = m.encode(cats, normalize_embeddings=True, show_progress_bar=False)
        return a @ b.T
    except Exception:
        return None

def _fit_lr(F, y, w, C):
    classes = np.unique(y)
    if len(classes) < 2:
        return None
    clf = LogisticRegression(C=C, max_iter=200)
    clf.fit(F, y, sample_weight=w)
    return clf

def _proba(clf, F, K):
    P = np.zeros((F.shape[0], K))
    P[:, clf.classes_] = clf.predict_proba(F)
    return P

def classify(df, tree, cfg, guide=None, log=print):
    cc = cfg["classify"]
    K = len(tree)
    inc = df[df.exclusion_reason == ""]
    out = pd.DataFrame(index=inc.index, columns=["leaf", "confidence", "method"])
    if inc.empty:
        return out
    if K == 1:
        out["leaf"], out["confidence"], out["method"] = 0, 1.0, "single_category"
        return out

    txt = inc["desc"].where(inc["desc"].str.strip() != "", inc["ctx"]).fillna("").str.lower()
    key = pd.DataFrame({"t": txt.values, "s": inc["supplier_key"].fillna("").values, "c": inc["ctx"].fillna("").values},
                       index=inc.index)
    gid = key.groupby(["t", "s", "c"], sort=False).ngroup().values
    G = gid.max() + 1
    rep = key.assign(g=gid).drop_duplicates("g").set_index("g").sort_index()
    cnt = np.bincount(gid, minlength=G).astype(float)
    spend = np.bincount(gid, weights=inc["spend_raw"].abs().fillna(0).values, minlength=G)
    log(f"  classify: {len(inc)} lines -> {G} unique (text,supplier,type) groups, {K} categories")

    # ---- features ----
    w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[a-z]{2,}")
    ch = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    corpus = rep["t"].tolist() + tree.texts
    w.fit(corpus); ch.fit(corpus)
    enc = lambda t: normalize(hstack([w.transform(t), ch.transform(t)]).tocsr())
    X, C = enc(rep["t"].tolist()), enc([t.lower() for t in tree.texts])
    S = (X @ C.T).toarray()
    method = np.array(["zero_shot"] * G, dtype=object)
    if cc.get("use_embeddings"):
        E = _embed_sim(rep["t"].tolist(), tree.texts, cc["embedding_model"])
        if E is not None:
            S = 0.5 * S + 0.5 * E
            method[:] = "zero_shot+emb"
            log("  embeddings: on")
    empty = ~rep["t"].str.contains("[a-z]", regex=True).values      # no readable text at all
    P0 = _softmax(S, 0.05)

    # ---- optional LLM on top-spend unique items ----
    llm_mask = np.zeros(G, dtype=bool)
    if llm_mod.available(cfg):
        lcfg = cc["llm"]
        order = np.argsort(-spend)[: int(lcfg.get("max_items", 5000))]
        order = [g for g in order if not empty[g]]
        items = [{"text": rep.at[g, "t"], "supplier": rep.at[g, "s"], "ctx": rep.at[g, "c"]} for g in order]
        res, usage = llm_mod.label_items(items, tree, cfg, os.path.join(cfg.get("_out_dir", "outputs"), ".llm_cache.json"))
        for k, (leaf, conf) in res.items():
            g = order[k]
            P0[g] = 0.1 * P0[g]
            P0[g, leaf] += 0.9
            llm_mask[g] = True
            method[g] = "llm"
        log(f"  LLM: labelled {len(res)}/{len(items)} items, tokens in={usage['input_tokens']} out={usage['output_tokens']}")
    elif cc["llm"].get("enabled"):
        log("  LLM enabled in config but ANTHROPIC_API_KEY not set -> local classifier only")

    # ---- self-training ----
    oh_s = OneHotEncoder(handle_unknown="ignore").fit_transform(rep[["s"]])
    oh_c = OneHotEncoder(handle_unknown="ignore").fit_transform(rep[["c"]])
    use_sup = cc.get("use_supplier_prior", True)
    F = hstack([X, oh_s, oh_c]).tocsr() if use_sup else X
    P = P0.copy()
    if cc.get("self_train", True) and G > 50:
        Pc = P0
        for r in range(3):
            srt = np.sort(Pc, 1)
            margin = srt[:, -1] - srt[:, -2]
            thr = np.quantile(margin[~empty], 1 - min(0.9, 0.3 + 0.2 * r)) if (~empty).any() else 1
            sel = (margin >= thr) & ~empty
            sel |= llm_mask
            clf = _fit_lr(F[sel], Pc.argmax(1)[sel], np.sqrt(cnt[sel]), 5)
            if clf is None:
                break
            Pn = _proba(clf, F, K)
            Pc = Pn if r == 2 else 0.5 * P0 + 0.5 * Pn
        P = Pc
        method = np.where(llm_mask, "llm+selftrain", np.where(method == "zero_shot+emb", "zero_shot+emb+selftrain", "zero_shot+selftrain"))
    conf_cap = 0.6                      # label-free predictions are never reported as very confident

    # ---- supervised on client labels (optional) ----
    exact = {}
    if guide is not None and guide.cats is not None and len(guide.cats) >= 200:
        pidx = {p: i for i, p in enumerate(tree.paths)}
        lab = guide.cats.set_index("record_id")
        lcols = [c for c in lab.columns if c.startswith("l")]
        paths = lab[lcols].apply(lambda r: tuple(v for v in r if isinstance(v, str) and v), axis=1)
        leaf = paths.map(pidx)
        leaf = leaf.dropna().astype(int)
        rid = inc["record_id"].values
        rowpos = pd.Series(np.arange(len(inc)), index=rid)
        known = leaf[leaf.index.isin(rowpos.index)]
        if len(known) >= 200:
            rows = rowpos[known.index].values
            tr = pd.DataFrame({"g": gid[rows], "y": known.values}).groupby(["g", "y"]).size().reset_index(name="n")
            clf = _fit_lr(F[tr["g"].values], tr["y"].values, tr["n"].values.astype(float), 10)
            if clf is not None:
                P = _proba(clf, F, K)
                method[:] = "supervised"
                conf_cap = 1.0
                exact = known.to_dict()
                log(f"  supervised on {len(known)} labelled lines")
    if empty.any():
        P[empty] = P[~empty].mean(0) if (~empty).any() else 1.0 / K
        method[empty] = "no_text"
    leaf_g = P.argmax(1)
    conf_g = np.minimum(P.max(1), conf_cap)
    conf_g[empty] = 0.05
    out["leaf"] = leaf_g[gid]
    out["confidence"] = np.round(conf_g[gid], 3)
    out["method"] = method[gid]
    if exact:                           # the client's own label for this very record
        m = inc["record_id"].isin(exact.keys()).values
        out.loc[m, "leaf"] = inc.loc[m, "record_id"].map(exact).values
        out.loc[m, "confidence"] = 0.99
        out.loc[m, "method"] = "client_label"
    return out
