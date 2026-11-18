"""Read the category tree from a file at run time. No category name lives in code."""
import re
import pandas as pd
from .config import resolve

_LVL = re.compile(r"(?i)(?:^|[^a-z])(?:l|lvl|level)[\s_\-]*(\d+)(?:[^0-9]|$)")
_DESC = re.compile(r"(?i)desc|definition|keyword|example|synonym|includes")

class Tree:
    def __init__(self, levels, paths, texts):
        self.levels, self.paths, self.texts = levels, paths, texts
        self.depth = max(len(p) for p in paths)
    def __len__(self):
        return len(self.paths)
    def names(self, i):
        p = self.paths[i]
        return list(p) + [""] * (max(self.depth, 4) - len(p))

def load_tree(cfg):
    path = resolve(cfg, cfg["tree_file"])
    t = pd.read_csv(path, dtype=str).fillna("")
    cols = cfg.get("tree_levels")
    if not cols:
        lv = sorted((int(m.group(1)), c) for c in t.columns if (m := _LVL.search(c)))
        cols = [c for _, c in lv]
    desc_cols = [c for c in t.columns if c not in (cols or []) and _DESC.search(c)]
    if not cols and len(t.columns) == 1:                      # single "A > B > C" column
        sep = next((s for s in (">", "|", "/", "::") if t.iloc[:, 0].str.contains(re.escape(s)).any()), None)
        if sep:
            t = t.iloc[:, 0].str.split(re.escape(sep), regex=True, expand=True).fillna("")
            t.columns = [f"l{i+1}" for i in range(t.shape[1])]
            cols = list(t.columns)
    if not cols:
        cols = [c for c in t.columns if c not in desc_cols]
    seen, paths, texts = set(), [], []
    for _, r in t.iterrows():
        p = [str(r[c]).strip() for c in cols]
        while p and not p[-1]:
            p.pop()
        if not p or "" in p:
            continue                                         # skip empty / gapped rows
        p = tuple(p)
        if p in seen:
            continue
        seen.add(p)
        paths.append(p)
        extra = " ".join(str(r[c]) for c in desc_cols)
        texts.append(" ".join(p) + " " + p[-1] + " " + p[-1] + " " + extra)
    return Tree(cols, paths, texts)
