import numpy as np
import pandas as pd

def pair_prf(pred, gold):
    """Pairwise precision/recall/F1 of two groupings (arrays of cluster ids)."""
    t = pd.crosstab(pd.Series(pred, name="p"), pd.Series(gold, name="g"))
    c2 = lambda x: x * (x - 1) / 2
    tp = c2(t.values).sum()
    pp = c2(t.sum(axis=1).values).sum()
    gp = c2(t.sum(axis=0).values).sum()
    p = tp / pp if pp else 1.0
    r = tp / gp if gp else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
