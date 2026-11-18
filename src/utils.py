import re
import numpy as np
import pandas as pd

_nonalnum = re.compile(r"[^a-z0-9]+")

def norm_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return _nonalnum.sub(" ", str(s).lower()).strip()

def parse_number(x):
    """'1,234.56' '1.234,56' '(12.5)' '12.5-' ' R 1 200,50' -> float; NaN if impossible."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if not s:
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.endswith("-"):
        neg, s = True, s[:-1]
    s = re.sub(r"[^\d,.\-]", "", s)
    if s.startswith("-"):
        neg, s = (not neg), s[1:]
    if not s:
        return np.nan
    if "," in s and "." in s:
        s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".") if (s.count(",") == 1 and re.search(r",\d{1,2}$", s)) else s.replace(",", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v

def parse_vat(x, default):
    v = parse_number(x)
    if np.isnan(v):
        return default
    return v / 100.0 if v > 1 else v

def parse_dates(raw, formats, dayfirst=True):
    """Return datetime Series. Tries explicit formats, then a lenient parser."""
    s = raw.astype("string").str.strip()
    s = s.mask(s == "")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for f in formats or []:
        todo = out.isna() & s.notna()
        if not todo.any():
            break
        out[todo] = pd.to_datetime(s[todo], format=f, errors="coerce")
    todo = out.isna() & s.notna()
    if todo.any():
        num = pd.to_numeric(s[todo], errors="coerce")       # excel serials
        ser = num.between(20000, 80000)
        if ser.any():
            out[ser[ser].index] = pd.to_datetime("1899-12-30") + pd.to_timedelta(num[ser], unit="D")
        todo = out.isna() & s.notna()
        for i in s[todo].index:
            try:
                out[i] = pd.to_datetime(s[i], dayfirst=dayfirst, errors="raise")
            except Exception:
                pass
    return out

def po_norm(x, regex):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    m = re.search(regex, str(x))
    if not m:
        return ""
    return (m.group(1) if m.groups() else m.group(0)).lstrip("0")

class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)
