"""Read every source file of a company into ONE common schema (driven only by config)."""
import numpy as np
import pandas as pd
from .config import resolve

FIELDS = ["amount", "date", "supplier", "vendor_id", "item_code", "po", "invoice_no",
          "qty", "currency", "rate", "vat", "status"]

def _read_file(path, src):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, dtype=str, sheet_name=src.get("sheet", 0))
    kw = dict(dtype=str, sep=src.get("sep", ","), low_memory=False)
    try:
        return pd.read_csv(path, encoding=src.get("encoding", "utf-8"), **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kw)

def _join(df, cols):
    cols = [c for c in (cols or []) if c in df.columns]
    if not cols:
        return pd.Series("", index=df.index)
    parts = df[cols].fillna("").astype(str).apply(lambda s: s.str.strip())
    return parts.apply(lambda r: " ; ".join(v for v in r if v), axis=1)

def load_source(cfg, src):
    df = _read_file(resolve(cfg, src["file"]), src)
    cm = src.get("columns", {})
    out = pd.DataFrame(index=df.index)
    out["record_id"] = src.get("id_prefix", "") + df[src["id_col"]].astype(str).str.strip()
    out["source"] = src["name"]
    for f in FIELDS:
        c = cm.get(f)
        out[f] = df[c] if c and c in df.columns else np.nan
    if "currency_fixed" in src:
        out["currency"] = src["currency_fixed"]
    scope = [c for c in src.get("vendor_scope", []) if c in df.columns]
    vcol = cm.get("vendor_id")
    if scope and vcol in df.columns:   # same vendor id can exist in several ERPs: make it unique per ERP/region
        pre = df[scope].fillna("").astype(str).agg("|".join, axis=1)
        v = df[vcol]
        out["vendor_id"] = np.where(v.notna(), pre + "|" + v.astype(str), None)
    out["desc"] = _join(df, src.get("desc_cols"))
    out["ctx"] = _join(df, src.get("context_cols"))
    out["is_gross"] = bool(src.get("amount_is_gross", False))
    out["amount_in_reporting"] = bool(src.get("amount_in_reporting", False))
    out["_date_formats"] = [src.get("date_formats", [])] * len(out)
    out["_status_cfg"] = [src.get("status", {})] * len(out)
    return out.reset_index(drop=True)

def ingest(cfg):
    frames = [load_source(cfg, s) for s in cfg["sources"]]
    df = pd.concat(frames, ignore_index=True)
    if df.record_id.duplicated().any():
        raise ValueError("record_id collision across sources: add/adjust id_prefix in config")
    return df
