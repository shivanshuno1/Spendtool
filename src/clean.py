"""Amounts (net of tax, reporting currency) and dates."""
import re
import numpy as np
import pandas as pd
from .config import resolve
from .utils import parse_number, parse_vat, parse_dates

def load_fx(cfg):
    fx = cfg.get("fx")
    if not fx:
        return None
    t = pd.read_csv(resolve(cfg, fx["file"]))
    mc = fx.get("month_col", t.columns[0])
    if fx.get("currency_col") and fx.get("rate_col"):            # long layout
        t = t.rename(columns={mc: "m", fx["currency_col"]: "cur", fx["rate_col"]: "rate"})[["m", "cur", "rate"]]
    else:                                                        # wide layout: one column per currency
        t = t.rename(columns={mc: "m"}).melt(id_vars="m", var_name="cur", value_name="rate")
        t["cur"] = t["cur"].astype(str).str.extract(r"([A-Za-z]{3})")[0].str.upper()
    t["m"] = pd.to_datetime(t["m"], errors="coerce").dt.to_period("M")
    t["rate"] = pd.to_numeric(t["rate"], errors="coerce")
    return t.dropna()

def clean(df, cfg):
    rep = cfg["reporting_currency"].upper()
    dflt = float(cfg["vat_default"])
    df["amount_num"] = df["amount"].map(parse_number)
    vat = df["vat"].map(lambda v: parse_vat(v, dflt))
    df["vat_used"] = np.where(df["is_gross"], vat, 0.0)
    df["net_local"] = df["amount_num"] / (1.0 + df["vat_used"])

    # ---- dates (per source: formats differ) ----
    df["date_parsed"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for src in cfg["sources"]:
        m = df.source == src["name"]
        df.loc[m, "date_parsed"] = parse_dates(df.loc[m, "date"], src.get("date_formats"), cfg["dayfirst"])
    lo, hi = cfg["date_range"]
    yr = df["date_parsed"].dt.year
    blank = df["date"].isna() | (df["date"].astype(str).str.strip() == "")
    df["date_flag"] = np.select(
        [blank, df["date_parsed"].isna(), (yr < lo) | (yr > hi)],
        ["missing", "unparsable", "out_of_range"], default="")
    df.loc[df["date_flag"] == "out_of_range", "date_parsed"] = pd.NaT

    # ---- currency -> reporting ----
    cur = df["currency"].fillna(rep).astype(str).str.strip().str.upper().replace("", rep)
    df["cur"] = cur
    fx = load_fx(cfg)
    per = df["date_parsed"].dt.to_period("M")
    rate_row = df["rate"].map(parse_number)
    rate = pd.Series(np.nan, index=df.index)
    rate[(cur == rep) | df["amount_in_reporting"]] = 1.0
    if rep == "USD":
        has = rate.isna() & rate_row.notna() & (rate_row > 0)
        rate[has] = rate_row[has]
    df["fx_source"] = np.where(rate.notna(), "row_or_same", "")
    if fx is not None:
        piv = fx.pivot_table(index="m", columns="cur", values="rate", aggfunc="last").sort_index()
        med = piv.median()
        need = rate.isna()
        def lookup(c, p):
            if c not in piv.columns:
                return np.nan, "missing"
            col = piv[c].dropna()
            if col.empty:
                return np.nan, "missing"
            if pd.isna(p):
                return float(col.median()), "median"
            if p in col.index:
                return float(col[p]), "fx_table"
            i = col.index.get_indexer([p], method="nearest")[0]
            return float(col.iloc[i]), "fx_nearest"
        for i in df.index[need]:
            r, how = lookup(cur[i], per[i])
            rate[i] = r
            df.at[i, "fx_source"] = how
        if rep != "USD":                       # table is per-USD: cross via reporting currency
            for i in df.index[need]:
                rr, _ = lookup(rep, per[i])
                if not np.isnan(rate[i]) and not np.isnan(rr):
                    rate[i] = rate[i] / rr
    rate = rate.where(rate > 0)
    df["fx_rate"] = rate
    df["spend_raw"] = (df["net_local"] / df["fx_rate"]).round(2)
    df["amount_flag"] = np.select(
        [df["amount_num"].isna(), df["fx_rate"].isna()], ["unparsable_amount", "no_fx_rate"], default="")
    return df
