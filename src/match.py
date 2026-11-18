
"""Conservative, configurable duplicate matching.

Rules:
1. Exact re-exports within the secondary source:
   same invoice + normalized description + net amount.
   Optional extra key fields can make matching stricter.

2. Secondary-to-primary matching:
   Route A: same PO + amount within tolerance.
   Route B: close amount + sufficiently similar description.
             If both supplier names exist, supplier similarity is
             also checked by default.

Every match is one-to-one. The primary record is retained.
"""

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .utils import norm_text, po_norm


# ---------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------

REEXPORT_FIELDS = {
    "invoice_no",
    "desc_norm",
    "spend_raw",
    "vendor_id",
    "cur",
    "qty",
    "po_norm",
}

ROUTE_B_POLICIES = {
    "if_available",
    "required",
    "ignore",
}


def _validate_config(cfg):
    """Validate matching configuration before doing any matching."""

    m = cfg.get("match", {})

    if not isinstance(m, dict):
        raise ValueError("'match' must be a mapping in the company config.")

    if not m.get("enabled", False):
        return m

    primary = m.get("primary")
    secondary = m.get("secondary")

    if not primary or not secondary:
        raise ValueError(
            "Matching is enabled, but 'match.primary' or "
            "'match.secondary' is missing."
        )

    if primary == secondary:
        raise ValueError(
            "'match.primary' and 'match.secondary' must be different."
        )

    source_names = [s.get("name") for s in cfg.get("sources", [])]

    if primary not in source_names:
        raise ValueError(
            f"Primary source '{primary}' does not exist in config sources. "
            f"Available sources: {source_names}"
        )

    if secondary not in source_names:
        raise ValueError(
            f"Secondary source '{secondary}' does not exist in config sources. "
            f"Available sources: {source_names}"
        )

    # Numeric tolerance validation
    numeric_settings = [
        "amount_tol_rel",
        "amount_tol_abs",
        "nopo_tol_rel",
    ]

    for key in numeric_settings:
        if key not in m:
            raise ValueError(f"Missing match setting: '{key}'")

        try:
            value = float(m[key])
        except (TypeError, ValueError):
            raise ValueError(f"match.{key} must be numeric.")

        if not np.isfinite(value) or value < 0:
            raise ValueError(f"match.{key} must be finite and non-negative.")

    # Description similarity threshold
    if "desc_min_sim" not in m:
        raise ValueError("Missing match setting: 'desc_min_sim'")

    try:
        value = float(m["desc_min_sim"])
    except (TypeError, ValueError):
        raise ValueError("match.desc_min_sim must be numeric.")

    if not 0 <= value <= 1:
        raise ValueError("match.desc_min_sim must be between 0 and 1.")

    # Configurable re-export key
    reexport_key = m.get(
        "reexport_key",
        ["invoice_no", "desc_norm", "spend_raw"],
    )

    if not isinstance(reexport_key, list) or not reexport_key:
        raise ValueError("match.reexport_key must be a non-empty list.")

    unknown = set(reexport_key) - REEXPORT_FIELDS

    if unknown:
        raise ValueError(
            f"Unsupported reexport_key fields: {sorted(unknown)}. "
            f"Allowed fields: {sorted(REEXPORT_FIELDS)}"
        )

    required = {"invoice_no", "desc_norm", "spend_raw"}

    if not required.issubset(set(reexport_key)):
        raise ValueError(
            "match.reexport_key must contain invoice_no, desc_norm "
            "and spend_raw."
        )

    if len(reexport_key) != len(set(reexport_key)):
        raise ValueError("match.reexport_key contains duplicate fields.")

    # Route B settings
    route_b = m.get("route_b", {})

    if not isinstance(route_b, dict):
        raise ValueError("match.route_b must be a mapping.")

    policy = route_b.get("supplier_policy", "if_available")

    if policy not in ROUTE_B_POLICIES:
        raise ValueError(
            f"Invalid route_b.supplier_policy '{policy}'. "
            f"Choose one of {sorted(ROUTE_B_POLICIES)}."
        )

    supplier_min_sim = route_b.get("supplier_min_sim", 0.95)

    try:
        supplier_min_sim = float(supplier_min_sim)
    except (TypeError, ValueError):
        raise ValueError("match.route_b.supplier_min_sim must be numeric.")

    if not np.isfinite(supplier_min_sim) or not 0 <= supplier_min_sim <= 1:
        raise ValueError(
            "match.route_b.supplier_min_sim must be between 0 and 1."
        )

    # PO regex
    try:
        re.compile(m.get("po_regex", r"(\d{6,})"))
    except re.error as exc:
        raise ValueError(f"Invalid match.po_regex: {exc}")

    return m


# ---------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------

def _prep(df, cfg):
    """Prepare normalized fields used by matching."""

    m = cfg.get("match", {})
    rg = m.get("po_regex", r"(\d{6,})")

    df["po_norm"] = df["po"].map(
        lambda x: po_norm(x, rg)
    )

    df["desc_norm"] = df["desc"].fillna("").map(norm_text)

    df["sup_norm"] = (
        df["supplier"]
        .fillna("")
        .map(norm_text)
        .str.replace(" ", "", regex=False)
    )

    df["_match_invoice"] = (
        df["invoice_no"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["_match_vendor"] = (
        df["vendor_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["_match_currency"] = (
        df["cur"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["_match_qty"] = pd.to_numeric(
        df["qty"], errors="coerce"
    ).round(6)

    df["_match_amount"] = pd.to_numeric(
        df["spend_raw"], errors="coerce"
    ).round(2)

    return df


# ---------------------------------------------------------------------
# Re-export matching
# ---------------------------------------------------------------------

def _reexport_duplicates(df, cfg, secondary):
    """Remove exact repeated exports within the secondary source."""

    m = cfg["match"]

    key_fields = m.get(
        "reexport_key",
        ["invoice_no", "desc_norm", "spend_raw"],
    )

    field_map = {
        "invoice_no": "_match_invoice",
        "desc_norm": "desc_norm",
        "spend_raw": "_match_amount",
        "vendor_id": "_match_vendor",
        "cur": "_match_currency",
        "qty": "_match_qty",
        "po_norm": "po_norm",
    }

    key = [field_map[field] for field in key_fields]

    S = df[
        (df["source"] == secondary)
        & (df["exclusion_reason"] == "")
    ].copy()

    if S.empty:
        return df

    valid = (
        S["_match_invoice"].ne("")
        & S["desc_norm"].ne("")
        & S["_match_amount"].notna()
    )

    S = S.loc[valid].copy()

    if S.empty:
        return df

    secondary_cfg = next(
        s for s in cfg["sources"] if s["name"] == secondary
    )

    approved = secondary_cfg.get("status", {}).get("approved", [])

    order = {
        str(value).strip().lower(): i
        for i, value in enumerate(approved)
    }

    S["_rank"] = (
        S["status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(order)
        .fillna(99)
    )

    S = S.sort_values(
        ["_rank", "record_id"],
        kind="stable",
    )

    first = S.drop_duplicates(key, keep="first")
    first_lookup = first.set_index(key)["record_id"]

    duplicates = S[S.duplicated(key, keep="first")]

    for i, row in duplicates.iterrows():
        values = tuple(row[col] for col in key)

        df.at[i, "exclusion_reason"] = "DUPLICATE"
        df.at[i, "dup_of"] = first_lookup.loc[values]
        df.at[i, "match_rule"] = "reexport"

    return df


# ---------------------------------------------------------------------
# Route A: same PO + similar amount
# ---------------------------------------------------------------------

def _route_a_candidates(P, S, m, p_amt, p_desc):
    """Generate candidate pairs using PO reference and amount."""

    by_po = {}

    for j, po in P["po_norm"].items():
        if po:
            by_po.setdefault(po, []).append(j)

    tol_r = float(m["amount_tol_rel"])
    tol_a = float(m["amount_tol_abs"])

    pairs = []
    found_secondary = set()

    for i, row in S.iterrows():

        amount = row["_match_amount"]

        if pd.isna(amount):
            continue

        po = row["po_norm"]

        if not po:
            continue

        for j in by_po.get(po, []):

            primary_amount = p_amt[j]

            if pd.isna(primary_amount):
                continue

            difference = abs(primary_amount - amount)

            tolerance = max(
                tol_a,
                tol_r * abs(primary_amount),
            )

            if difference <= tolerance:

                desc = row["desc_norm"]
                primary_desc = p_desc[j]

                similarity = (
                    fuzz.token_set_ratio(desc, primary_desc) / 100.0
                    if desc and primary_desc
                    else 0.0
                )

                score = (
                    1.0
                    + (1.0 - difference / (tolerance + 1e-9)) * 0.5
                    + similarity
                )

                pairs.append(
                    (score, i, j, "po+amount")
                )

                found_secondary.add(i)

    return pairs, found_secondary


# ---------------------------------------------------------------------
# Route B: close amount + conservative text/supplier matching
# ---------------------------------------------------------------------

def _supplier_match_allowed(row, primary_row, route_b):
    """Apply the configured supplier similarity policy."""

    policy = route_b.get("supplier_policy", "if_available")

    if policy == "ignore":
        return True, 0.0

    s1 = row["sup_norm"]
    s2 = primary_row["sup_norm"]

    if s1 and s2:
        similarity = fuzz.ratio(s1, s2) / 100.0

        threshold = float(
            route_b.get("supplier_min_sim", 0.95)
        )

        return similarity >= threshold, similarity

    if policy == "required":
        return False, 0.0

    return True, 0.0


def _route_b_candidates(
    P,
    S,
    m,
    p_amt,
    p_desc,
    p_sup,
    excluded_from_route_b,
):
    """Generate conservative no-PO matching candidates."""

    route_b = m.get("route_b", {})

    desc_threshold = float(m["desc_min_sim"])
    tol_r = float(m["nopo_tol_rel"])
    tol_a = float(m["amount_tol_abs"])

    # Index primary records by rounded amount.
    amount_index = {}

    for j, amount in p_amt.items():

        if pd.isna(amount):
            continue

        amount_index.setdefault(
            int(round(amount)), []
        ).append(j)

    pairs = []

    for i, row in S.iterrows():

        if i in excluded_from_route_b:
            continue

        amount = row["_match_amount"]

        if pd.isna(amount):
            continue

        desc = row["desc_norm"]

        if not desc:
            continue

        # Only inspect nearby rounded amounts.
        center = int(round(amount))
        candidate_indices = set()

        for bucket in (
            center - 1,
            center,
            center + 1,
        ):
            candidate_indices.update(
                amount_index.get(bucket, [])
            )

        for j in candidate_indices:

            # SAFEGUARD:
            # If both records have PO references, they must agree.
            secondary_po = row["po_norm"]
            primary_po = P.loc[j, "po_norm"]

            if secondary_po and primary_po and secondary_po != primary_po:
                continue

            primary_amount = p_amt[j]

            if pd.isna(primary_amount):
                continue

            difference = abs(primary_amount - amount)

            tolerance = max(
                tol_a,
                tol_r * abs(primary_amount),
            )

            if difference > tolerance:
                continue

            primary_desc = p_desc[j]

            if not primary_desc:
                continue

            desc_similarity = (
                fuzz.token_set_ratio(desc, primary_desc) / 100.0
            )

            if desc_similarity < desc_threshold:
                continue

            primary_row = P.loc[j]

            supplier_allowed, supplier_similarity = (
                _supplier_match_allowed(
                    row,
                    primary_row,
                    route_b,
                )
            )

            if not supplier_allowed:
                continue

            score = (
                desc_similarity
                + 0.30 * supplier_similarity
            )

            pairs.append(
                (score, i, j, "amount+text")
            )

    return pairs


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def run_match(df, cfg):
    """Run configured duplicate matching."""

    df["dup_of"] = ""
    df["match_rule"] = ""

    m = _validate_config(cfg)

    df = _prep(df, cfg)

    if not m.get("enabled", False):
        return df

    primary = m["primary"]
    secondary = m["secondary"]

    # Step 1: exact re-exports within the secondary source.
    df = _reexport_duplicates(
        df,
        cfg,
        secondary,
    )

    # Step 2: prepare eligible primary and secondary records.
    P = df[
        (df["source"] == primary)
        & (df["exclusion_reason"] == "")
    ].copy()

    S = df[
        (df["source"] == secondary)
        & (df["exclusion_reason"] == "")
    ].copy()

    if P.empty or S.empty:
        return df

    p_amt = P["_match_amount"].to_dict()
    p_desc = P["desc_norm"].to_dict()
    p_sup = P["sup_norm"].to_dict()

    # Step 3: Route A.
    route_a, matched_by_po = _route_a_candidates(
        P,
        S,
        m,
        p_amt,
        p_desc,
    )

    # Step 4: Route B for records without a usable PO match.
    route_b = _route_b_candidates(
        P,
        S,
        m,
        p_amt,
        p_desc,
        p_sup,
        matched_by_po,
    )

    # Step 5: globally rank candidate pairs and enforce one-to-one matching.
    pairs = route_a + route_b

    pairs.sort(
        key=lambda x: (-x[0], str(x[1]), str(x[2]))
    )

    used_secondary = set()
    used_primary = set()

    for score, i, j, rule in pairs:

        if i in used_secondary or j in used_primary:
            continue

        used_secondary.add(i)
        used_primary.add(j)

        df.at[i, "exclusion_reason"] = "DUPLICATE"
        df.at[i, "dup_of"] = df.at[j, "record_id"]
        df.at[i, "match_rule"] = rule

    return df
