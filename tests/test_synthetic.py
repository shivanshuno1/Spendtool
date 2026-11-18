"""Smoke test of the B/C-style code paths on a tiny synthetic company (no real data needed).
   python tests/test_synthetic.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.utils import parse_number, parse_vat
from src.run import run

assert parse_number("1,234.56") == 1234.56 and parse_number("1.234,56") == 1234.56
assert parse_number("(12.5)") == -12.5 and parse_number("12,5") == 12.5 and pd.isna(parse_number("abc"))
assert parse_vat("15%", .15) == .15 and parse_vat("0.15", .15) == .15 and parse_vat(None, .15) == .15

d = tempfile.mkdtemp()
pd.DataFrame({"L1": ["Packaging", "Packaging", "Services", "Services"], "L2": ["Tinplate", "Cans", "Freight", "Maintenance"],
              "L3": ["", "", "", ""]}).to_csv(f"{d}/tree.csv", index=False)
pd.DataFrame({"month": ["2024-01-01", "2024-02-01"], "EUR": [0.9, 0.92], "INR": [83.0, 83.5]}).to_csv(f"{d}/fx.csv", index=False)
rows = [  # id, erp, vendor, desc, amt, cur, rate, date, status, acct
    ("1", "JDE-US", "V1", "tinplate sheets 0.2mm", "1,000.00", "USD", "", "2024-01-15", "Paid", "5000"),
    ("2", "SAP-EU", "V1", "Blechtafeln Weissblech", "920,00", "EUR", "", "15.02.2024", "Approved", "5000"),
    ("3", "SAP-IN", "V2", "freight charges sea", "83,000", "INR", "83", "2024-01-20", "Paid", "6100"),
    ("4", "SAP-IN", "V2", "Maintenance cans line repair", "8,350", "INR", "", "2024-02-10", "Paid", "6200"),
    ("5", "JDE-US", "V3", "VAT payment Q1", "500", "USD", "", "2024-03-01", "Paid", "9100"),
    ("6", "JDE-US", "V4", "intercompany recharge", "700", "USD", "", "2024-03-02", "Paid", "9200"),
    ("7", "JDE-US", "V5", "cans", "10", "USD", "", "32/13/2024", "Cancelled", "5000"),
    ("8", "JDE-US", "V5", "cans", "10", "USD", "", "2024-03-03", "On hold", "5000"),
    ("9", "JDE-US", "V6", "", "55", "USD", "", "", "Paid", "5000"),
]
pd.DataFrame(rows, columns=["line_id", "erp", "vendor", "text", "amount", "currency", "currency_rate", "inv_date", "status", "account"]).to_csv(f"{d}/inv.csv", index=False)
open(f"{d}/cfg.yaml", "w").write(f"""
company: T
reporting_currency: USD
tree_file: {d}/tree.csv
fx: {{file: {d}/fx.csv, month_col: month}}
sources:
  - name: ap
    file: {d}/inv.csv
    id_col: line_id
    amount_is_gross: false
    columns: {{amount: amount, date: inv_date, vendor_id: vendor, currency: currency, rate: currency_rate, status: status}}
    vendor_scope: [erp]
    desc_cols: [text]
    context_cols: [account]
    date_formats: ["%Y-%m-%d", "%d.%m.%Y"]
    status: {{approved: [Paid, Approved], cancelled: [Cancelled]}}
""")
res, df, cls, tree, _ = run(f"{d}/cfg.yaml", f"{d}/out.csv", log=lambda *_: None)
r = res.set_index("record_id")
assert list(res.columns[:11]) == ["record_id","include","exclusion_reason","supplier_key","item_key","spend","l1","l2","l3","l4","confidence"]
assert abs(r.loc["1", "spend"] - 1000) < 1e-6
assert abs(r.loc["2", "spend"] - 1000) < 5       # 920 EUR / 0.92 (Feb rate) = 1000 USD, European number format
assert abs(r.loc["3", "spend"] - 1000) < 1e-6    # row rate 83 INR per USD
assert abs(r.loc["4", "spend"] - 100) < 1        # no row rate -> fx table (Feb 83.5)
assert r.loc["5", "exclusion_reason"] == "NON_PROCUREMENT" and r.loc["6", "exclusion_reason"] == "NON_PROCUREMENT"
assert r.loc["7", "exclusion_reason"] == "CANCELLED" and r.loc["8", "exclusion_reason"] == "NOT_APPROVED"
assert r.loc["9", "include"] == 1 and r.loc["9", "confidence"] <= 0.1
assert r.loc["1", "supplier_key"] != r.loc["3", "supplier_key"]
print(res.iloc[:, :11].to_string()); print("SMOKE TEST PASSED")
