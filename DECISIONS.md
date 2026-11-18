# DECISIONS (Company A, from the data; B sections to be added after profiling B)

**Data found**
- ERP: 37,029 lines, 4 source systems merged. 24,370 lines have no vendor_id / posting_date (supplier name only). 434 negative lines (returns).
  Dates in ERP are missing for those 24,370 lines; 3 non-ZAR document currencies, but `value_zar` is already ZAR.
- Tool: 28,520 lines; gross amounts as text ("25,704.47"); VAT as `15`, `0.15`, `15%` or blank; invoice_date blank on 65%; 5 PO formats;
  702 supplier spellings; 188 Cancelled, 315 On Hold/Rejected.
- Same invoice re-exported with a different date/supplier spelling: 1,244 repeat rows on (invoice no, text, net amount).

**Decisions**
1. Net spend: tool amount / (1 + VAT); VAT blank -> 15%; VAT 0 stays 0. ERP value used as is.
2. Status: Approved/Paid count; "cancel/void/revers/delete" -> CANCELLED; anything else -> NOT_APPROVED. Blank status -> approved (config).
3. Cancelled/not-approved tool records never "consume" an ERP line, so a re-issued invoice can still match it.
4. Matching: see APPROACH.md. Tolerance 2% with PO because ~1-2% price variances exist; 0.2% without PO. No many-to-one matching
   (only 27 of 846 unmatched PO lines were explained by sums of 2-3 ERP lines).
5. Negative ERP/tool lines are kept as negative spend (returns/credit notes reduce spend). Switch: `negative_amounts: exclude`.
6. NON_PROCUREMENT: word-boundary regexes (tax, intercompany, royalty, financing, partner billing, revenue) on supplier/description/context.
   They hit nothing in A (checked: no such records found); word boundaries avoid false hits such as "ESTAX" or "heavy duty".
7. Unparseable dates are flagged (audit file) but never change inclusion; unparseable amount -> spend 0, flagged. Missing FX rate: nearest month, else
   the currency's median rate.
8. Text with no letters (blank description, account code only) gets confidence 0.05 and the category mix of the other lines.
9. Labels are optional guidance (client keys / supervised model). All "tree only" numbers use `--no-labels`.

**Known weaknesses:** zero-label category accuracy on A; golden-supplier catch-all; B config is a draft; LLM/embedding paths are
untested; tool-vs-ERP matching quality has no ground truth.
