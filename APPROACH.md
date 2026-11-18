# APPROACH (draft: fill in your name, B numbers and the blind-test tooling you actually used)

**Pipeline** (config-driven, one command): ingest -> clean (VAT, FX, dates, flags) -> exclusions (status, non-procurement,
reason codes) -> cross-system match -> supplier / item de-dup -> tree-driven classification -> output (+ audit file).
All company specifics are in `configs/*.yaml`; generic rules in `_defaults.yaml`. No category names in code.

**Counting once (A).** Re-exports inside the invoice tool: same invoice number + normalised text + net amount -> later copies
DUPLICATE (supplier spelling and dates differ between exports, so they are not part of the key). Tool vs ERP: PO reference
normalised to its 10 digits; among ERP lines on that PO the nearest net amount (gross / (1+VAT), tolerance 2%) wins, one-to-one,
best amount+text score first. No usable PO: amount within 0.2% AND text similarity >= 0.6 (or supplier similarity >= 0.9).
Unmatched approved tool records are tool-only spend and stay included. Result on A: 23,147 duplicates, 4,870 tool-only
records (ZAR 365m of ZAR 2.92bn). There is no ground truth for this step in the kit, so it is checked by plausibility only.

**Supplier de-dup.** Union-find over: same vendor id, same compacted name (legal suffixes/punctuation/spacing removed), near-identical
names (typos, rapidfuzz >= 92). Matched tool records inherit their ERP partner's keys. **Item de-dup:** item code is the identity;
code-less lines use the normalised description, fuzzy-merged only if their numbers are identical.

**Unseen tree, zero labels.** (1) TF-IDF (words + char n-grams) similarity between line text and category paths; (2) optional
sentence-embeddings and (3) optional Gemini labels for the highest-spend unique items (env `GEMINI_API_KEY`, else skipped);
(4) self-training: confident lines become pseudo-labels for a text + supplier + spend-type model that re-predicts all lines.
With a labels file the supervised model replaces step 4. Confidence is capped at 0.6 for label-free predictions.

**Measured on A (ERP lines vs labels_A.csv)**

| Mode | Supplier pair F1 | Item pair F1 | L1 / L2 / L3 acc (lines) | L1 / L2 / L3 (spend-weighted) |
|---|---|---|---|---|
| Labels hidden entirely (tree only) | 0.18 (P=1.0, R=0.10) | 0.64 (P=1.0, R=0.47) | 55 / 21 / 18 % | 69 / 35 / 30 % |
| 20% of labels hidden, rest used | 1.00 | 0.95 | 98 / 94 / 93 % | 97 / 94 / 91 % |

Zero-label is weak on A: its tree is equipment-specific ("CM Repair Items", "Roof Bolters") and the text is abbreviated; a lexical
method cannot bridge that. The supplier recall gap is one golden "supplier" of ~122 unrelated anonymised names (56% of ERP lines)
that no name-based method can recover without labels. B and the LLM / embedding layers were NOT measured (no B files, no API key
available while building).

**Runtime:** ~80 s zero-label / ~190 s with labels for 65,549 records on one CPU (limit: 60 min per 25k lines).
**LLM cost:** none by default. If enabled, only unique high-spend items are sent in batches of 40; estimate ~12k input + 3k output
tokens per 1,000 lines on A (not measured): multiply by the current price of the configured model.
