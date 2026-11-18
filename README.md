# spendtool

Raw purchase data + category tree -> one clean spend row per record: counted once, net of tax, in the
reporting currency, with `supplier_key`, `item_key`, an exclusion reason and a category (l1..l4) with confidence.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Put the kit files in `data/A/` and `data/B/` (never committed: see `.gitignore`).

## The one command
```bash
python -m src.run --config configs/A.yaml --out outputs/yourname__A.csv
```
Also writes `outputs/yourname__A_audit.csv` (match rule, FX, flags, classifier used per record).

| Flag | Meaning |
|---|---|
| `--no-labels` | ignore the labels file: classify from the tree alone (zero-label mode) |
| `--holdout 0.2` | hide 20% of labels, then `python eval/eval_A.py --out ... --labels data/A/labels_A.csv --hidden-ids ..._hidden_ids.txt` |
| `--llm` | enable LLM labelling of high-spend items (needs `ANTHROPIC_API_KEY`; otherwise ignored) |

## New company = new config
Copy `configs/_template.yaml`, fill the column names, run the same command. No code changes.
Generic rules live in `configs/_defaults.yaml`.

## Tests
`python tests/test_synthetic.py` (multi-currency / FX / status / exclusions on a tiny synthetic company).

## Layout
`src/` ingest -> clean -> exclude -> match -> dedup_supplier / dedup_item -> taxonomy + classify (+ llm) -> output.
