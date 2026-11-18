"""Optional Gemini labelling of the highest-spend unique items."""
import hashlib, json, os, re, time
import urllib.error
import urllib.parse
import urllib.request

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

def available(cfg):
    return bool(cfg["classify"]["llm"].get("enabled")) and bool(os.environ.get("GEMINI_API_KEY"))

def _call(model, system, user, key, max_tokens=4096):
    body = json.dumps({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "responseMimeType": "application/json"},
    }).encode()
    url = API.format(model=urllib.parse.quote(model, safe=""))
    req = urllib.request.Request(url, data=body, headers={
        "content-type": "application/json", "x-goog-api-key": key})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            candidates = out.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            usage = out.get("usageMetadata", {})
            return "".join(part.get("text", "") for part in parts), {
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                raise RuntimeError(f"Gemini API request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt == 3:
                raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc
        time.sleep(2 ** attempt)

def label_items(items, tree, cfg, cache_path):
    """items: list of dicts {text, supplier, ctx}. Returns {index: (leaf_idx, conf)} and usage totals."""
    lc = cfg["classify"]["llm"]
    key = os.environ["GEMINI_API_KEY"]
    cats = "\n".join(f"{i}: " + " > ".join(p) for i, p in enumerate(tree.paths))
    system = ("You classify procurement purchase lines into a company's category tree. "
              "Choose the single best leaf category number for every line using only the numbered list. "
              "Reply with ONLY a JSON array like [{\"id\":0,\"c\":12,\"conf\":0.8}] - no prose.\n\nCATEGORIES:\n" + cats)
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    sig = hashlib.md5(cats.encode()).hexdigest()[:8]
    res, usage = {}, {"input_tokens": 0, "output_tokens": 0}
    todo = []
    for i, it in enumerate(items):
        h = sig + hashlib.md5(json.dumps(it, sort_keys=True).encode()).hexdigest()
        if h in cache:
            res[i] = tuple(cache[h])
        else:
            todo.append((i, h, it))
    bs = int(lc.get("batch_size", 40))
    for s in range(0, len(todo), bs):
        chunk = todo[s:s + bs]
        user = json.dumps([{"id": j, "text": it["text"][:200], "supplier": it.get("supplier", ""), "type": it.get("ctx", "")}
                           for j, (_, _, it) in enumerate(chunk)])
        txt, u = _call(lc["model"], system, user, key)
        usage["input_tokens"] += u.get("input_tokens", 0); usage["output_tokens"] += u.get("output_tokens", 0)
        if not txt:
            continue
        m = re.search(r"\[.*\]", txt, re.S)
        try:
            arr = json.loads(m.group(0)) if m else []
        except Exception:
            arr = []
        for o in arr:
            try:
                j, c = int(o["id"]), int(o["c"])
                if 0 <= j < len(chunk) and 0 <= c < len(tree):
                    i, h, _ = chunk[j]
                    res[i] = (c, float(o.get("conf", 0.7)))
                    cache[h] = list(res[i])
            except Exception:
                continue
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    json.dump(cache, open(cache_path, "w"))
    return res, usage
