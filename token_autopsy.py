#!/usr/bin/env python3
"""token-autopsy — find where your AI agent's tokens actually go.

Zero dependencies (stdlib only). Reads agent transcripts, reconstructs each API
call's context from stored messages, and breaks it into categories:

    reasoning_replay   prior thinking re-sent to the provider (some providers require it)
    static_prefix      system prompt + skills + tool schemas (inferred, or logged inline)
    assistant_toolargs your agent's generated tool calls / code, re-sent every call
    tool_results       tool output entering context (file reads, stdout, searches)
    user               the human's messages
    assistant_text     the agent's prose replies

Also counts retry storms (billed API calls vs recorded assistant responses).

Usage:
    python3 token_autopsy.py ~/.hermes/state.db      # Hermes transcripts (verified)
    python3 token_autopsy.py session.jsonl           # generic/Claude-Code-style JSONL
    python3 token_autopsy.py some/log/dir            # recurse for .jsonl / .db
    python3 token_autopsy.py --selftest              # built-in fixture test
    python3 token_autopsy.py --json path/to/logs     # machine-readable output

Token estimates: tiktoken o200k when installed, else chars/4. Shares are what
matter; absolute numbers are approximate.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
import os
import sqlite3
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime

CATS = ["static_prefix", "reasoning_replay", "assistant_toolargs", "tool_results",
        "user", "assistant_text", "session_meta"]

# ---------------------------------------------------------------- token estimate
_tok = None
def tok(s) -> int:
    global _tok
    if not s:
        return 0
    if isinstance(s, (bytes, bytearray)):
        s = s.decode("utf-8", "replace")
    s = str(s)
    if _tok is None:
        try:
            from tiktoken import get_encoding  # type: ignore
            _tok = get_encoding("o200k_base").encode
        except Exception:
            _tok = None
    if _tok is not None:
        return len(_tok(s))
    return max(1, math.ceil(len(s) / 4))

def tok_estimator_name() -> str:
    return "tiktoken/o200k" if _tok is not None else "chars/4 (install tiktoken for exact counts)"

# ---------------------------------------------------------------- session model
def new_session(sid, date="", source="", model="", msgs=None, usage=None):
    return {"id": sid, "date": date, "source": source, "model": model,
            "messages": msgs or [],           # (ts, role, {cat: tokens}, tool_name)
            "usage": usage or {},             # calls, prompt, completion, cache_read, reasoning, cost
            }

def _inc(u, **kw):
    for k, v in kw.items():
        u[k] = u.get(k, 0) + (v or 0)

def _cat_key(role):
    if role == "tool":
        return "tool_results"
    return "assistant_text" if role == "assistant" else role

def categorize_content(role, content, tool_calls=None, reasoning=None):
    """-> ({cat: tokens}, tool_name)"""
    cats = Counter()
    tool_name = None

    def blocks(items):
        nonlocal tool_name
        for b in items:
            if not isinstance(b, dict):
                cats[_cat_key(role)] += tok(b)
                continue
            t = b.get("type")
            if t in ("text", None):
                cats[_cat_key(role)] += tok(b.get("text") or b.get("content") or b)
            elif t in ("tool_use", "function_call", "tool_call"):
                cats["assistant_toolargs"] += tok(json.dumps(b.get("input") or b.get("arguments") or b, default=str))
                tool_name = b.get("name") or tool_name
            elif t in ("tool_result", "function_call_output"):
                cats["tool_results"] += tok(b.get("content") or b)
            elif t in ("reasoning", "thinking", "reasoning_content"):
                cats["reasoning_replay"] += tok(b.get("thinking") or b.get("text") or b.get("reasoning") or b)
            elif t == "tool_calls":  # openai-style nested
                cats["assistant_toolargs"] += tok(b.get("tool_calls") or b)
            else:
                cats[_cat_key(role)] += tok(b)

    if isinstance(content, str):
        cats[_cat_key(role)] += tok(content)
    elif isinstance(content, list):
        blocks(content)
    elif isinstance(content, dict):
        blocks([content])

    if tool_calls:
        cats["assistant_toolargs"] += tok(tool_calls)
        try:
            first = tool_calls[0]
            tool_name = first.get("name") or first.get("function", {}).get("name") or tool_name
        except Exception:
            pass
    if reasoning and role == "assistant":
        cats["reasoning_replay"] += tok(reasoning)
    return dict(cats), tool_name

# ---------------------------------------------------------------- readers
def read_hermes(path, limit=40):
    """Hermes state.db — verified against real data (see README case study)."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sessions = []
    rows = con.execute("""
        select id, source, model, started_at, message_count, tool_call_count
        from sessions where tool_call_count >= 1 and message_count >= 4
        order by started_at desc limit ?""", (limit,))
    for s in rows:
        sid = s["id"]
        usage = {"calls": 0, "prompt": 0, "completion": 0, "cache_read": 0,
                 "reasoning": 0, "cost": 0.0}
        for r in con.execute("""select api_call_count, input_tokens, output_tokens,
                                       cache_read_tokens, reasoning_tokens, estimated_cost_usd
                                from session_model_usage where session_id=?""", (sid,)):
            _inc(usage, calls=r["api_call_count"], prompt=r["input_tokens"],
                 completion=r["output_tokens"], cache_read=r["cache_read_tokens"],
                 reasoning=r["reasoning_tokens"], cost=r["estimated_cost_usd"] or 0)
        msgs = []
        for m in con.execute("""select timestamp, role, content, tool_calls, tool_name,
                                       reasoning, reasoning_content, active
                                from messages
                                where session_id=? and role in ('user','assistant','tool','session_meta')
                                order by timestamp""", (sid,)):
            if not m["active"]:
                continue
            cats, tname = categorize_content(m["role"], m["content"],
                                             tool_calls=m["tool_calls"],
                                             reasoning=m["reasoning_content"] or m["reasoning"])
            msgs.append((m["timestamp"] or 0, m["role"], cats, tname or m["tool_name"]))
        date = time.strftime("%Y-%m-%d", time.localtime(s["started_at"])) if s["started_at"] else ""
        sessions.append(new_session(sid, date, s["source"] or "", s["model"] or "", msgs, usage))
    con.close()
    return sessions

def _ts(obj, fallback):
    v = obj.get("timestamp")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except Exception:
            pass
    return fallback

def read_jsonl(path):
    """Generic JSONL. Handles OpenAI-style {role, content, usage} lines and
    Claude-Code-style {type, message:{role, content, usage}} lines.
    Best-effort across tools; category shares are the point, not absolutes."""
    msgs, usage = [], {"calls": 0, "prompt": 0, "completion": 0, "cache_read": 0,
                       "reasoning": 0, "cost": 0.0}
    model, source = "", ""
    call_names = {}   # tool_call_id -> tool name
    n_assistant = 0
    with open(path, errors="replace") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            inner = obj.get("message") if isinstance(obj.get("message"), dict) else None
            role = (inner or {}).get("role") or obj.get("role") or obj.get("type")
            if role not in ("user", "assistant", "system", "developer", "tool", "session_meta"):
                continue
            if role == "developer":
                role = "system"
            content = (inner or {}).get("content", obj.get("content"))
            tool_calls = (inner or {}).get("tool_calls") or obj.get("tool_calls")
            reasoning = (inner or {}).get("reasoning_content") or obj.get("reasoning_content") \
                or (inner or {}).get("reasoning") or obj.get("reasoning")
            model = model or (inner or {}).get("model") or obj.get("model") or ""
            source = source or str(obj.get("source") or "")
            # per-call usage (openai + anthropic shapes)
            u = (inner or {}).get("usage") or obj.get("usage")
            if isinstance(u, dict):
                if role == "assistant":
                    _inc(usage, calls=1)
                prompt = u.get("input_tokens") or u.get("prompt_tokens") or 0
                comp = u.get("output_tokens") or u.get("completion_tokens") or 0
                det = u.get("prompt_tokens_details") or {}
                cache = u.get("cache_read_input_tokens") or det.get("cached_tokens") or 0
                rsn = u.get("reasoning_tokens") or (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                _inc(usage, prompt=prompt, completion=comp, cache_read=cache, reasoning=rsn)
            if role == "system":
                cats = {"static_prefix": tok(content)}
                msgs.append((_ts(obj, i), "system", cats, None))
                continue
            if role == "assistant":
                n_assistant += 1
                for tc in (tool_calls or (content if isinstance(content, list) else [])) or []:
                    if isinstance(tc, dict) and tc.get("name") or isinstance(tc, dict) and tc.get("function"):
                        cid = tc.get("id") or tc.get("call_id")
                        nm = tc.get("name") or (tc.get("function") or {}).get("name")
                        if cid and nm:
                            call_names[cid] = nm
            cats, tname = categorize_content(role, content, tool_calls, reasoning)
            if role == "tool" and not tname:
                tname = call_names.get(obj.get("tool_call_id"))
            msgs.append((_ts(obj, i), role, cats, tname))
    if not usage["calls"]:
        usage["calls"] = n_assistant
    date = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(path)))
    return [new_session(os.path.basename(path), date, source or "jsonl", model, msgs, usage)]

def load(path, limit=40):
    if os.path.isdir(path):
        out = []
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.endswith((".jsonl", ".db")) and not f.startswith("~"):
                    try:
                        out += load(os.path.join(root, f), limit)
                    except Exception as e:
                        print(f"skip {f}: {e}", file=sys.stderr)
        return out
    if path.endswith(".db"):
        with open(path, errors="replace") as fh:
            head = fh.read(4096)
        if "session_model_usage" in head or "CREATE TABLE sessions" in head:
            return read_hermes(path, limit)
        raise SystemExit(f"{path}: sqlite but not a Hermes state.db (adapter wanted, PRs welcome)")
    return read_jsonl(path)

# ---------------------------------------------------------------- analysis
def analyze(sessions):
    """Reconstruct avg context per API call (assistant messages = call boundaries),
    infer static prefix = billed prompt/call - reconstructable messages/call."""
    out = []
    for s in sessions:
        msgs = sorted(s["messages"], key=lambda m: m[0])
        bounds = [m[0] for m in msgs if m[1] == "assistant"]
        tss = [m[0] for m in msgs]
        prefix = [Counter()]
        for _, _, cats, _ in msgs:
            nxt = Counter(prefix[-1]); nxt.update(cats)
            prefix.append(nxt)
        agg = Counter()
        for b in bounds:
            i = bisect.bisect_left(tss, b)
            agg.update(prefix[i])
        n = max(len(bounds), 1)
        avg = {c: agg[c] / n for c in CATS}
        avg["session_meta"] = avg.get("session_meta", 0)

        usage = s["usage"]
        calls = usage.get("calls") or len(bounds)
        billed_prompt_per_call = ((usage.get("prompt", 0) + usage.get("cache_read", 0)) / calls) if calls else 0
        avg_msg_total = sum(avg.values())
        avg["static_prefix"] += max(0.0, billed_prompt_per_call - avg_msg_total) if usage.get("prompt") else 0

        out_cat = Counter()
        for _, _, cats, _ in msgs:
            for k in ("assistant_text", "assistant_toolargs", "reasoning_replay"):
                out_cat[k] += cats.get(k, 0)
        tools = Counter(); tool_calls = Counter()
        for _, _, cats, tn in msgs:
            if cats.get("tool_results"):
                tools[tn or "?"] += cats["tool_results"]
                tool_calls[tn or "?"] += 1

        r = dict(s)
        r["avg_ctx"] = avg
        r["ctx_total"] = sum(avg.values())
        r["out_cat"] = dict(out_cat)
        r["retry_ratio"] = (calls / len(bounds)) if bounds else None
        r["calls"] = calls
        r["top_tools"] = dict(tools.most_common(8))
        r["tool_call_counts"] = dict(tool_calls)
        out.append(r)
    return out

def hints(agg):
    h = []
    ctx = agg["avg_ctx_pct"]
    if ctx.get("reasoning_replay", 0) >= 15:
        h.append("reasoning replay is >=15% of every prompt: on echo-back providers "
                 "(deepseek/mimo/kimi) it's required; elsewhere it's stripped. "
                 "Lower reasoning_effort, or use a provider that doesn't replay it.")
    if agg["out_pct"].get("reasoning_replay", 0) >= 40:
        h.append("reasoning is >=40% of output tokens: trim reasoning_effort for simple turns.")
    if ctx.get("static_prefix", 0) >= 15:
        h.append("static prefix (system+skills+tool schemas) is >=15% of every prompt: "
                 "slim skills and disable unused toolsets per task.")
    rr = agg.get("retry_ratio")
    if rr and rr >= 1.5:
        h.append(f"billed API calls are {rr:.1f}x recorded responses: retry storms. "
                 "Harden retries/fallbacks; every retry re-pays the prompt.")
    if ctx.get("tool_results", 0) >= 15:
        h.append("tool results are >=15% of context: truncate stdout harder, read narrower slices.")
    if agg["ctx_total"] >= 100_000:
        h.append(f"avg context {agg['ctx_total']:,.0f} tokens: near common 128k windows, "
                 "expect compaction. Shrink ride-along terms first.")
    if not h:
        h.append("no single term dominates; profile again after any config change.")
    return h

def report(res, as_json=False):
    n = len(res)
    ctx_avg = {c: sum(r["avg_ctx"].get(c, 0) for r in res) / n for c in CATS}
    ctx_total = sum(ctx_avg.values()) or 1
    out_cat = Counter()
    for r in res:
        out_cat.update(r["out_cat"])
    out_total = sum(out_cat.values()) or 1
    billed = Counter()
    for r in res:
        _inc(billed, calls=r["usage"].get("calls", 0), prompt=r["usage"].get("prompt", 0),
             completion=r["usage"].get("completion", 0), cache_read=r["usage"].get("cache_read", 0),
             reasoning=r["usage"].get("reasoning", 0), cost=r["usage"].get("cost", 0))
    responses = sum(len([m for m in r["messages"] if m[1] == "assistant"]) for r in res)
    rr = (billed["calls"] / responses) if responses else None
    tools = Counter()
    for r in res:
        tools.update(r["top_tools"])
    agg = {"avg_ctx_pct": {k: round(100 * v / ctx_total, 1) for k, v in ctx_avg.items()},
           "out_pct": {k: round(100 * v / out_total, 1) for k, v in out_cat.items()},
           "ctx_total": round(ctx_total), "retry_ratio": rr}

    if as_json:
        return json.dumps({"sessions": [
            {"id": r["id"], "date": r["date"], "calls": r["calls"],
             "context_avg_by_cat": {k: round(v) for k, v in r["avg_ctx"].items()},
             "output_by_cat": r["out_cat"], "retry_ratio": r["retry_ratio"],
             "usage": r["usage"], "top_tools": r["top_tools"]} for r in res],
            "aggregate": agg, "hints": hints(agg)}, indent=1)

    L = [f"# token-autopsy report — {n} session(s), {time.strftime('%Y-%m-%d')}",
         f"estimator: {tok_estimator_name()}\n",
         "## sessions", "| date | source | model | calls | retry x | fresh in | cache in | out | reasoning | $ |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(res, key=lambda x: x["date"]):
        u = r["usage"]
        rr_s = f"{r['retry_ratio']:.2f}" if r["retry_ratio"] else "-"
        L.append(f"| {r['date']} | {r['source'][:9]} | {str(r['model'])[:20]} | {r['calls']} | {rr_s} | "
                 f"{u.get('prompt', 0):,} | {u.get('cache_read', 0):,} | {u.get('completion', 0):,} | "
                 f"{u.get('reasoning', 0):,} | {u.get('cost', 0):.3f} |")
    L += ["\n## avg context composition per API call", "| category | share | tokens |", "|---|---|---|"]
    for k, v in sorted(ctx_avg.items(), key=lambda kv: -kv[1]):
        if v > 0:
            L.append(f"| {k} | {100*v/ctx_total:.1f}% | {v:,.0f} |")
    L += [f"\navg context: **{ctx_total:,.0f} tokens/call**",
          "\n## output tokens", "| category | share | tokens |", "|---|---|---|"]
    for k, v in out_cat.most_common():
        L.append(f"| {k} | {100*v/out_total:.1f}% | {v:,.0f} |")
    L += ["\n## totals",
          f"- billed API calls: {billed['calls']:,}  vs recorded responses: {responses:,}"
          + (f"  → **retry ratio {rr:.2f}x**" if rr else "")]
    tot_in = billed["prompt"] + billed["cache_read"]
    if tot_in:
        L.append(f"- input: {billed['prompt']:,} fresh + {billed['cache_read']:,} cache-read "
                 f"({100*billed['cache_read']/tot_in:.0f}% cached)")
    if billed["cost"]:
        L.append(f"- est. cost: ${billed['cost']:.2f}")
    if tools:
        L += ["\n## top tool-result tokens", "| tool | tokens |", "|---|---|"]
        for k, v in tools.most_common(6):
            L.append(f"| {k} | {v:,} |")
    L += ["\n## hints"]
    L += [f"- {h}" for h in hints(agg)]
    L.append("- remedies for every hint: see PLAYBOOK.md (bundled); verify with --compare")
    return "\n".join(L)

# ---------------------------------------------------------------- compare
def _avg_cat(doc, key):
    ss = doc.get("sessions") or []
    n = max(len(ss), 1)
    out = Counter()
    for s in ss:
        for k, v in (s.get(key) or {}).items():
            out[k] += (v or 0) / n
    return out

def _pct(b, a):
    return f"{100 * (a - b) / b:+.1f}%" if b else "n/a"

def compare(before, after):
    """Diff two --json reports: before/after a config change."""
    L = ["# before → after", ""]
    for title, key, unit in (("avg context per API call", "context_avg_by_cat", "/call"),
                             ("output tokens", "output_by_cat", "/session")):
        b, a = _avg_cat(before, key), _avg_cat(after, key)
        L += [f"## {title} ({unit})", "| category | before | after | Δ | Δ% |", "|---|---|---|---|---|"]
        for k in sorted(set(b) | set(a), key=lambda k: -abs(a[k] - b[k])):
            L.append(f"| {k} | {b[k]:,.0f} | {a[k]:,.0f} | {a[k]-b[k]:+,.0f} | {_pct(b[k], a[k])} |")
        tb, ta = sum(b.values()), sum(a.values())
        L.append(f"| **total** | **{tb:,.0f}** | **{ta:,.0f}** | **{ta-tb:+,.0f}** | **{_pct(tb, ta)}** |")
        tail = (f"**{tb/ta:.2f}x fewer**" if 0 < ta < tb else
                (f"**{ta/tb:.2f}x more**" if ta > tb else "**unchanged**"))
        L.append(f"→ {tail}\n")
    bb = (before.get("aggregate") or {}).get("retry_ratio")
    aa = (after.get("aggregate") or {}).get("retry_ratio")
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "-"
    L.append(f"retry ratio: {fmt(bb)} → {fmt(aa)}" +
             ("  ⚠ worse: fixes traded tokens for failures" if bb and aa and aa > bb else ""))
    sb, sa = len(before.get("sessions") or []), len(after.get("sessions") or [])
    L.append(f"sessions compared: {sb} → {sa}")
    L.append("sanity: if retries or error rate rose, the change failed even where tokens fell.")
    return "\n".join(L)

# ---------------------------------------------------------------- selftest
def selftest():
    import tempfile
    ok = []
    with tempfile.TemporaryDirectory() as td:
        # --- hermes-shaped fixture ---
        db = os.path.join(td, "state.db")
        con = sqlite3.connect(db)
        con.executescript("""
          create table sessions (id text, source text, model text, started_at real,
            message_count int, tool_call_count int);
          create table messages (session_id text, timestamp real, role text, content text,
            tool_calls text, tool_name text, reasoning text, reasoning_content text, active int);
          create table session_model_usage (session_id text, api_call_count int, input_tokens int,
            output_tokens int, cache_read_tokens int, reasoning_tokens int, estimated_cost_usd real);
        """)
        con.execute("insert into sessions values ('s1','telegram','m/s',1790000000,4,1)")
        rows = [
            ("s1", 100.0, "user", "do the thing", None, None, None, None, 1),
            ("s1", 200.0, "assistant", "", '[{"function":{"name":"execute_code","arguments":"{...}"}}]',
             "execute_code", "thinking hard", "thinking hard", 1),
            ("s1", 300.0, "tool", '{"output":"x"*900}', None, "execute_code", None, None, 1),
            ("s1", 400.0, "assistant", "done!", None, None, None, None, 1),
        ]
        con.executemany("insert into messages values (?,?,?,?,?,?,?,?,?)", rows)
        # 2 billed calls, prompt 1000/call => static inferred = (2000/2) - msgs_avg
        con.execute("insert into session_model_usage values ('s1',2,1000,400,0,100,0.01)")
        con.commit(); con.close()

        sess = load(db)
        assert len(sess) == 1 and sess[0]["usage"]["calls"] == 2, "hermes reader"
        res = analyze(sess)
        a = res[0]["avg_ctx"]
        assert a["user"] > 0 and a["assistant_toolargs"] > 0 and a["tool_results"] > 0, "categories"
        assert a["reasoning_replay"] > 0, "reasoning cat"
        assert res[0]["out_cat"]["assistant_text"] > 0, "output text cat (final reply)"
        assert res[0]["retry_ratio"] and abs(res[0]["retry_ratio"] - 1.0) < 1e-9, "retry ratio"
        rep = report(res)
        assert "avg context composition" in rep and "hints" in rep, "report sections"
        ok.append("hermes adapter")

        # --- claude-code-shaped jsonl fixture ---
        jl = os.path.join(td, "conv.jsonl")
        lines = [
            {"type": "user", "timestamp": "2026-10-01T10:00:00Z",
             "message": {"role": "user", "content": "hi"}},
            {"type": "assistant", "timestamp": "2026-10-01T10:00:05Z",
             "message": {"role": "assistant",
                         "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}],
                         "usage": {"input_tokens": 1000, "output_tokens": 50,
                                   "cache_read_input_tokens": 800}}},
            {"type": "user", "timestamp": "2026-10-01T10:00:07Z",
             "message": {"role": "user",
                         "content": [{"type": "tool_result", "content": "file1 file2"}]}},
            {"type": "assistant", "timestamp": "2026-10-01T10:00:09Z",
             "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}],
                         "usage": {"input_tokens": 1200, "output_tokens": 30,
                                   "cache_read_input_tokens": 900}}},
        ]
        with open(jl, "w") as fh:
            for o in lines:
                fh.write(json.dumps(o) + "\n")
        sess = load(jl)
        assert sess[0]["usage"]["calls"] == 2 and sess[0]["usage"]["prompt"] == 2200, "jsonl usage"
        res = analyze(sess)
        a = res[0]["avg_ctx"]
        assert a["tool_results"] > 0 and a["assistant_toolargs"] > 0 and a["user"] > 0, "jsonl cats"
        assert abs(res[0]["retry_ratio"] - 1.0) < 1e-9, "jsonl retry ratio"
        assert set(a) <= set(CATS), f"unexpected categories: {set(a) - set(CATS)}"
        ok.append("jsonl adapter (claude-code shape)")

        # --- openai-shaped jsonl fixture ---
        jl2 = os.path.join(td, "oai.jsonl")
        with open(jl2, "w") as fh:
            fh.write(json.dumps({"role": "user", "content": "q"}) + "\n")
            fh.write(json.dumps({
                "role": "assistant", "content": None, "reasoning_content": "let me think",
                "tool_calls": [{"function": {"name": "exec", "arguments": "{}"}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 100,
                          "prompt_tokens_details": {"cached_tokens": 400}}}) + "\n")
            fh.write(json.dumps({"role": "tool", "content": "result text"}) + "\n")
            fh.write(json.dumps({"role": "assistant", "content": "answer",
                                 "usage": {"prompt_tokens": 600, "completion_tokens": 20}}) + "\n")
        sess = load(jl2)
        assert sess[0]["usage"]["prompt"] == 1100 and sess[0]["usage"]["cache_read"] == 400, "openai usage"
        res = analyze(sess)
        a = res[0]["avg_ctx"]
        assert a["reasoning_replay"] > 0 and a["assistant_toolargs"] > 0 and a["tool_results"] > 0, "openai cats"
        ok.append("jsonl adapter (openai shape)")

        # --- compare ---
        import copy as _copy
        d1 = json.loads(report(res, as_json=True))
        d2 = _copy.deepcopy(d1)
        for s in d2["sessions"]:
            s["context_avg_by_cat"] = {k: v / 2 for k, v in s["context_avg_by_cat"].items()}
            s["output_by_cat"] = {k: v / 2 for k, v in s["output_by_cat"].items()}
        txt = compare(d1, d2)
        assert "-50.0%" in txt and "2.00x fewer" in txt and "before → after" in txt, "compare halves"
        bigger = compare(d2, d1)
        assert "2.00x more" in bigger, "compare direction"
        txt2 = compare(d1, _copy.deepcopy(d1))
        assert "0.0%" in txt2 or "n/a" in txt2, "compare identical"
        ok.append("compare")

        # --- autodetect + clean errors ---
        got = autodetect(40, candidates=[db])
        assert got[0] == db and len(got[1]) == 1, "autodetect finds fixture db"
        try:
            autodetect(40, candidates=[os.path.join(td, "nope.jsonl"), db + "-missing"])
            raise AssertionError("autodetect must fail on empty probe")
        except SystemExit as e:
            assert "no agent transcripts found" in str(e) and "not found" in str(e), "probe list in error"
        try:
            main(["/definitely/not/here.jsonl"])
            raise AssertionError("bad path must exit")
        except SystemExit as e:
            assert "not found" in str(e), "clean one-line path error"
        ok.append("autodetect + clean errors")
    print("SELFTEST PASS: " + ", ".join(ok))
    return 0

# ---------------------------------------------------------------- auto-discovery
AUTO_CANDIDATES = [
    "~/.hermes/state.db",        # Hermes (sqlite)
    "~/.claude/projects",        # Claude Code sessions (jsonl)
    "~/.codex/sessions",         # Codex CLI rollouts (jsonl)
    "~/.local/share/opencode",   # OpenCode
    "~/.gemini/tmp",             # Gemini CLI logs
]

def autodetect(limit=40, candidates=None):
    """Find agent transcripts without being told where they are -> zero-arg runs work."""
    tried = []
    for pat in (candidates or AUTO_CANDIDATES):
        full = os.path.expanduser(pat)
        paths = sorted(glob.glob(full)) or ([full] if os.path.exists(full) else [])
        if not paths:
            tried.append(f"{pat} (not found)")
            continue
        for path in paths:
            try:
                sessions = load(path, limit)
            except SystemExit as e:
                tried.append(f"{path} ({e})")
                continue
            except Exception as e:
                tried.append(f"{path} ({type(e).__name__})")
                continue
            if sessions:
                return path, sessions
            tried.append(f"{path} (no sessions)")
    raise SystemExit("no agent transcripts found. probed:\n  " + "\n  ".join(tried) +
                     "\nrun with an explicit path: python3 token_autopsy.py <file-or-dir>")

# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="token-autopsy",
                                 description="Where do your AI agent's tokens actually go?")
    ap.add_argument("path", nargs="?", help="transcript file, Hermes state.db, or directory")
    ap.add_argument("--selftest", action="store_true", help="run built-in fixture tests")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--limit", type=int, default=40, help="max sessions to read (default 40, 0=all)")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="diff two --json reports (before/after a config change)")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.compare:
        docs = [json.load(open(os.path.expanduser(x))) for x in args.compare]
        print(compare(*docs))
        return 0
    limit = args.limit or 10**9
    if args.path:
        where = os.path.abspath(os.path.expanduser(args.path))
        try:
            sessions = load(where, limit)
        except SystemExit:
            raise
        except FileNotFoundError:
            raise SystemExit(f"not found: {where}")
        except Exception as e:
            raise SystemExit(f"cannot read {where}: {type(e).__name__}: {e}")
        if not sessions:
            raise SystemExit(f"no sessions in {where} (need >=4 messages and >=1 tool call, "
                             "or jsonl messages with role/content)")
    else:
        where, sessions = autodetect(limit)
    res = analyze(sessions)
    print(report(res, as_json=args.json))
    return 0

if __name__ == "__main__":
    sys.exit(main())
