# The playbook — turning the diagnosis into fewer tokens

Token-autopsy tells you the composition. This tells you what to change, in what order,
and how to prove it worked. **Every savings figure here is an estimate from one case
study (n=10 sessions) — verify with `--compare` before you believe it for your setup.**

Order = measured size × how safe it is to change.

## 0. Measure first (30 seconds)

```
python3 token_autopsy.py <your-transcripts> --json > before.json
# ... make one change ...
python3 token_autopsy.py <your-transcripts> --json > after.json
python3 token_autopsy.py --compare before.json after.json
```

One change per cycle. If retry ratio or error rate goes *up*, the change failed even if
tokens went down — cheaper tokens with more failures is a loss.

## 1. Reasoning budget & replay — est. −27% of every prompt, −~55% of output

The single biggest term in the case study.

- **Where it's replayed:** only "echo-back" providers require prior `reasoning_content`
  in history (deepseek / xiaomi-mimo / kimi families — Hermes detects and replays these;
  strict providers strip it to 0). Check your report: a `reasoning_replay` context share
  > 15% means you're on an echo family.
- **Hermes:** `reasoning_effort` (config.yaml) → `low`/`medium`; per-model
  `reasoning_overrides:` for cheap models; switch casual sessions to a non-echo model.
- **Claude Code / Cursor / others:** lower the thinking budget or use a no-thinking
  model for simple turns. There is no replay lever — only budget.
- **Risk:** hard tasks need thinking. Raise it back the moment retries/errors climb.
  This is the one lever that can *cost* tokens if overdone.

## 2. Static prefix — est. −15% of every prompt

System prompt + skills + tool schemas riding along on every single call (case study:
~31k tokens/call). It's fully cacheable (cheap in $) but pure context-window dead weight.

- **Hermes:** trim the skills list to what the task needs; `disabled_toolsets:` for
  task-scoped sessions (no browser tools on a code task); keep SOUL/memory lean.
- **Claude Code:** slim `CLAUDE.md`; revoke MCP servers you don't use (`/mcp`);
  tighten the permissions allowlist so it stops prompting/reading about unused tools.
- **Generic:** every tool/schema you enable is billed on every call, forever.
- **Risk:** under-tooling → the agent loops trying to do things it can't. Watch tool-call
  count; if it rises, you cut too much.

## 3. Retry storms — est. up to −50% of calls in affected sessions

Case study: billed API calls were 1.98× recorded responses; two sessions ran 2.76× and
4.93× — in those, most API calls were pure waste (each retry re-pays the whole prompt).

- Find sessions with `retry x` ≥ 1.5 in the report. The cause is always the same:
  flaky/rate-limited provider, timeouts, or 5xx.
- Fix at the provider layer: fallback provider configured, sane timeouts, no free-tier
  endpoint for real work.
- **Risk:** none. This is the only lever that is pure profit.

## 4. Tool results — est. −10% of every prompt

The growth engine: each call's fresh input (case study: ~5.9k tokens/call) mostly enters
as tool output, then rides along forever.

- Truncate stdout harder; `head`/`tail` big outputs; read line ranges, not whole files.
- Don't dump entire files into `execute_code` prints — read what you need, once.
- Prefer search-with-context over cat-the-repo.
- **Risk:** too-truncated output → re-reads (more calls). If tool-call count rises, back off.

## 5. Tool-arg discipline — agent behavior, est. −5–10% of output

Generated tool-call arguments were ~45% of output tokens and ~23% of every prompt
(they're re-sent after you make them).

- Batch independent tool calls instead of one-per-turn.
- Patch edits, don't rewrite whole files.
- No restating the plan in prose before acting — the prose is also tokens (small, but
  it rides along too).

## What "2-3x" really means

- **Context tokens per call:** 2–2.5x is mechanically plausible if levers 1–4 stack
  (0.73 × 0.85 × 0.87 × 0.8 ≈ 0.43). Unmeasured — that's what `--compare` is for.
- **Claude/OpenAI users:** no replay slice → realistic ceiling ~1.3–1.6x.
- **Dollars:** don't bother — caching already made input ~10x cheaper; the case study
  total was $2.34 for 10 sessions. The wins are **context window (less compaction)**
  and **latency (fewer calls)**.
- **Floors:** system prompt + schemas have a floor; reasoning has a quality floor.
  Nobody gets 10x. 2x with unchanged quality would be a good outcome.

If a change drops tokens but raises retries or error rate, it was a bad change —
the metric that matters is *tokens per correct result*, not tokens per call.
