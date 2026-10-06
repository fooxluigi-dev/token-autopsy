#!/usr/bin/env python3
"""Draw the token-autopsy report as a dark, color-coded PNG.

Input: a report from `token_autopsy.py --json` (same file --compare reads).
Output: a 1080x1500 PNG — donut of every prompt's composition, output bars, stat cards.

Needs Pillow (the only dependency in this repo, and only for this file):
    pip install pillow
    python3 token_autopsy.py <logs> --json > report.json
    python3 visualize.py report.json chart.png

No design tools, no AI image generation — this is just drawing primitives.
"""
from __future__ import annotations

import json
import os
import sys

W, H = 1080, 1500
BG, CARD, GRID = "#0a0a0a", "#16181d", "#262a31"
GOLD, CYAN, RED, GREEN, PURPLE, GRAY = "#f5c518", "#4cc9f0", "#ff6b6b", "#51cf66", "#b197fc", "#adb5bd"
WHITE, DIM = "#e9ecef", "#b8bec7"

GOLDEN = ("static_est", "static_prefix")  # legacy + current key for the same thing
CAT_COLOR = {"reasoning_replay": GOLD, "static_prefix": CYAN, "static_est": CYAN,
             "assistant_toolargs": RED, "tool_results": GREEN,
             "user": PURPLE, "assistant_text": GRAY, "session_meta": "#495057"}
CAT_LABEL = {"reasoning_replay": "reasoning re-sent",
             "static_prefix": "static prefix (system+skills+tools)",
             "static_est": "static prefix (system+skills+tools)",
             "assistant_toolargs": "generated tool-calls",
             "tool_results": "tool results", "user": "your messages",
             "assistant_text": "agent prose", "session_meta": "session meta"}
CAT_ORDER = ["reasoning_replay", "static_prefix", "static_est", "assistant_toolargs",
             "tool_results", "user", "assistant_text", "session_meta"]
OUT_LABEL = {"reasoning": "reasoning (thinking)", "reasoning_replay": "reasoning (thinking)",
             "assistant_toolargs": "tool-call code", "assistant_text": "prose replies"}

FONTS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/System/Library/Fonts/Helvetica.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/TTF/DejaVuSans.ttf"]
FONTS_R = ["/System/Library/Fonts/Supplemental/Arial.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
           "/usr/share/fonts/TTF/DejaVuSans.ttf"]


def fmt(n):
    for div, sfx in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if abs(n) >= div:
            return f"{n/div:.1f}{sfx}"
    return f"{n:,.0f}"


def wrap(text, width=26, max_lines=3):
    lines, cur = [], ""
    for w in text.replace("\n", " \n ").split(" "):
        if w == "\\n":
            lines.append(cur); cur = ""; continue
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines[:max_lines])


def load_stats(doc):
    ss = doc.get("sessions") or []
    if not ss:
        raise SystemExit("no sessions in report")
    n = len(ss)
    ctx = {}
    tot_w = sum(((s.get("usage") or {}).get("calls") or 1) for s in ss) or n
    for s in ss:
        w = (s.get("usage") or {}).get("calls") or 1
        for k, v in (s.get("context_avg_by_cat") or {}).items():
            ctx[k] = ctx.get(k, 0) + (v or 0) * w / tot_w
    out, usage = {}, {"calls": 0, "prompt": 0, "completion": 0, "cache_read": 0, "cost": 0.0}
    for s in ss:
        for k, v in (s.get("output_by_cat") or {}).items():
            out[k] = out.get(k, 0) + (v or 0)
        u = s.get("usage") or {}
        for k in usage:
            usage[k] += (u.get(k) or 0)
    retry = (doc.get("aggregate") or {}).get("retry_ratio")
    return ctx, out, usage, retry, n


def render(doc, out_path):
    from PIL import Image, ImageDraw, ImageFont

    ctx, outc, usage, retry, nsess = load_stats(doc)
    ctx_total = sum(ctx.values()) or 1
    out_total = sum(outc.values()) or 1
    in_total = usage["prompt"] + usage["cache_read"]

    def f(size, bold=True):
        for p in (FONTS if bold else FONTS_R):
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # header
    d.text((60, 46), "WHERE YOUR AGENT'S TOKENS GO", font=f(54), fill=GOLD)
    d.text((60, 116), f"token-autopsy  ·  {nsess} sessions  ·  {usage['calls']:,} API calls",
           font=f(26, False), fill=DIM)
    d.line((60, 172, W - 60, 172), fill=GRID, width=2)

    # section 1: donut
    d.text((60, 212), f"EVERY PROMPT  (avg {ctx_total:,.0f} tokens per API call)",
           font=f(30), fill=WHITE)
    keys = [k for k in CAT_ORDER if ctx.get(k, 0) > 0.05] + \
           [k for k in sorted(ctx, key=lambda x: -ctx[x]) if k not in CAT_ORDER and ctx[k] > 0.05]
    cx, cy, R, hole = 300, 520, 205, 118
    start = -90.0
    for k in keys:
        end = start + ctx[k] / ctx_total * 360
        d.pieslice((cx - R, cy - R, cx + R, cy + R), start, end,
                   fill=CAT_COLOR.get(k, GRAY))
        start = end
    d.ellipse((cx - hole, cy - hole, cx + hole, cy + hole), fill=BG)
    d.text((cx, cy - 34), fmt(ctx_total), font=f(58), fill=WHITE, anchor="mm")
    d.text((cx, cy + 24), "per call", font=f(24, False), fill="#9aa0a8", anchor="mm")

    y = 380
    for k in keys:
        color = CAT_COLOR.get(k, GRAY)
        d.rounded_rectangle((580, y + 4, 612, y + 38), radius=8, fill=color)
        label = CAT_LABEL.get(k, k)
        pct_s, right = f"{100 * ctx[k] / ctx_total:.1f}%", 1020
        pf, lf = f(30), f(27, False)
        while lf.size > 15 and d.textlength(label, font=lf) > \
                (right - d.textlength(pct_s, font=pf) - 28) - 632:
            lf = f(lf.size - 1, False)
        d.text((632, y + 3), label, font=lf, fill=WHITE)
        d.text((right, y), pct_s, font=pf, fill=color, anchor="ra")
        y += 56

    # section 2: output bars
    d.line((60, 740, W - 60, 740), fill=GRID, width=2)
    d.text((60, 770), f"OUTPUT TOKENS  ({fmt(out_total)} counted)", font=f(30), fill=WHITE)
    bars = [(k, OUT_LABEL.get(k, k), CAT_COLOR.get(k, GRAY))
            for k in sorted(outc, key=lambda x: -outc[x]) if outc[k] > 0]
    y = 830
    for k, label, color in bars:
        pct = 100 * outc[k] / out_total
        d.text((60, y + 8), label, font=f(26, False), fill=WHITE)
        x0, x1, h = 380, 930, 44
        d.rounded_rectangle((x0, y, x1, y + h), radius=14, fill=CARD)
        d.rounded_rectangle((x0, y, x0 + max(24, int((x1 - x0) * pct / 100)), y + h),
                            radius=14, fill=color)
        d.text((1015, y + 9), f"{pct:.0f}%", font=f(28), fill=color, anchor="ra")
        y += 66

    # section 3: stat cards (skip empties)
    cards = []
    if in_total:
        all_tok = in_total + usage["completion"]
        cards.append((f"{100 * in_total / all_tok:.1f}%", GOLD,
                      f"of ALL tokens are input\n({fmt(in_total)} in : {fmt(usage['completion'])} out)"))
        if usage["cache_read"]:
            cards.append((f"{100 * usage['cache_read'] / in_total:.0f}%", CYAN,
                          "of input are cache reads"
                          + (f" — total cost ${usage['cost']:.2f}" if usage["cost"] else "")))
    if retry:
        cards.append((f"{retry:.2f}x", RED, "billed calls vs responses\n= retry storms"))
    cards = cards[:3]
    if cards:
        d.line((60, 1055, W - 60, 1055), fill=GRID, width=2)
        cw, gap, cy0, ch = 320, 30, 1090, 200
        total_w = len(cards) * cw + (len(cards) - 1) * gap
        x_start = (W - total_w) // 2
        for i, (num, color, sub) in enumerate(cards):
            x = x_start + i * (cw + gap)
            d.rounded_rectangle((x, cy0, x + cw, cy0 + ch), radius=22, fill=CARD,
                                outline=GRID, width=2)
            d.text((x + cw / 2, cy0 + 64), num, font=f(64), fill=color, anchor="mm")
            d.text((x + cw / 2, cy0 + 145), wrap(sub), font=f(22, False), fill=GRAY,
                   anchor="mm", align="center")

    d.line((60, H - 92, W - 60, H - 92), fill=GRID, width=2)
    d.text((W / 2, H - 56), "github.com/Cacaomeraviglia/token-autopsy   —   measure → fix → prove",
           font=f(26, False), fill=DIM, anchor="mm")
    img.save(out_path)
    return out_path


def selftest():
    import tempfile
    doc = {"sessions": [{
        "context_avg_by_cat": {"reasoning_replay": 100, "static_prefix": 100,
                               "assistant_toolargs": 60, "tool_results": 50,
                               "user": 10, "assistant_text": 5},
        "output_by_cat": {"reasoning": 60, "assistant_toolargs": 30, "assistant_text": 10},
        "usage": {"calls": 3, "prompt": 500, "completion": 100,
                  "cache_read": 1500, "reasoning": 50, "cost": 0.01}}],
        "aggregate": {"retry_ratio": 1.5}}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.png")
        render(doc, p)
        from PIL import Image
        im = Image.open(p)
        colors = im.convert("RGB").getcolors(maxcolors=10_000_000)
        assert im.size == (W, H) and colors and len(colors) > 10, "image drawn with content"
    print("SELFTEST PASS: render")
    return 0


def main(argv=None):
    argv = argv or sys.argv[1:]
    if argv == ["--selftest"]:
        return selftest()
    if len(argv) not in (1, 2):
        raise SystemExit(__doc__)
    doc = json.load(open(os.path.expanduser(argv[0])))
    out = os.path.expanduser(argv[1]) if len(argv) == 2 else "chart.png"
    try:
        print(render(doc, out))
    except ImportError:
        raise SystemExit("visualize needs Pillow (the report itself needs nothing): pip install pillow")


if __name__ == "__main__":
    sys.exit(main())
