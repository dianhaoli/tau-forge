"""Paper-style figures for the README (docs/figures/*.svg).

Diagrams are drawn by hand; the result figure reads the real run data:
the two tau2 results JSONs (per-task success counts only) and
runs/ep-main/{metrics,val_log}.jsonl.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from math import comb
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
SIM = ROOT / "data" / "simulations"
RUN = ROOT / "runs" / "ep-main"

FONT = "'Times New Roman', Times, 'Liberation Serif', serif"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
INK, INK2, MUTED, RULE = "#1a1a1a", "#4d4c48", "#8a8984", "#d9d8d3"
USER_BG, AGENT_BG, CALL_BG, RET_BG = "#f4d9db", "#dcebf9", "#fbdcb4", "#e6e6e3"
BASE_C, OURS_C, NEG_C, ZERO_C = "#b4b3ad", "#2a78d6", "#eb6834", "#cfcec8"
WALL = "#c0392b"


def t(x, y, s, size=13, anchor="start", weight="normal", fill=INK, style="normal", family=FONT):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-style="{style}" fill="{fill}">{escape(s)}</text>')


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>{body}</svg>\n')


def arrow_defs():
    return ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker>'
            '<marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{WALL}"/></marker></defs>')


def line(x1, y1, x2, y2, color=INK2, width=1.5, dash=None, marker="ah"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d}{m}/>'


def box(x, y, w, h, fill="#ffffff", stroke=INK2, r=8, width=1.3, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{d}/>'


# ---------------------------------------------------------------- icons


def icon_person(cx, cy, s=1.0, fill="#e8b9a0"):
    return (f'<g transform="translate({cx},{cy}) scale({s})">'
            f'<path d="M-14,16 Q-14,2 0,2 Q14,2 14,16 Z" fill="#c8504f" stroke="{INK}" stroke-width="1.2"/>'
            f'<circle cx="0" cy="-7" r="8" fill="{fill}" stroke="{INK}" stroke-width="1.2"/>'
            f'<path d="M-8,-8 Q-8,-17 0,-17 Q8,-17 8,-8 Q4,-12 0,-12 Q-4,-12 -8,-8 Z" fill="#3a3a3a"/></g>')


def icon_robot(cx, cy, s=1.0):
    return (f'<g transform="translate({cx},{cy}) scale({s})">'
            f'<line x1="0" y1="-17" x2="0" y2="-11" stroke="{INK}" stroke-width="1.2"/>'
            f'<circle cx="0" cy="-18" r="2" fill="#5b8fc7" stroke="{INK}" stroke-width="1"/>'
            f'<rect x="-11" y="-11" width="22" height="15" rx="4" fill="#a9c7e8" stroke="{INK}" stroke-width="1.2"/>'
            f'<circle cx="-5" cy="-4" r="2.4" fill="{INK}"/><circle cx="5" cy="-4" r="2.4" fill="{INK}"/>'
            f'<rect x="-9" y="6" width="18" height="11" rx="3" fill="#5b8fc7" stroke="{INK}" stroke-width="1.2"/></g>')


def icon_wrench(cx, cy, s=1.0):
    return (f'<g transform="translate({cx},{cy}) scale({s}) rotate(-45)">'
            f'<rect x="-2.5" y="-4" width="5" height="20" rx="2" fill="#9aa3ad" stroke="{INK}" stroke-width="1.1"/>'
            f'<path d="M-7,-10 a7,7 0 1,0 14,0 l-3,0 l0,5 l-8,0 l0,-5 z" fill="#c2c9d0" stroke="{INK}" stroke-width="1.1"/>'
            f'</g>')


def icon_db(cx, cy, s=1.0):
    return (f'<g transform="translate({cx},{cy}) scale({s})">'
            f'<path d="M-12,-10 L-12,10 A12,4.5 0 0,0 12,10 L12,-10" fill="#5d6577" stroke="{INK}" stroke-width="1.1"/>'
            f'<path d="M-12,0 A12,4.5 0 0,0 12,0" fill="none" stroke="#9aa3b5" stroke-width="1.1"/>'
            f'<ellipse cx="0" cy="-10" rx="12" ry="4.5" fill="#8790a3" stroke="{INK}" stroke-width="1.1"/></g>')


def icon_gear(cx, cy, r=6, fill="#ffffff"):
    teeth = "".join(
        f'<rect x="{-1.6}" y="{-r - 2.6}" width="3.2" height="3.6" fill="{INK2}" transform="rotate({a})"/>'
        for a in range(0, 360, 45)
    )
    return (f'<g transform="translate({cx},{cy})">{teeth}<circle r="{r}" fill="{INK2}"/>'
            f'<circle r="{r * 0.4}" fill="{fill}"/></g>')


def bubble(x, y, text, fill, size=13, mono=False, h=26):
    cw = 7.4 if mono else 6.2
    w = int(len(text) * cw * size / 13) + 22
    fam = MONO if mono else FONT
    fs = size - 1 if mono else size
    return box(x, y, w, h, fill=fill, stroke="none", r=10) + t(x + 11, y + h / 2 + 4.5, text, size=fs, family=fam), w


# ---------------------------------------------------------------- figure 1


def fig_overview():
    W, H = 1180, 410
    b = [arrow_defs()]
    # held-out region (right of the wall is the real benchmark)
    b.append(box(870, 20, 295, 364, fill="#fbf3f2", stroke=WALL, r=10, dash="6 4"))
    b.append(t(1017, 44, "Held out: real τ²-bench retail", 14, "middle", "bold", WALL))
    b.append(t(1017, 62, "114 tasks, never trained on or used for selection", 12, "middle", style="italic", fill=WALL))

    def node(x, y, w, h, title, lines, fill="#ffffff", stroke=INK2):
        out = [box(x, y, w, h, fill=fill, stroke=stroke)]
        out.append(t(x + w / 2, y + 22, title, 14, "middle", "bold"))
        for i, s in enumerate(lines):
            out.append(t(x + w / 2, y + 42 + 17 * i, s, 12, "middle", fill=INK2))
        return "".join(out)

    # data
    b.append(node(20, 90, 170, 92, "τ² retail database", ["500 users · 1,000 orders", "50 products", "(db.json only)"]))
    b.append(line(190, 136, 222, 136))
    b.append(node(224, 90, 190, 92, "Episode generator", ["11 templates + composites", "gold chain replayed & verified", "ids discoverable, hints unique"]))
    b.append(line(414, 136, 446, 136))
    b.append(node(448, 90, 150, 92, "Synthetic tasks", ["1,410 train", "163 validation", "0 real tasks"]))

    # training loop
    b.append(box(224, 214, 620, 166, fill="#f7f9fc", stroke="#9fb6d3", r=10))
    b.append(t(236, 234, "GRPO training loop  (1 × H100, LoRA r64, TRL + vLLM)", 13, weight="bold", fill="#2c5282"))
    b.append(line(523, 182, 523, 250))
    b.append(node(244, 252, 250, 112, "Rollouts (16 tasks × 8)", ["policy ⇄ scripted customer", "policy ⇄ real τ² retail tools", "on a private copy of the DB", "≤ 20 turns per conversation"]))
    b.append(line(494, 308, 528, 308))
    b.append(node(530, 252, 140, 112, "Reward", ["end-state hash", "= gold?", "− consent / auth", "  gates"]))
    b.append(line(670, 308, 704, 308))
    b.append(node(706, 252, 124, 112, "Update", ["group-relative", "advantage", "DAPO loss", "LoRA weights"]))
    b.append(f'<path d="M768,252 C768,200 640,200 600,215" fill="none" stroke="{INK2}" stroke-width="1.4" '
             'stroke-dasharray="4 3" marker-end="url(#ah)"/>')
    b.append(t(700, 205, "next step", 11, "middle", style="italic", fill=MUTED))

    # selection + eval
    b.append(node(640, 90, 200, 92, "Checkpoint selection", ["synthetic validation", "every 25 steps", "→ step 75 chosen"]))
    b.append(f'<path d="M790,252 L790,182" fill="none" stroke="{INK2}" stroke-width="1.4" marker-end="url(#ah)"/>')
    b.append(line(840, 136, 900, 136))
    b.append(node(902, 90, 232, 120, "τ²-bench retail eval", ["τ²'s own orchestrator + grader", "LLM customer (gpt-6-luna)", "114 tasks × 4 trials", "one checkpoint, run once"], fill="#ffffff", stroke=WALL))
    b.append(box(902, 232, 232, 80, fill="#ffffff", stroke=WALL))
    b.append(t(1018, 256, "pass^1  53.1% → 61.2%", 15, "middle", "bold"))
    b.append(t(1018, 278, "pass^4  28.9% → 36.8%", 15, "middle", "bold"))
    b.append(t(1018, 299, "base model → step 75", 12, "middle", style="italic", fill=INK2))
    b.append(line(1018, 210, 1018, 230))

    # decontamination: reject-only signal across the wall
    b.append(box(902, 326, 232, 44, fill="#ffffff", stroke=WALL, dash="3 3"))
    b.append(t(1018, 344, "decontamination (reject-only):", 12, "middle", fill=WALL))
    b.append(t(1018, 361, "user ids + gold end-state hashes", 12, "middle", fill=WALL))
    b.append(f'<path d="M902,348 L860,348 L860,396 L207,396 L207,165 L222,165" fill="none" stroke="{WALL}" '
             'stroke-width="1.3" stroke-dasharray="5 3" marker-end="url(#ahr)"/>')
    b.append(t(540, 391, "rejects any candidate touching a real task's user or reproducing its end state", 11.5, "middle", style="italic", fill=WALL))
    return svg(W, H, "".join(b))


# ---------------------------------------------------------------- figure 2


def fig_episode():
    W, H = 1180, 640
    b = [arrow_defs()]
    b.append(t(24, 30, "(a) One training episode", 15, weight="bold"))
    b.append(t(800, 30, "(b) How it is graded", 15, weight="bold"))
    b.append(f'<line x1="770" y1="16" x2="770" y2="{H - 16}" stroke="{INK}" stroke-width="1.2" stroke-dasharray="3 4"/>')

    rows = [
        ("user", "Hi, I'd like to exchange the water bottle from an order I got recently.", None),
        ("agent", "Happy to help. Could you give me your email to verify your account?", None),
        ("user", "Sure, it's maya.k@example.com.", None),
        ("call", 'find_user_id_by_email(email="maya.k@example.com")', "auth earned: the customer said it"),
        ("call", 'get_user_details(…)  →  get_order_details("#W5431789")  →  get_product_details(…)', "reads: find order + variant"),
        ("agent", "I'll exchange the 750 ml bottle for the 1000 ml blue one, paid with Visa ···1234. Proceed?", None),
        ("user", "That's not what I asked for. I want 1000 ml red, everything else the same.", "wrong detail → correction, no yes"),
        ("agent", "Got it: 750 ml → 1000 ml red, difference on Visa ···1234. Shall I go ahead?", None),
        ("user", "Yes, go ahead.", "consent slot: yes"),
        ("call", 'exchange_delivered_order_items(order_id="#W5431789", …)', "write: confirmed + authed"),
        ("ret", '{"status": "exchange requested", …}', None),
        ("user", "Great, thank you! That's all I needed.  ###STOP###", "episode ends: user_stop"),
    ]
    y = 52
    for kind, text, note in rows:
        if kind == "user":
            b.append(icon_person(46, y + 14, 0.9))
            g, w = bubble(72, y, text, USER_BG)
        elif kind == "agent":
            g, w = bubble(0, 0, text, AGENT_BG)
            x = 700 - w
            g, w = bubble(x, y, text, AGENT_BG)
            b.append(icon_robot(726, y + 14, 0.9))
        elif kind == "call":
            b.append(icon_wrench(46, y + 13, 0.9))
            g, w = bubble(72, y, text, CALL_BG, size=12, mono=True)
        else:
            b.append(icon_db(46, y + 13, 0.85))
            g, w = bubble(72, y, text, RET_BG, size=12, mono=True)
        b.append(g)
        if note:
            nx = 72 + w + 10 if kind != "agent" else None
            if nx is not None and nx < 560:
                b.append(t(nx, y + 17, "← " + note, 11.5, style="italic", fill="#8a4b0f"))
        y += 45

    # legend
    ly = H - 30
    lx = 40
    for lab, fill in (("scripted customer", USER_BG), ("policy (Qwen3-4B)", AGENT_BG), ("tool call (real τ² tools)", CALL_BG), ("tool result", RET_BG)):
        b.append(box(lx, ly - 11, 14, 14, fill=fill, stroke="none", r=3))
        b.append(t(lx + 20, ly, lab, 12, fill=INK2))
        lx += 175

    # (b) grading
    gx = 800
    b.append(icon_db(gx + 20, 80, 1.1))
    b.append(t(gx + 44, 76, "final DB state", 13, weight="bold"))
    b.append(t(gx + 44, 93, "after the episode", 12, fill=INK2))
    b.append(icon_db(gx + 210, 80, 1.1))
    b.append(t(gx + 234, 76, "gold end state", 13, weight="bold"))
    b.append(t(gx + 234, 93, "replayed gold chain", 12, fill=INK2))
    b.append(line(gx + 20, 100, gx + 160, 136))
    b.append(line(gx + 210, 100, gx + 190, 136))
    b.append(box(gx + 35, 138, 280, 62, fill="#f6f6f4"))
    b.append(t(gx + 175, 160, "hash(final) = hash(gold) ?", 13.5, "middle", "bold"))
    b.append(t(gx + 175, 177, "τ²'s own test: any path reaching the gold", 11.5, "middle", fill=INK2))
    b.append(t(gx + 175, 191, "state counts, no step-by-step matching", 11.5, "middle", fill=INK2))

    b.append(line(gx + 100, 200, gx + 77, 238))
    b.append(t(gx + 80, 222, "yes", 12, "end", style="italic", fill=MUTED))
    b.append(line(gx + 250, 200, gx + 270, 238))
    b.append(t(gx + 268, 222, "no", 12, style="italic", fill=MUTED))

    b.append(box(gx - 10, 240, 175, 160, fill="#eef5ee", stroke="#5a8f5a"))
    b.append(t(gx + 77, 262, "success: start at 1.0", 13, "middle", "bold"))
    for i, s in enumerate(("−0.3 write without a", "        standing “yes”", "−0.3 write before auth", "        with customer-given", "        email / name+zip")):
        b.append(t(gx + 2, 284 + 17 * i, s, 12, fill=INK2))
    b.append(t(gx + 77, 388, "floor 0.4", 12, "middle", style="italic", fill=MUTED))

    b.append(box(gx + 185, 240, 170, 160, fill="#fbf1ea", stroke="#c07a43"))
    b.append(t(gx + 270, 262, "failure: ≤ 0.2", 13, "middle", "bold"))
    for i, s in enumerate(("+0.05 authenticated", "+0.05 read target order", "+0.10 confirmed gold-", "        write attempt", "all removed if a write", "touched another record")):
        b.append(t(gx + 195, 284 + 17 * i, s, 12, fill=INK2))

    b.append(t(gx + 175, 432, "This episode: hash matches, no gates  →  R = 1.0", 13.5, "middle", "bold", OURS_C))

    tab = [("perfect agent", "1.0"), ("skipped asking for consent", "0.7"), ("“authenticated” with an email read from the DB", "0.7"),
           ("exchanged to the wrong variant", "0.2"), ("refusal task: refused after checking", "1.0"),
           ("refusal task: refused blindly on turn 1", "0.4"), ("refusal task: did the forbidden write", "0.0")]
    ty = 466
    b.append(t(gx - 6, ty, "Reference agents (pinned by tests)", 13, weight="bold"))
    b.append(f'<line x1="{gx - 6}" y1="{ty + 7}" x2="{gx + 360}" y2="{ty + 7}" stroke="{RULE}"/>')
    for i, (lab, r) in enumerate(tab):
        yy = ty + 26 + 18 * i
        b.append(t(gx - 6, yy, lab, 12, fill=INK2))
        b.append(t(gx + 360, yy, r, 12, "end", "bold"))
    return svg(W, H, "".join(b))


# ---------------------------------------------------------------- figure 3


def fig_tokens():
    W, H = 1180, 330
    b = [arrow_defs()]
    b.append(t(24, 30, "(a) What one rollout looks like to the trainer", 15, weight="bold"))
    segs = [
        ("system + tools + greeting + opening", 250, "#efefec", "prompt"),
        ("assistant: tool call", 120, OURS_C, 1),
        ("tool results", 120, "#dddcd6", 0),
        ("assistant: question", 120, OURS_C, 1),
        ("customer reply", 110, "#dddcd6", 0),
        ("assistant: recap", 110, OURS_C, 1),
        ("customer “yes”", 90, "#dddcd6", 0),
        ("assistant: write", 110, OURS_C, 1),
    ]
    x, y = 24, 52
    for lab, w, fill, m in segs:
        b.append(f'<rect x="{x}" y="{y}" width="{w - 3}" height="40" rx="4" fill="{fill}"/>')
        col = "#ffffff" if m == 1 else INK
        b.append(t(x + (w - 3) / 2, y + 25, lab, 12, "middle", fill=col))
        if m != "prompt":
            b.append(t(x + (w - 3) / 2, y + 58, f"mask {m}", 11.5, "middle", fill=OURS_C if m == 1 else MUTED, family=MONO))
        x += w
    b.append(t(24 + 125, y + 58, "rendered once", 11.5, "middle", fill=MUTED, style="italic"))
    b.append(t(24, 140, "Sampled tokens are kept exactly as vLLM produced them; tool results and customer replies are appended as new tokens, never re-rendered,", 12.5, fill=INK2))
    b.append(t(24, 158, "so the trainer scores the same ids the sampler did. The loss covers only the policy's own tokens (mask 1).", 12.5, fill=INK2))

    b.append(t(24, 196, "(b) One GRPO group: 8 rollouts of the same task, same customer seed", 15, weight="bold"))
    rewards = [1.0, 0.7, 1.0, 0.2, 0.0, 1.0, 0.1, 1.0]
    mean = sum(rewards) / len(rewards)
    bx, base_y, hscale = 60, 306, 80
    b.append(f'<line x1="{bx - 10}" y1="{base_y}" x2="{bx + 8 * 46}" y2="{base_y}" stroke="{RULE}"/>')
    my = base_y - mean * hscale
    for i, r in enumerate(rewards):
        xx = bx + i * 46
        hh = max(r * hscale, 1.5)
        fill = OURS_C if r > mean else NEG_C
        b.append(f'<path d="M{xx},{base_y} L{xx},{base_y - hh + 4} Q{xx},{base_y - hh} {xx + 4},{base_y - hh} '
                 f'L{xx + 26},{base_y - hh} Q{xx + 30},{base_y - hh} {xx + 30},{base_y - hh + 4} L{xx + 30},{base_y} Z" fill="{fill}"/>')
        b.append(t(xx + 15, base_y - hh - 5, f"{r:.1f}", 11, "middle", fill=INK2))
    b.append(f'<line x1="{bx - 10}" y1="{my}" x2="{bx + 8 * 46}" y2="{my}" stroke="{INK}" stroke-dasharray="4 3"/>')
    b.append(t(bx + 8 * 46 + 6, my + 4, f"group mean {mean:.2f}", 12, fill=INK))
    tx = 560
    b.append(t(tx, 232, "advantage  Aᵢ = Rᵢ − mean(R)", 14, weight="bold"))
    b.append(t(tx, 254, "blue rollouts are pushed up, orange pushed down (no std scaling).", 12.5, fill=INK2))
    b.append(t(tx, 274, "The scripted customer is deterministic, so differences in a group come", 12.5, fill=INK2))
    b.append(t(tx, 292, "only from the policy. Groups where all 8 rewards are equal carry no", 12.5, fill=INK2))
    b.append(t(tx, 310, "signal and are masked out of the loss.", 12.5, fill=INK2))
    return svg(W, H, "".join(b))


# ---------------------------------------------------------------- figure 4 (results)


def task_successes(path: Path) -> dict[str, int]:
    d = json.loads(path.read_text())
    c: dict[str, int] = defaultdict(int)
    for s in d["simulations"]:
        c[s["task_id"]] += int((s.get("reward_info") or {}).get("reward", 0) >= 1 - 1e-6)
    return dict(c)


def pass_hat(succ: dict[str, int], k: int, n: int = 4) -> float:
    return sum(comb(c, k) / comb(n, k) for c in succ.values()) / len(succ)


def fig_results():
    base = task_successes(SIM / "tau_forge_baseline-luna-stoprule_retail_base.json")
    ours = task_successes(SIM / "tau_forge_step75-luna-stoprule_retail_base.json")
    metrics = [json.loads(l) for l in (RUN / "metrics.jsonl").open()]
    val = [json.loads(l) for l in (RUN / "val_log.jsonl").open()]

    W, H = 940, 380
    b = []
    # ---- (a) pass^k
    b.append(t(24, 28, "(a) τ²-bench retail, 114 tasks × 4 trials", 15, weight="bold"))
    ox, oy, ph, pw = 64, 320, 240, 340
    for v in (0, 20, 40, 60):
        yy = oy - v / 70 * ph
        b.append(f'<line x1="{ox}" y1="{yy}" x2="{ox + pw}" y2="{yy}" stroke="{RULE}" stroke-width="{1 if v else 1.2}"/>')
        b.append(t(ox - 8, yy + 4, f"{v}%", 11.5, "end", fill=MUTED))
    gw = pw / 4
    for k in range(1, 5):
        gx = ox + (k - 1) * gw + 12
        for j, (succ, col) in enumerate(((base, BASE_C), (ours, OURS_C))):
            v = round(100 * pass_hat(succ, k) + 1e-9, 1)
            hh = v / 70 * ph
            x = gx + j * 26
            b.append(f'<path d="M{x},{oy} L{x},{oy - hh + 4} Q{x},{oy - hh} {x + 4},{oy - hh} L{x + 20},{oy - hh} '
                     f'Q{x + 24},{oy - hh} {x + 24},{oy - hh + 4} L{x + 24},{oy} Z" fill="{col}"/>')
            b.append(t(x + 12, oy - hh - 5, f"{v:.1f}", 11, "middle", "bold" if j else "normal", INK if j else INK2))
        b.append(t(gx + 25, oy + 18, f"pass^{k}", 12.5, "middle", fill=INK2))
    lx = ox + 150
    for lab, col, yy in (("base Qwen3-4B", BASE_C, 52), ("step 75 (ours)", OURS_C, 70)):
        b.append(f'<rect x="{lx}" y="{yy - 10}" width="12" height="12" rx="2" fill="{col}"/>')
        b.append(t(lx + 18, yy, lab, 12, fill=INK2))
    b.append(t(ox, 360, "pass^1 +8.1 pts, paired-bootstrap 95% CI [+2.9, +13.4]", 12, fill=INK2))

    # ---- (c) training curve
    b.append(t(464, 28, "(b) Reward during training", 15, weight="bold"))
    ox3, pw3 = 504, 410
    steps = [m["step"] for m in metrics if m.get("episodes/mean_reward") is not None]
    rew = [m["episodes/mean_reward"] for m in metrics if m.get("episodes/mean_reward") is not None]
    lo, hi = 0.4, 1.0
    X = lambda s: ox3 + s / 85 * pw3  # noqa: E731
    Y = lambda r: oy - (r - lo) / (hi - lo) * ph  # noqa: E731
    for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        b.append(f'<line x1="{ox3}" y1="{Y(v)}" x2="{ox3 + pw3}" y2="{Y(v)}" stroke="{RULE}" stroke-width="{1.2 if v == lo else 1}"/>')
        b.append(t(ox3 - 8, Y(v) + 4, f"{v:.1f}", 11.5, "end", fill=MUTED))
    for s in (0, 25, 50, 75):
        b.append(t(X(s), oy + 18, str(s), 12, "middle", fill=INK2))
    b.append(t(ox3 + pw3, oy + 18, "step", 12, "end", style="italic", fill=MUTED))
    raw = " ".join(f"{X(s):.1f},{Y(r):.1f}" for s, r in zip(steps, rew))
    b.append(f'<polyline points="{raw}" fill="none" stroke="{BASE_C}" stroke-width="1.2"/>')
    win = 10
    ma = [(steps[i], sum(rew[max(0, i - win + 1):i + 1]) / len(rew[max(0, i - win + 1):i + 1])) for i in range(len(rew))]
    b.append(f'<polyline points="{" ".join(f"{X(s):.1f},{Y(r):.1f}" for s, r in ma)}" fill="none" stroke="{INK2}" stroke-width="2"/>')
    vp = [(v["step"], v["overall"]["mean_reward"]) for v in val]
    b.append(f'<polyline points="{" ".join(f"{X(s):.1f},{Y(r):.1f}" for s, r in vp)}" fill="none" stroke="{OURS_C}" stroke-width="2"/>')
    for s, r in vp:
        b.append(f'<circle cx="{X(s):.1f}" cy="{Y(r):.1f}" r="4.5" fill="{OURS_C}" stroke="#ffffff" stroke-width="2"/>')
    s75, r75 = vp[-1]
    b.append(t(X(s75) - 4, Y(r75) - 12, f"step 75 selected ({r75:.3f})", 11.5, "end", "bold", OURS_C))
    b.append(t(X(0) + 6, Y(vp[0][1]) - 10, f"{vp[0][1]:.3f}", 11.5, fill=OURS_C))
    ly = Y(0.985) + 4
    for lab, col, wdt in (("validation (163 held-out synthetic)", OURS_C, 2), ("training batch (10-step mean)", INK2, 2), ("training batch, per step", BASE_C, 1.2)):
        b.append(f'<line x1="{ox3 + 10}" y1="{ly - 4}" x2="{ox3 + 30}" y2="{ly - 4}" stroke="{col}" stroke-width="{wdt}"/>')
        b.append(t(ox3 + 36, ly, lab, 11.5, fill=INK2))
        ly += 17
    b.append(t(ox3, 360, "validation mean reward 0.695 → 0.787", 12, fill=INK2))
    return svg(W, H, "".join(b))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("overview", fig_overview), ("episode", fig_episode), ("training", fig_tokens), ("results", fig_results)):
        (OUT / f"{name}.svg").write_text(fn())
        print("wrote", OUT / f"{name}.svg")


if __name__ == "__main__":
    main()
