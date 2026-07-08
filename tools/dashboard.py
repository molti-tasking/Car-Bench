#!/usr/bin/env python
"""Generate the self-contained improvement-console HTML from experiments data.

Usage:  uv run python tools/dashboard.py   ->  experiments/dashboard.html

Reads runs.jsonl, prompts.py, and the latest judge report; emits a single HTML
file (inline CSS + SVG, no external requests) suitable for publishing as an
Artifact and for print-exporting figures into the technical report.
"""
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_FILE = REPO_ROOT / "experiments" / "runs.jsonl"
REPORTS_DIR = REPO_ROOT / "experiments" / "reports"
OUT_FILE = REPO_ROOT / "experiments" / "dashboard.html"

sys.path.insert(0, str(REPO_ROOT / "src" / "my_agent"))
from prompts import PROMPT_VARIANTS  # noqa: E402
sys.path.pop(0)

# Categorical slots (validated reference palette), fixed order — assigned to
# variants by first appearance in the registry, never re-shuffled.
SLOTS_LIGHT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
SLOTS_DARK = ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"]

CATEGORIES = [("base", "Base"), ("hallucination", "Hallucination"), ("disambiguation", "Disambiguation")]


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def load_runs() -> list[dict]:
    if not RUNS_FILE.exists():
        return []
    return [json.loads(l) for l in RUNS_FILE.read_text().splitlines() if l.strip()]


def git_subjects() -> dict[str, str]:
    """Map short sha -> commit subject for the whole repo history."""
    try:
        out = subprocess.run(
            ["git", "log", "--format=%h\t%s"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout
    except Exception:
        return {}
    subjects = {}
    for line in out.splitlines():
        sha, _, subject = line.partition("\t")
        subjects[sha] = subject
    return subjects


def variant_slots(runs: list[dict]) -> dict[str, int]:
    """Colored slots for the most recently active variants; others fold to
    gray ('retired') rather than cycling hues — 8 slots is the hard cap."""
    last_seen: dict[str, int] = {}
    for i, r in enumerate(runs):
        last_seen[r["variant"]] = i
    active = sorted(last_seen, key=lambda v: -last_seen[v])[:len(SLOTS_LIGHT)]
    first: dict[str, int] = {}
    for r in runs:
        first.setdefault(r["variant"], len(first))
    return {v: i for i, v in enumerate(sorted(active, key=lambda v: first[v]))}


def slot_cls(slots: dict[str, int], variant: str, prefix: str = "s") -> str:
    n = slots.get(variant)
    return f"{prefix}{n}" if n is not None else f"{prefix}ret"


def max_passk(d: dict | None) -> float | None:
    """Highest-k Pass^k / Pass@k value in a {'Pass^1': x, ...} dict."""
    if not d:
        return None
    key = sorted(d, key=lambda k: int(k.split("^")[-1].split("@")[-1]))[-1]
    return d[key]


def latest_comparable(runs: list[dict]) -> list[dict]:
    """Latest run per variant at the widest task width measured so far.

    Mixing widths misleads (small subsets score higher), so only variants
    measured at the maximum tasks_per_category compete for 'best'.
    """
    real = [r for r in runs if r["tasks_per_category"] not in (0, 1)]
    if not real:
        real = runs
    def _w(r):
        n = r["tasks_per_category"]
        return float("inf") if n == -1 else n
    max_width = max(_w(r) for r in real)
    widest = [r for r in real if _w(r) == max_width]
    by_variant: dict[str, dict] = {}
    for r in widest:
        by_variant[r["variant"]] = r  # registry is chronological; last wins
    return list(by_variant.values())


# ------------------------------------------------------------------ SVG charts

def bars_by_category(rows: list[dict], slots: dict[str, int]) -> str:
    """Grouped bars: x = category groups, one bar per variant, y = pass rate."""
    groups = [("Overall", lambda r: (r.get("pass_rate") or 0) / 100.0)]
    for key, label in CATEGORIES:
        groups.append((label, lambda r, k=key: (r.get("pass_power_k_by_split") or {}).get(k, {}) and max_passk(r["pass_power_k_by_split"][k]) or 0.0))

    W, H, PAD_L, PAD_B, PAD_T = 680, 300, 40, 46, 16
    plot_w, plot_h = W - PAD_L - 12, H - PAD_B - PAD_T
    n_groups, n_bars = len(groups), max(len(rows), 1)
    group_w = plot_w / n_groups
    bar_w = min(26, (group_w - 24) / n_bars)

    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Pass rate by task category and prompt variant">']
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = PAD_T + plot_h * (1 - frac)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-12}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{PAD_L-6}" y="{y+4:.1f}" class="tick" text-anchor="end">{int(frac*100)}%</text>')
    parts.append(f'<line x1="{PAD_L}" y1="{PAD_T+plot_h}" x2="{W-12}" y2="{PAD_T+plot_h}" class="baseline"/>')

    for gi, (glabel, fn) in enumerate(groups):
        gx = PAD_L + gi * group_w
        total_bars_w = n_bars * bar_w + (n_bars - 1) * 2
        x0 = gx + (group_w - total_bars_w) / 2
        for bi, r in enumerate(rows):
            v = fn(r) or 0.0
            bh = plot_h * v
            x = x0 + bi * (bar_w + 2)
            y = PAD_T + plot_h - bh
            c = slot_cls(slots, r["variant"])
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(bh,1.5):.1f}" rx="3" class="{c}">'
                f'<title>{esc(r["variant"])} — {glabel}: {v*100:.0f}% (run {esc(r["run_id"])})</title></rect>'
            )
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" class="val" text-anchor="middle">{v*100:.0f}</text>')
        parts.append(f'<text x="{gx+group_w/2:.1f}" y="{H-24}" class="cat" text-anchor="middle">{glabel}</text>')
    parts.append("</svg>")
    return "".join(parts)


def trajectory(runs: list[dict], slots: dict[str, int]) -> str:
    """Dot-line: x = run sequence, y = overall pass rate, colored by variant."""
    seq = [r for r in runs if r["tasks_per_category"] not in (0, 1)] or runs
    W, H, PAD_L, PAD_B, PAD_T = 680, 260, 40, 40, 14
    plot_w, plot_h = W - PAD_L - 16, H - PAD_B - PAD_T
    n = max(len(seq), 2)
    xs = lambda i: PAD_L + plot_w * (i / (n - 1)) if n > 1 else PAD_L + plot_w / 2
    ys = lambda v: PAD_T + plot_h * (1 - v / 100.0)

    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Pass rate per run over the campaign">']
    for frac in (25, 50, 75, 100):
        y = ys(frac)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-16}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{PAD_L-6}" y="{y+4:.1f}" class="tick" text-anchor="end">{frac}%</text>')
    parts.append(f'<line x1="{PAD_L}" y1="{PAD_T+plot_h}" x2="{W-16}" y2="{PAD_T+plot_h}" class="baseline"/>')

    # Commit markers: dashed rule wherever the code state changed between runs
    subjects = git_subjects()
    for i in range(1, len(seq)):
        prev, cur = seq[i - 1], seq[i]
        prev_state = (prev.get("git_sha"), prev.get("git_dirty"))
        cur_state = (cur.get("git_sha"), cur.get("git_dirty"))
        if cur_state != prev_state:
            x = (xs(i - 1) + xs(i)) / 2
            sha = cur.get("git_sha", "?")
            label = subjects.get(sha, "uncommitted changes" if cur.get("git_dirty") else "")
            parts.append(
                f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{PAD_T+plot_h}" class="commitmark">'
                f'<title>code change → {sha}{"*" if cur.get("git_dirty") else ""}: {esc(label)}</title></line>'
            )
            parts.append(f'<text x="{x:.1f}" y="{PAD_T-3}" class="tick" text-anchor="middle">{esc(sha)}</text>')

    by_variant: dict[str, list[tuple[int, dict]]] = {}
    for i, r in enumerate(seq):
        by_variant.setdefault(r["variant"], []).append((i, r))
    for v, pts in by_variant.items():
        c = slot_cls(slots, v)
        if len(pts) > 1:
            d = " ".join(f'{"M" if j==0 else "L"}{xs(i):.1f},{ys(r["pass_rate"] or 0):.1f}' for j, (i, r) in enumerate(pts))
            parts.append(f'<path d="{d}" class="line {c}" fill="none"/>')
        for i, r in pts:
            parts.append(
                f'<circle cx="{xs(i):.1f}" cy="{ys(r["pass_rate"] or 0):.1f}" r="5" class="dot {c}">'
                f'<title>{esc(v)} — {r["pass_rate"]:.0f}% | {esc(r["run_id"])} | {r["tasks_per_category"]} tasks/cat × {r["num_trials"]} trials</title></circle>'
            )
        i_last, r_last = pts[-1]
        parts.append(f'<text x="{xs(i_last)+8:.1f}" y="{ys(r_last["pass_rate"] or 0)-8:.1f}" class="val">{esc(v)}</text>')
    for i, r in enumerate(seq):
        parts.append(f'<text x="{xs(i):.1f}" y="{H-18}" class="tick" text-anchor="middle">{i+1}</text>')
    parts.append(f'<text x="{PAD_L+plot_w/2:.1f}" y="{H-4}" class="cat" text-anchor="middle">run sequence</text>')
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ page

def build() -> str:
    runs = load_runs()
    slots = variant_slots(runs)
    comparable = latest_comparable(runs)
    best = max(comparable, key=lambda r: r.get("pass_rate") or 0) if comparable else None
    import os
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    user_sim = (os.getenv("USER_SIM_MODEL") or "gemini-2.5-flash").split("/")[-1]
    judge = (os.getenv("JUDGE_MODEL") or "?").split("/")[-1]
    total_tokens = sum(r.get("total_tokens") or 0 for r in runs)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    slot_css_light = "\n".join(f".s{i}{{fill:{c}}} path.s{i}{{stroke:{c}}} .sw{i}{{background:{c}}}" for i, c in enumerate(SLOTS_LIGHT))
    slot_css_dark = "\n".join(f".s{i}{{fill:{c}}} path.s{i}{{stroke:{c}}} .sw{i}{{background:{c}}}" for i, c in enumerate(SLOTS_DARK))

    legend = "".join(
        f'<span class="lg"><i class="sw {slot_cls(slots, v, 'sw')}"></i>{esc(v)}</span>'
        for v in dict.fromkeys([r["variant"] for r in comparable])
    )

    table_rows = "".join(
        f'<tr><td class="mono">{esc(r["run_id"])}</td><td><i class="sw {slot_cls(slots, r["variant"], 'sw')}"></i> {esc(r["variant"])}</td>'
        f'<td class="mono">{esc(r["model"].split("/")[-1])}</td><td>{esc(r["split"])}</td>'
        f'<td class="num">{r["tasks_per_category"]}</td><td class="num">{r["num_trials"]}</td>'
        f'<td class="num">{(r["pass_rate"] or 0):.1f}%</td>'
        f'<td class="num">{(max_passk(r.get("pass_power_k")) or 0)*100:.0f}%</td>'
        f'<td class="num">{(r.get("total_tokens") or 0)//1000}k</td>'
        f'<td class="mono">{esc(r.get("git_sha",""))}{"*" if r.get("git_dirty") else ""}</td></tr>'
        for r in reversed(runs)
    )

    variant_cards = "".join(
        f'<div class="vcard"><div class="vhead"><i class="sw {slot_cls(slots, v, 'sw')}"></i><b>{esc(v)}</b></div>'
        f'<pre class="prompt">{esc((spec["prefix"] + spec["suffix"]).strip() or "(evaluator prompt unchanged)")}</pre></div>'
        for v, spec in PROMPT_VARIANTS.items()
    )

    # Improvement log: each code state (sha) with its runs and score movement
    subjects = git_subjects()
    log_rows = []
    seen_states: list[tuple] = []
    state_runs: dict[tuple, list[dict]] = {}
    for r in runs:
        state = (r.get("git_sha"), bool(r.get("git_dirty")))
        if state not in state_runs:
            seen_states.append(state)
            state_runs[state] = []
        state_runs[state].append(r)
    prev_best = None
    for state in seen_states:
        sha, dirty = state
        rs = state_runs[state]
        best_here = max((x.get("pass_rate") or 0) for x in rs)
        delta = "" if prev_best is None else f'{best_here - prev_best:+.1f} pp'
        delta_cls = "" if prev_best is None else ("up" if best_here >= prev_best else "down")
        prev_best = best_here
        subject = subjects.get(sha, "")
        if dirty:
            subject = (subject + " + uncommitted edits").strip(" +")
        runs_str = ", ".join(f'{esc(x["variant"])} {x["pass_rate"]:.0f}%' for x in rs)
        log_rows.append(
            f'<tr><td class="mono">{esc(sha)}{"*" if dirty else ""}</td>'
            f'<td>{esc(subject or "—")}</td><td>{runs_str}</td>'
            f'<td class="num">{best_here:.0f}%</td><td class="num {delta_cls}">{delta or "—"}</td></tr>'
        )
    improvement_log = (
        '<section><h2>Improvement log <span class="mono muted">git ⨯ runs</span></h2>'
        '<div class="tablewrap"><table>'
        '<thead><tr><th>code state</th><th>change</th><th>runs under this state</th>'
        '<th>best pass</th><th>Δ best</th></tr></thead>'
        f'<tbody>{"".join(log_rows)}</tbody></table></div>'
        '<p class="footnote">* = working tree had uncommitted changes during the run. Each prompt/harness '
        'edit is committed before its round, so every score is attributable to an exact code state.</p></section>'
    )

    # Benchmark-validity: simulator-noise audit (optional; tool is separate)
    simulator_section = ""
    try:
        from simulator_audit import audit as run_sim_audit
        a = run_sim_audit()
        if a["total_fail"]:
            sims = sorted(a["by_sim"].items(), key=lambda kv: -(kv[1]["misfire"] / kv[1]["fail"] if kv[1]["fail"] else 0))
            barW, gap, maxh = 90, 28, 150
            bars = []
            for i, (sim, s) in enumerate(sims):
                rate = (s["misfire"] / s["fail"]) if s["fail"] else 0
                h = maxh * rate
                x = 60 + i * (barW + gap)
                y = 20 + (maxh - h)
                small = s["fail"] < 10
                bars.append(
                    f'<rect x="{x}" y="{y:.0f}" width="{barW}" height="{max(h,2):.0f}" rx="3" '
                    f'class="{"sret" if small else "sbad"}"><title>{esc(sim)}: {s["misfire"]}/{s["fail"]} failures are misfires'
                    f'{" (n too small)" if small else ""}</title></rect>'
                    f'<text x="{x+barW/2}" y="{y-6:.0f}" class="val" text-anchor="middle">{rate*100:.0f}%</text>'
                    f'<text x="{x+barW/2}" y="185" class="cat" text-anchor="middle">{esc(sim)}</text>'
                    f'<text x="{x+barW/2}" y="199" class="tick" text-anchor="middle">n={s["fail"]} fails</text>'
                )
            chart = (
                f'<svg class="chart" viewBox="0 0 {60 + len(sims)*(barW+gap)} 210" role="img" '
                f'aria-label="Simulator misfire rate by model">'
                + "".join(f'<line x1="60" y1="{20+maxh*(1-f):.0f}" x2="{40+len(sims)*(barW+gap)}" y2="{20+maxh*(1-f):.0f}" class="grid"/>'
                          f'<text x="54" y="{24+maxh*(1-f):.0f}" class="tick" text-anchor="end">{int(f*100)}%</text>'
                          for f in (0.25, 0.5, 0.75, 1.0))
                + "".join(bars) + "</svg>"
            )
            frag_counts = {}
            for _, tid, split, *_ in a["fragile"]:
                frag_counts[tid] = frag_counts.get(tid, 0) + 1
            frag_rows = "".join(
                f'<tr><td class="mono">{esc(t)}</td><td class="num">{c}</td></tr>'
                for t, c in sorted(frag_counts.items(), key=lambda kv: -kv[1])
            )
            simulator_section = (
                '<section><h2>Benchmark validity — simulator noise '
                '<span class="mono muted">tools/simulator_audit.py</span></h2>'
                '<div class="figures"><figure>' + chart +
                '<figcaption><b>Figure.</b> Share of an agent\'s <em>failures</em> that are simulator '
                'misfires — the user-sim ending the conversation on turn one, before the agent acts. '
                'Gray = sample too small to judge (the official Gemini sim is barely represented here).'
                '</figcaption></figure>'
                '<figure><table><thead><tr><th>chronically fragile task</th><th>configs affected</th></tr></thead>'
                f'<tbody>{frag_rows}</tbody></table>'
                '<figcaption><b>Table.</b> Tasks that flip pass↔misfire across independent agent '
                'configurations — the flip is the simulator, not the agent.</figcaption></figure></div>'
                f'<p class="footnote">{a["total_misfire"]}/{a["total_fail"]} failures '
                f'({a["total_misfire"]/a["total_fail"]*100:.1f}%) are simulator artifacts across '
                f'{a["total_rows"]} trial rows; {a["p3_broken_misfire_only"]} Pass^3 breaks would flip to '
                'pass without them. Misfire rate is strongly model-dependent — a fixable validity axis, '
                'not an agent property.</p></section>'
            )
    except Exception:
        simulator_section = ""

    latest_report = ""
    if REPORTS_DIR.exists():
        reports = sorted(REPORTS_DIR.glob("*.md"))
        if reports:
            latest_report = (
                f'<section><h2>Latest failure analysis <span class="mono muted">{esc(reports[-1].name)}</span></h2>'
                f'<pre class="report">{esc(reports[-1].read_text()[:6000])}</pre></section>'
            )

    stat = lambda label, value, sub: (
        f'<div class="tile"><div class="tlabel">{label}</div><div class="tvalue">{value}</div><div class="tsub">{sub}</div></div>'
    )
    tiles = "".join([
        stat("Best variant", esc(best["variant"]) if best else "—",
             f'{best["pass_rate"]:.0f}% pass rate' if best else "no runs yet"),
        stat("Runs recorded", str(len(runs)), f"last: {esc(runs[-1]['run_id'])}" if runs else ""),
        stat("Agent model", esc((runs[-1]["model"].split("/")[-1]) if runs else "—"), "self-hosted via LiteLLM proxy"),
        stat("Tokens spent", f"{total_tokens/1e6:.1f}M", "agent-side, all runs"),
    ])

    return f"""<title>CAR-bench Improvement Console</title>
<style>
:root {{
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --baseline:#c3c2b7; --border:rgba(11,11,11,.10);
  --good:#006300; --bad:#d03b3b;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --plane:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
          --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,.10); --good:#0ca30c; }}
}}
:root[data-theme="light"] {{ --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --grid:#e1e0d9; --baseline:#c3c2b7; --border:rgba(11,11,11,.10); --good:#006300; }}
:root[data-theme="dark"] {{ --plane:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
  --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,.10); --good:#0ca30c; }}

{slot_css_light}
@media (prefers-color-scheme: dark) {{ {slot_css_dark} }}
:root[data-theme="light"] body {{ }}
{"".join("")}
body {{ background:var(--plane); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
       margin:0; padding:28px 20px 60px; }}
main {{ max-width:1140px; margin:0 auto; display:flex; flex-direction:column; gap:28px; }}
h1 {{ font-size:22px; font-weight:650; letter-spacing:-.01em; margin:0; text-wrap:balance; }}
h2 {{ font-size:15px; font-weight:650; margin:0 0 12px; }}
header .sub {{ color:var(--ink2); margin-top:4px; font-size:13.5px; }}
.eyebrow {{ text-transform:uppercase; letter-spacing:.07em; font-size:11px; color:var(--muted); font-weight:600; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
.tile {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }}
.tlabel {{ text-transform:uppercase; letter-spacing:.07em; font-size:10.5px; color:var(--muted); font-weight:600; }}
.tvalue {{ font-size:24px; font-weight:650; margin-top:2px; }}
.tsub {{ font-size:12px; color:var(--ink2); margin-top:2px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.figures {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:900px) {{ .figures {{ grid-template-columns:1fr; }} }}
figure {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px; margin:0; }}
figcaption {{ font-size:12.5px; color:var(--ink2); margin-top:10px; }}
figcaption b {{ color:var(--ink); }}
.chart {{ width:100%; height:auto; display:block; }}
.chart .grid {{ stroke:var(--grid); stroke-width:1; }}
.chart .baseline {{ stroke:var(--baseline); stroke-width:1; }}
.chart .tick {{ fill:var(--muted); font-size:10.5px; font-family:ui-monospace,monospace; }}
.chart .cat {{ fill:var(--ink2); font-size:11.5px; font-weight:600; }}
.chart .val {{ fill:var(--ink2); font-size:10.5px; font-family:ui-monospace,monospace; }}
.chart .line {{ stroke-width:2; opacity:.85; }}
.chart .dot {{ stroke:var(--surface); stroke-width:2; }}
.chart .commitmark {{ stroke:var(--baseline); stroke-width:1; stroke-dasharray:3 3; }}
td.up {{ color:var(--good); }}
td.down {{ color:var(--bad); }}
.footnote {{ font-size:12px; color:var(--muted); margin:10px 0 0; }}
.chart rect.sbad {{ fill:#d03b3b; }}
.chart rect.sret {{ fill:#898781; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; margin-bottom:8px; }}
.lg {{ display:inline-flex; align-items:center; gap:6px; font-size:12.5px; color:var(--ink2); }}
.sw {{ width:10px; height:10px; border-radius:3px; display:inline-block; flex:none; }}
section {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 18px; }}
.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th {{ text-align:left; text-transform:uppercase; letter-spacing:.06em; font-size:10.5px; color:var(--muted);
     border-bottom:1px solid var(--baseline); padding:6px 10px 6px 0; }}
td {{ border-bottom:1px solid var(--grid); padding:7px 10px 7px 0; vertical-align:top; }}
td.num {{ font-variant-numeric:tabular-nums; text-align:right; font-family:ui-monospace,monospace; font-size:12.5px; }}
th:nth-child(n+5) {{ text-align:right; }}
.mono {{ font-family:ui-monospace,monospace; font-size:12px; }}
.muted {{ color:var(--muted); font-weight:400; }}
.vgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; }}
.vcard {{ border:1px solid var(--border); border-radius:8px; padding:12px 14px; }}
.vhead {{ display:flex; align-items:center; gap:8px; margin-bottom:8px; }}
pre.prompt, pre.report {{ white-space:pre-wrap; font:12px/1.5 ui-monospace,monospace; color:var(--ink2);
  margin:0; max-height:220px; overflow-y:auto; }}
pre.report {{ max-height:420px; }}
.arch {{ display:flex; align-items:stretch; gap:0; flex-wrap:wrap; }}
.abox {{ border:1px solid var(--baseline); border-radius:8px; padding:10px 14px; flex:1; min-width:200px; }}
.abox b {{ display:block; font-size:13px; }}
.abox span {{ font-size:12px; color:var(--ink2); }}
.arrow {{ align-self:center; padding:0 10px; color:var(--muted); font-size:18px; }}
@media print {{
  body {{ background:#fff; color:#0b0b0b; padding:0; }}
  section, .tiles, header .sub {{ display:none; }}
  .figures {{ grid-template-columns:1fr; gap:24px; }}
  figure {{ border:none; padding:0; page-break-inside:avoid; }}
  #arch-section {{ display:block; border:none; }}
}}
</style>
<main>
<header>
  <div class="eyebrow">CAR-bench · IJCAI-ECAI 2026 · Track 1</div>
  <h1>Agent Improvement Console</h1>
  <div class="sub">{esc((runs[-1]["model"].split("/")[-1]) if runs else "—")} agent under test · user-sim {esc(user_sim)} · judges {esc(judge)} · generated {now}</div>
</header>

<div class="tiles">{tiles}</div>

<div class="legend">{legend}</div>
<div class="figures">
<figure>
  {bars_by_category(comparable, slots)}
  <figcaption><b>Figure 1.</b> Pass rate by task category per prompt variant — latest comparable run each
  (train split). Hallucination and disambiguation are the reliability categories the harness targets.</figcaption>
</figure>
<figure>
  {trajectory(runs, slots)}
  <figcaption><b>Figure 2.</b> Campaign trajectory: overall pass rate per benchmark run, in execution order.
  Hover a point for run details.</figcaption>
</figure>
</div>

<section>
<h2>Run registry <span class="mono muted">experiments/runs.jsonl</span></h2>
<div class="tablewrap">
<table>
<thead><tr><th>run</th><th>variant</th><th>model</th><th>split</th><th>tasks/cat</th><th>trials</th><th>pass rate</th><th>Pass^k</th><th>tokens</th><th>git</th></tr></thead>
<tbody>{table_rows}</tbody>
</table>
</div>
</section>

{improvement_log}

{simulator_section}

<section>
<h2>Prompt variants under test</h2>
<div class="vgrid">{variant_cards}</div>
</section>

<section id="arch-section">
<h2>System architecture</h2>
<div class="arch">
  <div class="abox"><b>Evaluation engine</b><span>official CAR-bench evaluator · simulated user + policy judge (Kimi-K2.5 via proxy locally, Gemini officially) · owns tools, state, scoring</span></div>
  <div class="arrow">⇄</div>
  <div class="abox"><b>Agent under test</b><span>GLM-5.2 via LiteLLM proxy · evaluator policy prompt wrapped by env-selected variant · A2A protocol · ships as Docker image</span></div>
  <div class="arrow">→</div>
  <div class="abox"><b>Improvement toolkit</b><span>run → registry → LLM-judge failure clustering → prompt/harness edits · Langfuse traces both sides · never ships</span></div>
</div>
</section>

{latest_report}
</main>
"""


if __name__ == "__main__":
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(build())
    print(f"dashboard written to {OUT_FILE}")
