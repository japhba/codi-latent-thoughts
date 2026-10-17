"""Figure: LogitLens vs J-Lens top-5 token tables over one CODI latent rollout.

Two panels (LogitLens left, J-Lens right). Each is a (hidden-state index x position)
table: positions on the x-axis, layers on the y-axis with the first layer at the
bottom. Each cell stacks the top-5 decoded tokens (tiny monospace). Tokens are matched by
the complete number they express (numerals, number words in any language, via the companion
package's gold-blind token dictionary plus a deterministic parser), the criterion of the
population figures: the gold answer is tinted dark green, a computed intermediate (a value of
the reference solution other than the answer that is not given in the question, as numeral or word) light green,
both bolded, so color is never the only encoding.
TWO outputs: a compact main-text version (rows h1, h24, h30, h36, z x columns Enc, z1, z6)
and the full strided grid for the appendix (env FIG_OUT / FIG_OUT_FULL).
Example selection (auto-pick, first in dataset order): CODI correct, 2-3 hops, answer not
given in the question; in the main-text cells the answer and a multi-digit computed
intermediate (so a lone digit cannot match by chance) appear in the h30/h36 rows, and
neither answer nor any intermediate appears in the h1, h24 or z rows.

CPU-only. Reads the corrected readouts in artifacts/ (codi_repro.lenses
readouts): top-5 token ids for 1200 saved CODI rollouts x 37 hidden
states x 7 positions, LogitLens and J-lens (recovered CODI-specific Jacobian maps).
Conventions there: hidden_states[0] = the exact fed input (the z row at thought
positions), 1..35 = pre-final-norm block outputs decoded via unembed(norm(.)),
36 = post-final-norm output decoded directly (the earlier version of this figure
normalized it a second time). J-lens transports state l with J_{l-1} (J_emb for the
fed input); the final state is shared by construction.
Output: figs/lens_dual.png and lens_dual_full.png (headless chromium via playwright)
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import tempfile
import unicodedata

import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from codi_repro.all_layer_analysis import dictionary, literal  # noqa: E402
from codi_repro.analysis import canonical_number  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from html_render import screenshot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.environ.get("LENS_ART", os.path.join(HERE, "..", "artifacts"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs", "lens_dual.png"))
OUT_FULL = os.environ.get("FIG_OUT_FULL", os.path.join(HERE, "..", "figs", "lens_dual_full.png"))
PICK = int(os.environ.get("PICK_ROW", "-1"))    # -1 = auto-select

# hidden-state indices to display (1 = first block output ... 36 = final; 0 = embeddings
# excluded). Strided every 3, plus 24 (NLA read layer) and 27 (AO read band layer).
ROWS = [1, 4, 7, 10, 13, 16, 19, 22, 24, 27, 30, 33, 36]
POS = ["Enc"] + [rf"$\mathbf{{z}}_{k}$" for k in range(1, 7)]
TOPK = 5

# palette: text stays ink, tints carry match status (bold text as the secondary
# encoding). Answers dark green, intermediates light green, post-wide convention.
TINT_ANSWER = "#8fd19c"      # dark green fill
TINT_CHAIN = "#ddf1e0"       # light green fill
INK = "#1a1a1a"


def display(tok_str):
    """A token as shown: real glyphs in any script (Chrome shapes them); visible markers for spaces and
    newlines; a gray code point only for characters that render as nothing (controls, variation selectors)."""
    out = []
    for ch in tok_str:
        cat = unicodedata.category(ch)
        if ch == " ":
            out.append("␣")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif cat in ("Cc", "Cf", "Mn") and not ("\u0e00" <= ch <= "\u0e7f") or 0xfe00 <= ord(ch) <= 0xfe0f:
            out.append(f'<span class="cp">\\u{ord(ch):04x}</span>')
        else:
            out.append(html.escape(ch))
    shown = "".join(out)
    plain = re.sub(r"<[^>]+>", "", shown)
    return shown if len(plain) <= 12 else html.escape(tok_str[:11]) + "…"


WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
              "hundred": 100, "thousand": 1000})
IMPLIED = {"half": 2, "halves": 2, "twice": 2, "double": 2, "doubled": 2, "triple": 3, "tripled": 3, "thrice": 3,
           "quarter": 4, "dozen": 12}  # number words that state a given without a numeral
HANZI = dict(zip("零一二两三四五六七八九十百千万", [0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000]))
SEMANTICS = dictionary()


def token_values(tok_str):
    """complete numbers a single token expresses"""
    s = tok_str.strip()
    if s in SEMANTICS:
        return set(SEMANTICS[s])
    if (v := literal(s)) is not None:
        return {v}
    w = re.sub(r"[^a-z]", "", s.lower())
    if w in WORDS and len(w) == len(re.sub(r"[^A-Za-z]", "", s)):
        return {canonical_number(str(WORDS[w]))}
    if s in HANZI:
        return {canonical_number(str(HANZI[s]))}
    return set()


def gold_sets(m):
    """canonical gold answer and computed intermediates (chain values not given in the question)"""
    given = {canonical_number(x) for x in re.findall(r"\d+(?:\.\d+)?", m["question"].replace(",", ""))}
    words = re.findall(r"[a-z]+", m["question"].lower())
    given |= {canonical_number(str(WORDS[w])) for w in words if w in WORDS}
    given |= {canonical_number(str(IMPLIED[w])) for w in words if w in IMPLIED}
    answer = canonical_number(m["gold"])
    return answer, {canonical_number(v) for v in m["chain"]} - {answer} - given, given


def token_status(t, answer, computed):
    """'answer' | 'chain' | None for one raw token string."""
    v = token_values(t)
    if answer in v:
        return "answer"
    if v & computed:
        return "chain"
    return None


def main():
    meta = json.load(open(os.path.join(ART, "lens_problems.json")))
    top = np.load(os.path.join(ART, "lens_readouts.npz"))["top_ids"]  # [LL/JL, 37, n, 7, 5]
    strings = json.load(open(os.path.join(ART, "lens_token_strings.json")))
    decode = lambda ids: [strings[str(int(t))] for t in ids]

    if PICK >= 0:
        i = PICK
    else:
        def cell_values(c, hs, cols):
            return set().union(*(token_values(t) for k in (0, 1) for col in cols for t in decode(top[k, hs, c, col])))
        def passes(c, m):
            answer, computed, given = gold_sets(m)
            if not (m["correct"] and 2 <= m["hops"] <= 3 and computed and answer not in given):
                return False
            late = cell_values(c, 30, (0, 1, 6)) | cell_values(c, 36, (0, 1, 6))
            early = cell_values(c, 1, (0, 1, 6)) | cell_values(c, 24, (0, 1, 6)) | cell_values(c, 0, (1, 6))
            return (answer in late and any(len(v.replace(".", "")) >= 2 for v in computed & late)
                    and not ({answer} | set(canonical_number(x) for x in m["chain"])) & early)
        i = next(c for c, m in enumerate(meta) if passes(c, m))
    m = meta[i]
    answer, computed, _ = gold_sets(m)
    print(f"[pick] row {i} ds_index {m['ds_index']} gold={m['gold']} chain={m['chain']} computed={sorted(computed)}")
    print(f"[pick] question: {m['question']}")

    cells = {}                                                        # (panel, hs, pos)
    for k, panel in enumerate(("LL", "JL")):
        for hs in ROWS:
            for pi in range(7):
                cells[(panel, hs, pi)] = decode(top[k, hs, i, pi])
        for pi in range(1, 7):                                        # fed input; none at Enc
            cells[(panel, "z", pi)] = decode(top[k, 0, i, pi])
    print("[z-row] LL z1 top-5:", cells[("LL", "z", 1)])
    print("[h36] LL z1 top-5:", cells[("LL", 36, 1)])

    def render(yrows, pos_idx, out, scale):
        """One dual-panel figure over display rows (bottom -> top; 'z' = fed-back thought) and position
        columns, as HTML screenshotted by headless Chrome: it shapes every script (Arabic joined and
        right to left, Thai marks placed), with the Noto fonts as fallbacks."""
        label = lambda hs: "<b>z</b>" if hs == "z" else f"<b>h</b><sup>({hs})</sup>"
        col = lambda pi: "Enc" if pi == 0 else f"<b>z</b><sub>{pi}</sub>"
        titles = {"LL": "LogitLens: <b>h</b><sup>(ℓ)</sup>",
                  "JL": "J-Lens: <i>J</i><sub>ℓ</sub><b>h</b><sup>(ℓ)</sup>, <i>J</i><sub>ℓ</sub> ≈ "
                        "E[∂<b>h</b><sup>(36)</sup>/∂<b>h</b><sup>(ℓ)</sup>]"}
        panels = []
        for panel in ("LL", "JL"):
            rows = []
            for hs in reversed(yrows):                                # top row first
                tds = []
                for pi in pos_idx:
                    toks = cells.get((panel, hs, pi))
                    if toks is None:                                  # z row @ Enc: nothing fed
                        tds.append('<td class="cell"></td>')
                        continue
                    lines = []
                    for t in toks:
                        status = token_status(t, answer, computed)
                        cls = {"answer": "answer", "chain": "chain"}.get(status, "")
                        lines.append(f'<div><bdi class="{cls}">{display(t)}</bdi></div>')
                    tds.append(f'<td class="cell">{"".join(lines)}</td>')
                rows.append(f'<tr class="{"z" if hs == "z" else ""}"><th class="row">{label(hs)}</th>{"".join(tds)}</tr>')
            head = "".join(f'<th class="col">{col(pi)}</th>' for pi in pos_idx)
            panels.append(f'<div class="panel"><h2>{titles[panel]}</h2><table>{"".join(rows)}'
                          f'<tr><th></th>{head}</tr></table><div class="xlab">thought position</div></div>')
        n_c = len(computed)
        title = (f"GSM8k-Aug-NL #{m['ds_index']}: gold answer {m['gold']}, computed intermediate"
                 f"{'s' if n_c > 1 else ''} {', '.join(sorted(computed, key=float))}")
        mono = ('"DejaVu Sans Mono", "Noto Sans Mono CJK SC", "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans Thai", '
                '"Noto Sans Syriac", "Noto Sans Arabic", "Noto Sans Symbols2", "Noto Sans Symbols", monospace')
        page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin: 0; background: white; color: {INK}; font-family: "DejaVu Sans", "Noto Sans", sans-serif; }}
.wrap {{ display: inline-block; padding: 18px 26px 14px; }}
h1 {{ font-weight: normal; font-size: {round(20 * scale)}px; text-align: center; margin: 0 0 12px; }}
.panels {{ display: flex; gap: 34px; align-items: flex-start; }}
.side {{ writing-mode: vertical-rl; transform: rotate(180deg); text-align: center; font-size: {round(15 * scale)}px; }}
.panel h2 {{ font-weight: normal; font-size: {round(17 * scale)}px; text-align: center; margin: 0 0 6px; height: 1.7em; line-height: 1.7em; }}
table {{ border-collapse: collapse; }}
td.cell {{ border: 1px solid #dddddd; padding: {round(3 * scale)}px {round(9 * scale)}px; text-align: center;
          vertical-align: middle; min-width: {round(104 * scale)}px; }}
td.cell div {{ font-family: {mono}; font-size: {round(11 * scale)}px; height: 1.5em; line-height: 1.5em; white-space: pre; }}
tr.z td.cell {{ border-bottom: 2px solid #999999; }}
.answer {{ background: {TINT_ANSWER}; font-weight: bold; padding: 0 2px; }}
.chain {{ background: {TINT_CHAIN}; font-weight: bold; padding: 0 2px; }}
.cp {{ color: #8a8a8a; }}
th.row {{ font-size: {round(17 * scale)}px; text-align: right; padding-right: 8px; white-space: nowrap; }}
th.col {{ font-size: {round(17 * scale)}px; font-weight: normal; padding-top: 4px; }}
.xlab {{ text-align: center; font-size: {round(15 * scale)}px; margin-top: 2px; }}
.legend {{ text-align: center; font-size: {round(15 * scale)}px; margin-top: 12px; }}
.sw {{ display: inline-block; width: 30px; height: 14px; border: 1px solid #bbbbbb; vertical-align: middle; margin: 0 7px 0 26px; }}
</style></head><body><div class="wrap"><h1>{html.escape(title)}</h1>
<div class="panels"><div class="side">hidden state; <b>z</b> = fed-back thought vector</div>{"".join(panels)}</div>
<div class="legend"><span class="sw" style="background:{TINT_ANSWER}"></span>token expresses the gold answer
<span class="sw" style="background:{TINT_CHAIN}"></span>token expresses a computed intermediate result</div>
</div></body></html>"""
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            src, shot = os.path.join(tmp, "grid.html"), os.path.join(tmp, "grid.png")
            open(src, "w", encoding="utf-8").write(page)
            screenshot(f"file://{src}", shot)
            img = Image.open(shot).convert("RGB")
        bbox = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
        img.crop((max(bbox[0] - 20, 0), max(bbox[1] - 20, 0), bbox[2] + 20, bbox[3] + 20)).save(out)
        print(f"wrote {out}")

    # main text: 5 display rows (h1, h24, h30, h36, z) x 3 columns (Enc, z1, z6);
    # appendix: the full strided grid over all seven columns
    render([1, 24, 30, 36, "z"], [0, 1, 6], OUT, scale=1.0)
    render(ROWS + ["z"], list(range(7)), OUT_FULL, scale=0.85)


if __name__ == "__main__":
    main()
