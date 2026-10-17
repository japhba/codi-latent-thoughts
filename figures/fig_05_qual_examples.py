"""Qualitative showcase example rendered as ONE styled HTML card (post: Qualitative
analysis of NLA verbalizations): a SINGLE problem (g5087, apples and oranges, 36 fruits,
twice as many oranges: gold answer 12) verbalized by BOTH nz4 readers at its SECOND
thought, chosen as a success case: in the greedy reads of the 60 z-scanned problems the z reader's
predicted "final token" at thought 2 is the gold answer in 10 (none of them a given
value), and here it is in both samples, while the h24 reader writes out the problem's
equation. Columns = training steps, left to right (step 0 = warmstart, step 3200 =
final); sections = readers (z reader on z_2, h24 reader on h_2^(24)); rows = two samples
at the training rollout temperature (T = 1, scripts/nz4_qual_resample.py), so no
reading rests on a single decode.
Each cell leads with an italic one-line digest, tagged by a digest pill (editorial summaries,
artifacts/qual_nla_tldr.json, keyed by the sha1 of the verbatim text so a changed gen
fails loudly) above the verbatim verbalization (closing </explanation> tag dropped).
Numbers are highlighted by canonical value (mechanical, same scheme as the AO card
fig_23):
  ans   = gold answer
  gold  = gold intermediate or a number given in the problem
  fab   = any other number

Inputs: artifacts/nz4_qual_resample_{z,h24}.jsonl, artifacts/qual_nla_tldr.json,
        nz4_qual_meta.jsonl (the candidates' rows of the reader corpus's gsm_meta.jsonl, env NLA_META). Output figs/qual_nla_duo.png (env FIG_DIR).
"""
import hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from html_render import esc, mark_words, render_card

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
META_F = os.environ.get("NLA_META", os.path.join(ART, "nz4_qual_meta.jsonl"))
FIG_DIR = os.environ.get("FIG_DIR", os.path.join(HERE, "..", "figs"))
UID = os.environ.get("QUAL_UID", "g5087")
T = int(os.environ.get("QUAL_T", "1"))           # thought index 0..5, shown as t + 1
STEPS = {0: "warm-start", 3200: "after training"}
READERS = {"z": ('<i>z</i> reader', f'<i>z</i><sub>{T + 1}</sub>', "z"),
           "h24": ('<i>h</i> reader', f'<i>h</i><sub>{T + 1}</sub><sup>(24)</sup>', "h")}
LEGEND = {"ans": "gold answer", "gold": "gold intermediate or given", "fab": "other number"}


def canon(s):
    f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"


META = {m["uid"]: m for m in map(json.loads, open(META_F))}
m = META[UID]
ANS = {canon(m["answer"])}
GOLD = {canon(v) for v in m["intermediates"]}
for mt in re.findall(r"-?\d[\d,]*(?:\.\d+)?", m["question"]):
    GOLD.add(canon(mt))
GOLD -= ANS

gens = {}   # (src, step, k) -> verbatim gen, k = sample index
for src in READERS:
    for r in map(json.loads, open(os.path.join(ART, f"nz4_qual_resample_{src}.jsonl"))):
        if (r["uid"], r["t"], r["decode"]) == (UID, T, "sample"):
            gens[(src, r["step"], r["k"])] = r["gen"]
KS = sorted({k for (_, _, k) in gens})
assert len(gens) == len(READERS) * len(STEPS) * len(KS), sorted(gens)
TLDR = {t["sha1"]: t["tldr"] for t in json.load(open(os.path.join(ART, "qual_nla_tldr.json")))}


def word_cats(text):
    words = text.split()
    norm = " ".join(words)
    starts, pos = [], 0
    for w in words:
        starts.append(pos)
        pos += len(w) + 1
    cats = [None] * len(words)
    for mt in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", norm):
        try:
            c = canon(mt.group())
        except (ValueError, OverflowError):
            continue
        cat = "ans" if c in ANS else "gold" if c in GOLD else "fab"
        for i, st in enumerate(starts):
            if st < mt.end() and st + len(words[i]) > mt.start():
                cats[i] = cats[i] or cat
    return cats


def marked(text):
    flat = " ".join(text.split())
    return mark_words(flat, word_cats(flat))


def gen_html(text):
    text = re.sub(r"\s*</explanation>\s*$", "", text)
    return "\n\n".join(marked(p) for p in text.split("\n\n") if p.split())


CSS = """<style>
#card { width: 1120px; }
.top { display: flex; justify-content: space-between; align-items: flex-end; gap: 24px;
       border-bottom: 1px solid #e5e5ea; padding-bottom: 10px; margin-bottom: 6px; }
.top .sub { margin: 0; }
.top .legend { border: 0; margin: 0; padding: 0; white-space: nowrap; }
.grid { display: grid; grid-template-columns: 1fr 1fr; column-gap: 26px; }
.colhead { font-size: 15px; font-weight: 700; color: #1a1a1a; padding: 8px 10px; margin-top: 4px;
           background: #f0f0f3; border-radius: 6px; }
.colhead span { font-weight: 400; color: #666; }
.reader { grid-column: 1 / -1; display: flex; align-items: center; gap: 10px;
          margin: 14px 0 6px; padding: 4px 0 4px 10px; border-left: 4px solid; }
.reader.z { border-color: #b64a44; } .reader.h { border-color: #3d6fb0; }
.reader .name { font-size: 14px; font-weight: 700; }
.reader .name i { font-family: Georgia, "Times New Roman", serif; }
.cell { min-width: 0; margin-bottom: 12px; }
.tldr { font-style: italic; font-size: 14px; line-height: 1.45; color: #1a1a1a; margin: 0 0 4px; }
.tldr .k { font-style: normal; font-size: 10.5px; font-weight: 700; letter-spacing: .4px;
           text-transform: uppercase; color: #888; margin-right: 6px; }
.gen { font-size: 11px; line-height: 1.5; color: #444; background: #fafafb; margin: 0; }
</style>"""


def cell(src, step, k):
    g = gens[(src, step, k)]
    t = TLDR[hashlib.sha1(g.encode()).hexdigest()]
    return (f'<div class="cell"><div class="tldr"><span class="k">sample {k}</span><span class="pill digest">digest</span>{marked(t)}</div>'
            f'<div class="gen">{gen_html(g)}</div></div>')


goldline = (f'gold intermediates {esc(", ".join(c for c in map(canon, m["intermediates"]) if c not in ANS))} · '
            f'gold answer <b>{esc(canon(m["answer"]))}</b>')
legend = "".join(f'<mark class="{k}">{esc(v)}</mark>' for k, v in LEGEND.items())
heads = "".join(f'<div class="colhead">step {s} <span>· {lab}</span></div>' for s, lab in STEPS.items())
body = [heads]
for src, (name, vec, cls) in READERS.items():
    body.append(f'<div class="reader {cls}"><span class="name">{name} · {vec}</span></div>')
    body += [cell(src, s, k) for k in KS for s in STEPS]
parts = [CSS, '<span class="pill method nla">NLA verbalizer</span>',
         f'<div class="top"><div><h2>problem {esc(UID)}</h2><div class="sub"><b class="q">{esc(m["question"])}</b><br>{goldline}</div></div>'
         f'<div class="legend">{legend}</div></div>',
         '<div class="grid">' + "".join(body) + "</div>"]
render_card("\n".join(parts), os.path.join(FIG_DIR, "qual_nla_duo.png"), width=1200)
