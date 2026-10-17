"""Single AO example card (post: Appendix D), laid out like the NLA qualitative card
(fig_05): ONE problem (the load-bearing exemplar of the qual scan, gsmeval idx 126).
Columns = training steps, left to right (step 0 = untrained init, step 66000 = final);
sections = the two joint thought reads (full six-stream read, z-only read); rows = the
two label prompts. Each cell shows two samples at T = 1 (scripts/fd1_qual_resample.py,
on a batch-1 recapture of the problem's activations) under one italic digest of the
pair (tagged by a digest pill) (editorial, artifacts/qual_ao_tldr.json, keyed by the sha1 of the two verbatim
readouts so a changed readout fails loudly). Readouts keep their line breaks, markdown
**bold** is rendered. Numbers highlighted by canonical value (green = gold answer, light
green = gold intermediate, red = other); stream tags and list enumerators stay unmarked.

Inputs: artifacts/fd1_qual_resample.jsonl, artifacts/qual_ao_tldr.json,
        artifacts/nz2/lb41_capture.jsonl (env CAP_DIR).
Output: figs/ao_example.png (env FIG_DIR).
"""
import hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from html_render import esc, mark_words, render_card

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
CAP = os.environ.get("CAP_DIR", os.path.join(ART, "nz2"))
FIG_DIR = os.environ.get("FIG_DIR", os.path.join(HERE, "..", "figs"))
LEGEND = {"ans": "gold answer", "gold": "gold intermediate", "fab": "other number"}
MV = '<span class="mv">{}</span>'.format
H, Z = MV("h"), MV("z")
READS = {"ALL_joint": ("full read", "{" + H + "<sub>t</sub><sup>(ℓ)</sup>, " + Z + "<sub>t</sub>}<sub>ℓ</sub>, t = 1…6", "h"),
         "Z_joint": ("z-only read", "{" + Z + "<sub>t</sub>}, t = 1…6", "z")}
STEPS = {0: "untrained init", 66000: "final"}
QS = {"ans": "What is the final answer of the computation?",
      "vals": "List every intermediate result computed (any order)."}

rows = [json.loads(l) for l in open(os.path.join(ART, "fd1_qual_resample.jsonl"))]
idx = rows[0]["idx"]
gens = {(r["step"], r["read"], r["q"], r["k"]): r["text"].strip() for r in rows if r["decode"] == "sample"}
KS = sorted({k for (*_, k) in gens})
assert len(gens) == len(STEPS) * len(READS) * len(QS) * len(KS), sorted(gens)
m = {r["idx"]: r for r in map(json.loads, open(f"{CAP}/lb41_capture.jsonl"))}[idx]
TLDR = {t["sha1"]: t["tldr"] for t in json.load(open(os.path.join(ART, "qual_ao_tldr.json")))}


def pair_sha(step, read, q):
    return hashlib.sha1("\n\n".join(gens[(step, read, q, k)] for k in KS).encode()).hexdigest()


def canon(s):
    f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"


anss = {canon(m["answer"])}
ints = {canon(m["vr"])} - anss


def word_cats(text):
    words = text.split()
    norm = " ".join(words)
    starts, pos = [], 0
    for w in words:
        starts.append(pos); pos += len(w) + 1
    cats = [None] * len(words)
    for mt in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", norm):
        try:
            c = canon(mt.group())
        except (ValueError, OverflowError):
            continue
        cat = "ans" if c in anss else "gold" if c in ints else "fab"
        for i, st in enumerate(starts):
            if st < mt.end() and st + len(words[i]) > mt.start():
                cats[i] = cats[i] or cat
    return cats


TAG = re.compile(r"\[(?:Q )?(?:L\d+|LF|Z)\]")


def line_html(line):
    """one line -> marked HTML; stream tags ([L9], [Z]) and line-leading list
    enumerators ("1.") are labels, not stated values, so they stay unmarked."""
    words = line.split()
    cats = word_cats(" ".join(words))
    for i, w in enumerate(words):
        if TAG.search(w) or (i == 0 and re.fullmatch(r"\d+\.", w)):
            cats[i] = None
    return mark_words(" ".join(words), cats)


def gen_html(text):
    """verbatim readout with its line structure kept and markdown **bold** rendered"""
    html = "\n".join(line_html(l) if l.split() else "" for l in text.split("\n"))
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", re.sub(r"\n{3,}", "\n\n", html))


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
          margin: 16px 0 2px; padding: 4px 0 4px 10px; border-left: 4px solid; }
.reader.z { border-color: #b64a44; } .reader.h { border-color: #3d6fb0; }
.reader .name { font-size: 14px; font-weight: 700; }
.reader .feed { font-size: 13px; color: #555; }
.qrow { grid-column: 1 / -1; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
        font-size: 11.5px; color: #777; margin: 8px 0 4px; }
.cell { min-width: 0; margin-bottom: 4px; }
.tldr { font-style: italic; font-size: 14px; line-height: 1.45; color: #1a1a1a; margin: 0 0 4px; }
.samp { display: grid; grid-template-columns: 22px 1fr; align-items: start; }
.samp .k { font-size: 10.5px; font-weight: 700; color: #999; padding-top: 8px; }
.gen { font-size: 11px; line-height: 1.5; color: #444; background: #fafafb; margin: 0 0 4px; padding: 5px 10px; }
</style>"""


def cell(step, read, q):
    samples = "".join(f'<div class="samp"><span class="k">{k}</span><div class="gen">{gen_html(gens[(step, read, q, k)])}</div></div>' for k in KS)
    return f'<div class="cell"><div class="tldr"><span class="pill digest">digest</span>{line_html(TLDR[pair_sha(step, read, q)])}</div>{samples}</div>'


goldline = f"gold answer {canon(m['answer'])} · answer-feeding intermediate {canon(m['vr'])}"
legend = "".join(f'<mark class="{k}">{esc(v)}</mark>' for k, v in LEGEND.items())
body = ["".join(f'<div class="colhead">step {s} <span>· {lab}</span></div>' for s, lab in STEPS.items())]
for read, (name, feed, cls) in READS.items():
    body.append(f'<div class="reader {cls}"><span class="name">{name}</span><span class="feed">{feed}</span></div>')
    for q, qtext in QS.items():
        body.append(f'<div class="qrow">Q: "{esc(qtext)}"</div>')
        body += [cell(s, read, q) for s in STEPS]
parts = [CSS, '<span class="pill method ao">AO verbalizer</span>',
         f'<div class="top"><div><h2>problem gsm8k test #{idx}</h2><div class="sub"><b class="q">{esc(m["question"])}</b><br>{line_html(goldline)}</div></div>'
         f'<div class="legend">{legend}</div></div>',
         '<div class="grid">' + "".join(body) + "</div>"]
render_card("\n".join(parts), os.path.join(FIG_DIR, "ao_example.png"), width=1200)
