"""One concrete NLA round-trip edit, before and after (post: Intervening via the NLA).

READER PROVENANCE: this example uses the nz2/nz3-era z reader, whose explanations
still surfaced true chain intermediates. The intervention does NOT transfer to the
final nz4-3200 z reader: its cipher explanations surface a true intermediate only
at the permuted-gold coincidence rate (5/46 case-eligible at step 3200 vs ~0.10
coincidence floor), so there is no semantically load-bearing value to edit. Kept
on the original artifacts pending an editorial decision (candidate replacement:
an h24-side edit via the h24 critic, a different experiment).

Rebuilds the splice campaign's case list exactly as zlb_arsplice.py did (dedup by
explanation body, exactly one chain intermediate in the body, edit value v -> v' with
v' = 2v or v+7 or v+13 avoiding all problem numbers, cap 6 per (doc, slot)) and
verifies every case against zlb_splice_summary.json before rendering CASE. The card
shows the problem and the verbalization of the fed vector ONCE, with the edit
annotated inline at the read value (green original, red replacement). The splice
outcomes stay in the artifact and the post caption. Rendered as a styled HTML card
via html_render (headless chromium).

Inputs (env ZLB_DIR, default artifacts/nz2): zlb_gens.jsonl,
val_docs.jsonl, zlb_splice_summary.json.
Output: figs/nla_edit_example.png (env FIG_OUT).
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from html_render import esc, mark_words, render_card

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("ZLB_DIR", os.path.join(HERE, "..", "artifacts", "nz2"))
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs", "nla_edit_example.png"))
CASE = int(os.environ.get("EDIT_CASE", "6"))
CASE_CAP = 6
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def canon(s):
    f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"


def try_canon(s):
    try:
        return canon(s)
    except ValueError:
        return None


def nums_in(t):
    return {c for c in (try_canon(m) for m in NUM.findall(t)) if c is not None}


def rebuild_cases(docs):
    seen, per_rs, cases = set(), {}, []
    for g in (json.loads(l) for l in open(f"{D}/zlb_gens.jsonl")):
        d = docs[g["doc"]]
        m = re.search(r"<explanation>(.*?)</explanation>", g["text"], re.S)
        if not m:
            continue
        body = m.group(1)
        key = (g["doc"], g["stratum"], body.strip())
        if key in seen:
            continue
        qnums = nums_in(d["question"])
        ans = canon(d["answer"])
        chain = {canon(v) for v in d["intermediates"]} - qnums - {ans}
        hits = nums_in(body) & chain
        if len(hits) != 1:
            continue
        rs = (g["doc"], g["stratum"])
        if per_rs.get(rs, 0) >= CASE_CAP:
            continue
        v = next(iter(hits))
        taboo = qnums | {canon(x) for x in d["intermediates"]} | {ans, v}
        vs = next((c for c in (canon(2 * float(v)), canon(float(v) + 7),
                               canon(float(v) + 13)) if c not in taboo), None)
        if vs is None:
            continue
        swapped = NUM.sub(lambda mm: vs if try_canon(mm.group()) == v else mm.group(), body)
        if swapped == body:
            continue
        seen.add(key)
        per_rs[rs] = per_rs.get(rs, 0) + 1
        cases.append(dict(doc=g["doc"], stratum=g["stratum"], v=v, v_swap=vs,
                          body=body, swapped=swapped))
    return cases


docs = {d["doc_id"]: d for d in (json.loads(l) for l in open(f"{D}/val_docs.jsonl"))}
cases = rebuild_cases(docs)
S = json.load(open(f"{D}/zlb_splice_summary.json"))
assert len(cases) == len(S)
for c, s in zip(cases, S):
    assert (c["doc"], c["stratum"], c["v"], c["v_swap"]) == \
           (s["doc"], s["stratum"], s["v"], s["v_swap"]), (c["doc"], s["doc"])
c, s = cases[CASE], S[CASE]
d = docs[c["doc"]]
slot = c["stratum"].split("_p")[1]
print(f"[case {CASE}] {c['doc']} slot {slot}: v {c['v']} -> {c['v_swap']}, "
      f"gold {s['gold']}, cf {s['cf']}, recon_pred {s['recon_ctrl_pred']}, "
      f"swap_pred {s['swap_pred']}")

def gen_html_edit(text, v, v_swap):
    """Verbalization text -> HTML, the edit annotated inline: every word canon-equal
    to v is marked green and immediately followed by an arrow to the red v_swap."""
    paras = []
    for para in text.strip().split("\n\n"):
        words = " ".join(para.split()).split()
        out = []
        for w in words:
            if try_canon(w) == v:
                out.append(f'<mark class="gold">{esc(w)}</mark> → '
                           f'<mark class="fab">{esc(v_swap)}</mark>')
            else:
                out.append(esc(w))
        paras.append(" ".join(out))
    return "\n\n".join(paras)


body = f"""
<span class="pill method nla">NLA verbalizer · round-trip edit</span>
<h2>problem {esc(c['doc'])}</h2>
<div class="sub"><b class="q">{esc(d['question'])}</b><br>
gold answer <b>{esc(str(s['gold']))}</b> · the edit implies the counterfactual answer <b>{esc(str(s['cf']))}</b></div>

<span class="pill z"><i>z</i> reader · slot {slot}</span><span class="pill step">text edit {esc(c['v'])} → {esc(c['v_swap'])}</span>
<div class="gen">{gen_html_edit(c['body'], c['v'], c['v_swap'])}</div>

<div class="legend"><mark class="gold">the value the reader wrote (a true intermediate)</mark>
<mark class="fab">the edited value</mark></div>
"""
render_card(body, OUT)
