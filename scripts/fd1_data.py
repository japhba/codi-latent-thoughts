"""fd1 sample construction (pod): random position subsets x random source subsets x
random chain subsets. NO position tags (user: start-index is a confound); markers appear
in temporal order within a source group; sources appear in canonical order with tags.
Multi-value targets are emitted in RANDOM order (no sortedness bias), numeric and
sentence labels alike.

Streams: h positions [qend, bocot, t1..t6] x [L9,L18,L27,L35,Lfin]; z [t1..t6].
Sources: L9 L18 L27 L35 LF Z — Z weighted like one layer (uniform over the 6).
"""
import json, os, re
import numpy as np

DATA = os.environ.get("FD1_DATA", "/root/fd1/data")
MARKER = "㈎"
SRC = ["L9", "L18", "L27", "L35", "LF", "Z"]
HIDX = {"L9": 0, "L18": 1, "L27": 2, "L35": 3, "LF": 4}   # stream axis of *_h.npy
SIZE_P = [(1, 0.4), (2, 0.3), (3, 0.2), (6, 0.1)]          # source-subset size dist

OPWORD = {"add": "plus", "sub": "minus", "mul": "times", "div": "divided by"}
OPNAME = {"add": "addition", "sub": "subtraction", "mul": "multiplication", "div": "division"}

def canon(s):
    f = float(str(s).replace(",", "").replace("$", "").rstrip("%."))
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"

def _sentences(cot):
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n", cot) if s.strip()]

def step_sentence(step, rng):
    a, b, r, op = step["a"], step["b"], step["res"], step["op"]
    forms = [f"{a} {OPWORD[op]} {b} equals {r}.",
             f"We compute {a} {OPWORD[op]} {b} and get {r}.",
             f"Taking {a} {OPWORD[op]} {b} gives {r}."]
    return rng.choice(forms)

class Pool:
    """One capture split: meta rows + z [N,6,4096] + h [N,8,5,4096]."""
    def __init__(self, name):
        self.meta = [json.loads(l) for l in open(f"{DATA}/{name}_meta.jsonl")]
        self.z = np.load(f"{DATA}/{name}_z.npy", mmap_mode="r")
        self.h = np.load(f"{DATA}/{name}_h.npy", mmap_mode="r")
        self.qh = self.qlen = None
        if os.path.exists(f"{DATA}/{name}_qh.npy"):
            self.qh = np.load(f"{DATA}/{name}_qh.npy", mmap_mode="r")
            self.qlen = np.load(f"{DATA}/{name}_qlen.npy")
        for m in self.meta:
            if m["intermediates"] and m["intermediates"][-1] == m["answer"]:
                m["intermediates"] = m["intermediates"][:-1]
            m["chain"] = self._chain(m)
            m["givens"] = self._givens(m)
        self.ok = [i for i, m in enumerate(self.meta) if m["correct"] and len(m["chain"]) >= 1]

    def _givens(self, m):
        if m["src"] == "syn":
            out, res = [], set()
            for s in m["steps"]:
                for o in (s["a"], s["b"]):
                    c = canon(o)
                    if c not in res and c not in out: out.append(c)
                res.add(canon(s["res"]))
            return out
        out = []
        for mt in re.findall(r"-?\d[\d,]*(?:\.\d+)?", m["question"]):
            try:
                c = canon(mt)
                if c not in out: out.append(c)
            except ValueError: pass
        return out

    def _chain(self, m):
        """ordered chain steps: [{val, sentence(sig or None), step(dict or None)}]"""
        out = []
        if m["src"] == "syn":
            for s in m["steps"]:
                v = canon(s["res"])
                if v == m["answer"] and s is m["steps"][-1]: continue
                out.append(dict(val=v, step=s, sent=None))
        else:
            cot = m.get("cot", "")
            sents = _sentences(cot)
            OP = {"+": "add", "-": "sub", "*": "mul", "x": "mul", "×": "mul", "/": "div", "÷": "div"}
            eqs = {}
            for mt in re.finditer(r"(-?[\d,.]+)\s*([-+*/x×÷])\s*(-?[\d,.]+)\s*=\s*\$?(-?[\d,.]+)", cot):
                try: eqs.setdefault(canon(mt.group(4)), dict(a=mt.group(1), b=mt.group(3), op=OP[mt.group(2)], res=mt.group(4)))
                except ValueError: pass
            for v in m["intermediates"]:
                hit = next((s for s in sents if re.search(rf"=\s*\$?{re.escape(v)}(?![\d.])", s)), None)
                out.append(dict(val=v, step=eqs.get(v), sent=hit))
        return out

    def vec(self, i, src, pos):
        """pos: 'q' -> ALL question-token positions (last <=96, temporal order) when the
        full-question capture exists, else the 2 summary slots [qend,bocot];
        else list of thought idx 0..5 (temporal order). One block read per call
        (per-position mmap reads are NFS-latency-bound)."""
        if src == "Z":
            assert pos != "q"
            zb = np.array(self.z[i], dtype=np.float32)
            return [zb[t] for t in pos]
        s = HIDX[src]
        if pos == "q":
            if self.qh is not None:
                L = int(self.qlen[i])
                return list(np.array(self.qh[i, :L, s], dtype=np.float32))
            hb = np.array(self.h[i, :2, s], dtype=np.float32)
            return [hb[0], hb[1]]
        hb = np.array(self.h[i, :, s], dtype=np.float32)
        return [hb[2 + t] for t in pos]

def sample_spec(rng, srcs=None):
    """(sources, pos_mode, thought positions) — Z uniform with layers; no position tags."""
    srcs = srcs or SRC
    r = rng.random(); acc = 0
    for k, p in SIZE_P:
        acc += p
        if r < acc: break
    k = min(k, len(srcs))
    sources = sorted(rng.sample(srcs, k), key=SRC.index)
    pm = rng.random()
    pos_mode = "thoughts" if pm < 0.8 else ("q_thoughts" if pm < 0.9 else "q_only")
    if pos_mode == "q_only" and sources == ["Z"]:
        pos_mode = "thoughts"                      # Z has no q-region
    m = rng.randint(1, 6)
    tpos = sorted(rng.sample(range(6), m))
    return sources, pos_mode, tpos

def render_acts(pool, i, sources, pos_mode, tpos):
    """-> (act_block_text, vecs list aligned with markers)"""
    groups, vecs = [], []
    for src in sources:
        if pos_mode in ("q_thoughts", "q_only") and src != "Z":
            v = pool.vec(i, src, "q")
            groups.append(f"[Q {src}] " + MARKER * len(v)); vecs += v
        if pos_mode in ("thoughts", "q_thoughts"):
            v = pool.vec(i, src, tpos)
            groups.append(f"[{src}] " + MARKER * len(v)); vecs += v
    return "\n".join(groups), vecs

def make_qa(m, rng, kind=None):
    """random chain subset -> (question, target). Multi-item targets in random order."""
    chain = m["chain"]
    kind = kind or rng.choice(["stepval", "allvals", "stepsent", "allsents", "answer", "ops", "count", "givens"])
    if kind == "givens":
        g = list(m["givens"]) or ["unknown"]
        rng.shuffle(g)
        return "List the numbers given in the problem statement (any order).", ", ".join(g), kind
    if kind in ("stepsent", "allsents") and m["src"] == "syn" and not chain: kind = "answer"
    def render_step(c):
        return c["sent"] or (step_sentence(c["step"], rng) if c["step"] else f"The result is {c['val']}.")
    if kind == "stepval" and chain:
        ks = rng.sample(range(len(chain)), rng.randint(1, min(3, len(chain))))
        rng.shuffle(ks)
        q = "What value does computation step %s evaluate to?" % " and step ".join(str(k+1) for k in ks)
        tgt = "; ".join(f"step {k+1}: {chain[k]['val']}" for k in ks)
    elif kind == "allvals" and chain:
        vals = [c["val"] for c in chain]; rng.shuffle(vals)
        q = "List every intermediate result computed (any order)."
        tgt = ", ".join(vals)
    elif kind == "stepsent" and chain:
        k = rng.randrange(len(chain))
        q = f"Describe computation step {k+1} in one sentence."
        tgt = render_step(chain[k])
    elif kind == "allsents" and chain:
        parts = [render_step(c) for c in chain]
        rng.shuffle(parts)
        q = "Describe the computation steps as sentences (any order)."
        tgt = " ".join(parts)
    elif kind == "ops":
        ops = ({OPNAME[s["op"]] for s in m["steps"]} if m["src"] == "syn"
               else {OPNAME[o] for o in _gsm_ops(m)})
        ops = sorted(ops); rng.shuffle(ops)   # sorted first: set order is PYTHONHASHSEED-dependent
        q = "Which arithmetic operations occur in the computation?"
        tgt = ", ".join(ops) if ops else "unknown"
    elif kind == "count":
        n = len(chain) + 1
        q = "How many computation steps are there in total?"
        tgt = str(n)
    else:
        q = "What is the final answer of the computation?"
        tgt = m["answer"]
    return q, tgt, kind

def _gsm_ops(m):
    ops = set()
    for mt in re.finditer(r"(\d[\d,.]*)\s*([+\-*/x×÷])\s*(\d[\d,.]*)", m.get("cot", "")):
        ops.add({"+": "add", "-": "sub", "*": "mul", "x": "mul", "×": "mul", "/": "div", "÷": "div"}[mt.group(2)])
    return ops

def text_block(m, rng):
    """text-mode content: the chain rendered as text (teacher cot for gsm, cot_target for syn)"""
    return m["cot_target"] if m["src"] == "syn" else m.get("cot", "")

PREAMBLE = ("You are analyzing internal activation vectors captured from a language model "
            "while it solved a problem with latent (non-verbal) reasoning. Each group below "
            "is tagged with its source stream ([L9]..[L35] residual layers, [LF] final layer, "
            "[Z] fed-back thought vectors; [Q ...] = question-region positions). "
            "Answer the question from the activation contents.\n\n")
PREAMBLE_TXT = ("You are analyzing the reasoning of a language model solving a problem. "
                "The reasoning transcript is given below. Answer the question.\n\n")

def build_sample(pool, rng, mode=None, i=None, srcs=None, kind=None, spec=None):
    """mode in {act, text, text_act}; spec=(sources,pos_mode,tpos) override; kind = label type"""
    i = rng.choice(pool.ok) if i is None else i
    m = pool.meta[i]
    mode = mode or rng.choices(["act", "text", "text_act"], weights=[0.6, 0.2, 0.2])[0]
    q, tgt, kind = make_qa(m, rng, kind=kind)
    if mode == "text":
        prompt = PREAMBLE_TXT + f"Transcript: {text_block(m, rng)}\n\nQuestion: {q}"
        return dict(prompt=prompt, target=tgt, vecs=[], kind=kind, cell=f"text|{kind}")
    sources, pos_mode, tpos = spec if spec else sample_spec(rng, srcs)
    if pool.qh is None: pos_mode = "thoughts"   # qh-less pools train the thought channel only
    acts, vecs = render_acts(pool, i, sources, pos_mode, tpos)
    body = acts
    if mode == "text_act":
        body = f"Transcript: {text_block(m, rng)}\n\n" + acts
    itag = ("qonly" if pos_mode == "q_only" else "qth" if pos_mode == "q_thoughts" else
            (sources[0] if len(sources) == 1 else ("ALL" if len(sources) == 6 else f"k{len(sources)}")))
    prompt = PREAMBLE + body + f"\n\nQuestion: {q}"
    return dict(prompt=prompt, target=tgt, vecs=vecs, kind=kind,
                cell=f"{itag}|{kind}" + ("&txt" if mode == "text_act" else ""))
