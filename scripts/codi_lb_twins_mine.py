"""Mine value-edit twin candidates for the TRACE-NECESSARY GSM8k test rows (CODI
latent decode correct, trace-free decode wrong, from controls_instances.json), so
the twin cross-patch can run on pairs where the thoughts are load-bearing.

Mechanics: parse the solution's <<expr=result>> calculator chain, pick a question
given (a number literal appearing exactly once in the question), substitute an
edited value g' through the chain (operand tokens classified by value as given /
earlier result / constant, ambiguities skip), and keep edits whose repropagated
chain stays valid (finite, integer-ness preserved, positive where the original was
positive, final answer clean and != gold). The chain must reach the #### answer
(prose-only final steps skip the problem). Comparison/max problems can silently
break under edits, so candidates go through an LLM solve verification next
(codi_lb_twins_verify.py) before any CODI gating.

-> reports/codi_writeup/lb_twins_cands.jsonl
CPU-only, seconds.
"""
from __future__ import annotations

import json
import math
import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_REPO, "reports", "codi_writeup", "lb_twins_cands.jsonl")
MAX_PER_PROBLEM = 4


def close(a, b, tol=1e-6):
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (ValueError, OverflowError):
        return False


def is_int(x):
    return abs(x - round(x)) < 1e-9


def fmt(x):
    return str(int(round(x))) if is_int(x) else f"{x:g}"


def q_numbers(question):
    """number literals in the question -> list of (literal, value)."""
    out = []
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", question):
        out.append((m.group(0), float(m.group(0).replace(",", ""))))
    return out


def eval_expr(expr):
    expr = expr.replace(",", "").replace("$", "").strip()
    if not re.fullmatch(r"[\d.+\-*/() ]+", expr):
        raise ValueError(expr)
    return eval(expr, {"__builtins__": {}})


def reprop(chain, g, g_new):
    """chain = [(expr, result)]: substitute g -> g_new, earlier results -> their new
    values, evaluate forward. Returns new results or None on ambiguity/invalid."""
    res_map = {}                       # original result value -> new value
    new_results = []
    for expr, res in chain:
        toks = list(re.finditer(r"\d[\d,]*(?:\.\d+)?", expr))
        new_expr = expr.replace(",", "").replace("$", "")
        repl = []
        for m in toks:
            v = float(m.group(0).replace(",", ""))
            hits = [close(v, g)] + [close(v, k) for k in res_map]
            if sum(hits) > 1:
                return None            # value matches both the given and a result
            if close(v, g):
                repl.append((m.group(0), g_new))
            else:
                for k, nv in res_map.items():
                    if close(v, k):
                        repl.append((m.group(0), nv))
                        break
        for lit, nv in repl:
            new_expr = re.sub(r"(?<![\d.])" + re.escape(lit.replace(",", "")) + r"(?![\d.])",
                              fmt(nv), new_expr, count=1)
        try:
            nres = eval_expr(new_expr)
        except Exception:
            return None
        if not math.isfinite(nres) or abs(nres) > 1e7:
            return None
        if is_int(res) and not is_int(nres):
            return None
        if res > 0 and nres <= 0:
            return None
        if close(res, g):
            return None                # a result collides with the given: ambiguous
        res_map[res] = nres
        new_results.append(nres)
    return new_results


def main():
    from datasets import load_dataset
    inst = json.load(open(os.path.join(_REPO, "reports", "codi_writeup",
                                       "controls_instances.json")))
    need = [r["i"] for r in inst
            if close(r["pred"]["identity"], r["gold"])
            and not close(r["pred"]["trunc_0"], r["gold"])]
    ds = load_dataset("openai/gsm8k", "main")["test"]
    print(f"trace-necessary identity-correct rows: {len(need)}")

    cands, skipped = [], {"no_chain": 0, "chain_short": 0, "no_valid_edit": 0}
    for i in need:
        q, sol = ds[i]["question"], ds[i]["answer"]
        gold = float(sol.split("####")[-1].replace(",", "").strip())
        chain = [(m.group(1), eval_expr(m.group(2)))
                 for m in re.finditer(r"<<([^=>]+)=([^>]+)>>", sol)
                 if re.fullmatch(r"[\d.,$ ]+", m.group(2))]
        if not chain or not close(chain[-1][1], gold):
            skipped["no_chain"] += 1
            continue
        if len(chain) < 2:
            skipped["chain_short"] += 1
            continue
        qn = q_numbers(q)
        found = 0
        for lit, g in qn:
            if found >= MAX_PER_PROBLEM:
                break
            if sum(close(g, v) for _, v in qn) != 1:
                continue               # given value must be unique in the question
            if g <= 2 or close(g, 100) or not is_int(g):
                continue               # structural constants and decimals excluded
            g_int = int(round(g))
            cand_vals = [2 * g_int, g_int + 10, g_int // 2, 3 * g_int, g_int + 5,
                         g_int - 5]
            for gn in dict.fromkeys(cand_vals):
                if gn <= 0 or gn == g_int:
                    continue
                new_res = reprop(chain, g, float(gn))
                if new_res is None or close(new_res[-1], gold):
                    continue
                if not (is_int(new_res[-1]) or round(new_res[-1], 2) == new_res[-1]):
                    continue
                tq = re.sub(r"(?<![\d.,])" + re.escape(lit) + r"(?![\d.,])",
                            str(gn), q, count=1)
                if tq == q or len(re.findall(r"(?<![\d.,])" + re.escape(lit)
                                             + r"(?![\d.,])", q)) != 1:
                    continue
                cands.append(dict(doc=f"t{i}", i=i, question=q, twin_question=tq,
                                  gold=fmt(gold), cf_twin=fmt(new_res[-1]),
                                  g=fmt(g), g_twin=str(gn),
                                  chain=[fmt(r) for _, r in chain],
                                  chain_twin=[fmt(r) for r in new_res]))
                found += 1
                if found >= MAX_PER_PROBLEM:
                    break
        if not found:
            skipped["no_valid_edit"] += 1

    docs = {c["doc"] for c in cands}
    print(f"candidates: {len(cands)} across {len(docs)} problems; skipped {skipped}")
    with open(OUT, "w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
