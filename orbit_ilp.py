"""Orbit-style dense ILP for bootstrapping placement, solved by SciPy/HiGHS.

Following the Orbit paper, this keeps explicit integer level variables on
vertices and edges, binary bootstrap decisions, and big-M constraints that
activate the refresh condition.  Modswitch/upscale are free and inserted after
solving to satisfy the level constraints.  SciPy/HiGHS replaces Gurobi.  Every
solution is re-scored with dag_builder.evaluate_dag().

Run:
    python orbit_ilp.py                 # six-case attention experiment
"""

import argparse
import numpy as np
from scipy.optimize import Bounds, milp

import dag_builder as db
from scipy_ilp import (T1, T2, affine, assemble, bootstrap_counts,
                       predecessors, score, topology)


def solve_orbit(nodes, edges, l_init, l_max=14, time_limit=600):
    """Solve the Orbit-style ILP; return (summary, placed nodes, placed edges)."""
    names, n_ct = topology(nodes, edges)
    op = {n: o for n, o, *_ in nodes}
    d = {n: dd for n, _, dd, *_ in nodes}
    indeg = predecessors(nodes, edges)
    bigM = l_max + 1

    li, lo, le, pre, b1, b2, y1, y2, sel = {}, {}, {}, {}, {}, {}, {}, {}, {}
    c, lb, ub, integrality = [], [], [], []

    def new_var(cost=0.0, high=1.0):
        idx = len(c)
        c.append(cost); lb.append(0.0); ub.append(high); integrality.append(1)
        return idx

    for n in names:
        li[n] = new_var(high=l_max)
        lo[n] = new_var(high=l_max)
        y1[n] = new_var(cost=n_ct[n] * T1)
        y2[n] = new_var(cost=(n_ct[n] // 2) * T2
                        if n_ct[n] in (8, 16) else 0.0,
                        high=(1.0 if n_ct[n] in (8, 16) else 0.0))
    for i, (u, v, ct) in enumerate(edges):
        le[i] = new_var(high=l_max)
        pre[i] = new_var(high=l_max)
        b1[i] = new_var(high=(1.0 if n_ct[u] else 0.0))
        b2[i] = new_var(high=(1.0 if n_ct[u] in (8, 16) else 0.0))
    for v, preds in indeg.items():
        if len(preds) > 1:
            for i in preds:
                sel[v, i] = new_var()

    const = 0.0
    for n in names:
        a, b = affine(db.LATENCY[op[n]])
        c[li[n]] += a
        const += b

    rows, lows, highs = [], [], []

    def add(coeff, low=-np.inf, high=np.inf):
        rows.append(coeff); lows.append(low); highs.append(high)

    # Vertex levels and the fan-in minimum.
    for n in names:
        add({lo[n]: 1, li[n]: -1}, -d[n], -d[n])
        if not indeg[n]:
            add({li[n]: 1}, l_init, l_init)
        else:
            for i in indeg[n]:
                add({li[n]: 1, le[i]: -1}, high=0)
            if len(indeg[n]) > 1:
                add({sel[n, i]: 1 for i in indeg[n]}, 1, 1)
                for i in indeg[n]:
                    add({li[n]: 1, le[i]: -1, sel[n, i]: -bigM}, low=-bigM)
            else:
                add({li[n]: 1, le[indeg[n][0]]: -1}, 0, 0)

    # Big-M activation of B1/B2 on each outgoing edge (Orbit's conditional
    # constraints), followed by a free modswitch down to the edge level.
    for i, (u, v, ct) in enumerate(edges):
        add({b1[i]: 1, b2[i]: 1}, high=1)
        add({pre[i]: 1, lo[u]: -1, b1[i]: -bigM, b2[i]: -bigM}, high=0)
        add({pre[i]: 1, lo[u]: -1, b1[i]: bigM, b2[i]: bigM}, low=0)
        add({pre[i]: 1, b1[i]: bigM}, high=l_max + bigM)
        add({pre[i]: 1, b1[i]: -bigM}, low=l_max - bigM)
        add({pre[i]: 1, b2[i]: bigM}, high=l_max - 1 + bigM)
        add({pre[i]: 1, b2[i]: -bigM}, low=l_max - 1 - bigM)
        add({le[i]: 1, pre[i]: -1}, high=0)

    # Cost is paid once per distinct bootstrap prefix leaving a producer.
    for u in names:
        out = [i for i, (a, b, c) in enumerate(edges) if a == u]
        for flag, yy in ((b1, y1[u]), (b2, y2[u])):
            for i in out:
                add({yy: 1, flag[i]: -1}, low=0)
            add({yy: 1} | {flag[i]: -1 for i in out}, high=0)

    res = milp(c=np.array(c, dtype=float), integrality=np.array(integrality),
               bounds=Bounds(np.array(lb), np.array(ub)),
               constraints=assemble(rows, lows, highs, len(c)),
               options={"mip_rel_gap": 0.0, "time_limit": time_limit})
    if not res.success:
        raise RuntimeError(f"Orbit ILP failed: {res.message}")

    # Lower the ILP solution to the six chains and score it with the handout.
    lo_sol = {n: int(round(res.x[lo[n]])) for n in names}
    choices = []
    for i, (u, v, ct) in enumerate(edges):
        target = int(round(res.x[le[i]]))
        if round(res.x[b1[i]]):
            choices.append("B1" if target == l_max else f"B1M{target}")
        elif round(res.x[b2[i]]):
            choices.append("B2" if target == l_max - 1 else f"B2M{target}")
        else:
            choices.append("none" if target == lo_sol[u] else f"M{target}")

    result, placed_nodes, placed_edges = score(
        nodes, edges, choices, lo_sol, l_init, l_max)
    model_cost = float(res.fun) + const
    if abs(model_cost - result["latency"]) > 1e-4:
        raise AssertionError((model_cost, result["latency"]))
    b1ct, b2ct = bootstrap_counts(nodes, edges, choices)
    return {"latency": result["latency"], "model_cost": model_cost,
            "b1": b1ct, "b2": b2ct, "nodes": res.mip_node_count,
            "gap": res.mip_gap, "choices": choices}, placed_nodes, placed_edges


def cases():
    return [(li, lm) for li in (1, 2) for lm in (14, 15, 16)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l-init", type=int, default=None)
    ap.add_argument("--l-max", type=int, default=14)
    args = ap.parse_args()
    matrix = cases() if args.l_init is None else [(args.l_init, args.l_max)]
    for li, lm in matrix:
        nodes, edges = db.get_attn_empty()
        res, _, _ = solve_orbit(nodes, edges, li, lm)
        print(f"L_init={li} L_max={lm}: latency={res['latency']:.4f} "
              f"B1ct={res['b1']} B2ct={res['b2']} nodes={res['nodes']} "
              f"gap={res['gap']:.2e}")


if __name__ == "__main__":
    main()
