"""Compact chain-type ILP for bootstrapping placement, solved by SciPy/HiGHS.

Decision per edge: a prefix (none / B1 / B2) plus an integer target level.
Bootstrap prefixes are shared per producer through OR variables; a lower target
level means a free Modswitch, so the six chains of the handout are covered
without enumerating all 15 possible switch levels.  Levels are integers and the
fan-in minimum is linearised exactly with a one-hot selector.  Every solution is
re-scored with `dag_builder.evaluate_dag()` before being reported.

Run:
    python scipy_ilp.py                 # six-case attention experiment
    python scipy_ilp.py --compare       # Orbit / SciPy / greedy / baseline
"""

import argparse
import numpy as np
from scipy.optimize import Bounds, milp

import dag_builder as db
import solver as greedy_solver

M = 15                      # levels live in [0, 14], so 15 is a valid big-M
T1 = 1040.15                # Bootstrap1, per ciphertext
T2 = 1293.13                # Bootstrap2, per two ciphertexts


# --------------------------------------------------------------------------- #
#  Shared helpers (also imported by orbit_ilp.py)
# --------------------------------------------------------------------------- #
def predecessors(nodes, edges):
    """Map each node to the indices of its incoming edges."""
    preds = {n: [] for n, *_ in nodes}
    for i, (u, v, ct) in enumerate(edges):
        preds[v].append(i)
    return preds


def topology(nodes, edges):
    """Return node order and the #cipher carried by each producer."""
    names = [n for n, *_ in nodes]
    n_ct = {n: 0 for n in names}
    for u, v, ct in edges:
        if n_ct[u] not in (0, ct):
            raise ValueError(f"Producer {u} has inconsistent ciphertext counts")
        n_ct[u] = ct
    return names, n_ct


def affine(f):
    """Coefficients (a, b) of an affine latency function f(l) = a*l + b."""
    b = float(f(0))
    a = float(f(1)) - b
    if abs(float(f(2)) - (2 * a + b)) > 1e-6:
        raise ValueError("Latency function is not affine")
    return a, b


def option_target(option, l_max):
    """Output level fixed by a chain option; None means 'keep producer level'."""
    if option == "none":
        return None
    if option.startswith("B1"):
        return l_max if option == "B1" else int(option[3:])
    if option.startswith("B2"):
        return l_max - 1 if option == "B2" else int(option[3:])
    if option.startswith("M"):
        return int(option[1:])
    raise ValueError(option)


def build_inserted(nodes, edges, choices, l_out, l_max):
    """Splice the chosen chains into the empty DAG for reference scoring."""
    names, n_ct = topology(nodes, edges)
    result_nodes, result_edges = list(nodes), []
    boots, mods = {}, {}

    for u in names:
        if not n_ct[u]:
            continue
        if any("B1" in choices[i] for i, (a, b, c) in enumerate(edges) if a == u):
            name = f"{u}_B1"
            boots[(u, "B1")] = name
            result_nodes.append((name, "Bootstrap1", 0, f"B1 after {u}"))
            result_edges.append((u, name, n_ct[u]))
        if any("B2" in choices[i] for i, (a, b, c) in enumerate(edges) if a == u):
            name = f"{u}_B2"
            boots[(u, "B2")] = name
            result_nodes.append((name, "Bootstrap2", 0, f"B2 after {u}"))
            result_edges.append((u, name, n_ct[u]))

    for i, (u, v, ct) in enumerate(edges):
        opt = choices[i]
        if opt == "none":
            result_edges.append((u, v, ct))
            continue
        if "B1" in opt:
            src, pre = boots[(u, "B1")], l_max
        elif "B2" in opt:
            src, pre = boots[(u, "B2")], l_max - 1
        else:
            src, pre = u, int(round(l_out[u]))
        target = int(option_target(opt, l_max))
        if target != pre:
            key = (src, target)
            if key not in mods:
                name = f"{src}_M{target}"
                mods[key] = name
                result_nodes.append((name, "Modswitch", pre - target,
                                     f"M{target} after {src}"))
                result_edges.append((src, name, ct))
            src = mods[key]
        result_edges.append((src, v, ct))

    return result_nodes, result_edges


def score(nodes, edges, choices, l_out, l_init, l_max):
    """Reference score of a placement; raises if the placement is invalid."""
    placed_nodes, placed_edges = build_inserted(nodes, edges, choices, l_out, l_max)
    res = db.evaluate_dag(placed_nodes, placed_edges, l_init, l_max, db.LATENCY)
    if res is None:
        raise ValueError("Model produced an infeasible placement")
    return res, placed_nodes, placed_edges


def bootstrap_counts(nodes, edges, choices):
    """Count per-ciphertext Bootstrap1/Bootstrap2 operations in a solution.

    A B1 prefix on an edge is charged to its producer once; a B2 prefix is
    charged half the producer's ciphertext count.  ``choices[i]`` is the chain
    string for ``edges[i]`` returned by :func:`solve_scipy`.
    """
    names, n_ct = topology(nodes, edges)
    b1 = {u for u in names if any("B1" in choices[i]
                                  for i, (a, b, c) in enumerate(edges) if a == u)}
    b2 = {u for u in names if any("B2" in choices[i]
                                  for i, (a, b, c) in enumerate(edges) if a == u)}
    return sum(n_ct[u] for u in b1), sum(n_ct[u] // 2 for u in b2)


def assemble(rows, lows, highs, nvar):
    """Build the sparse LinearConstraint expected by scipy.optimize.milp."""
    from scipy.sparse import coo_array
    from scipy.optimize import LinearConstraint

    ri, ci, data = [], [], []
    for r, coeff in enumerate(rows):
        for j, val in coeff.items():
            if val:
                ri.append(r); ci.append(j); data.append(val)
    A = coo_array((data, (ri, ci)), shape=(len(rows), nvar))
    return LinearConstraint(A.tocsc(), np.array(lows), np.array(highs))


# --------------------------------------------------------------------------- #
#  Compact chain-type ILP
# --------------------------------------------------------------------------- #
def solve_scipy(nodes, edges, l_init, l_max=14, time_limit=600):
    """Solve the chain-type model; return (summary, placed nodes, placed edges)."""
    names, n_ct = topology(nodes, edges)
    op = {n: o for n, o, *_ in nodes}
    d = {n: dd for n, _, dd, *_ in nodes}
    indeg = predecessors(nodes, edges)
    bigM = l_max + 1

    b1, b2, target, lo, li, y1, y2, sel = {}, {}, {}, {}, {}, {}, {}, {}
    c, lb, ub, integrality = [], [], [], []

    def new_var(cost=0.0, high=1.0):
        """Append one integer variable and return its column index."""
        idx = len(c)
        c.append(cost); lb.append(0.0); ub.append(high); integrality.append(1)
        return idx

    for i, (u, v, ct) in enumerate(edges):
        b1[i] = new_var(high=(1.0 if n_ct[u] else 0.0))
        b2[i] = new_var(high=(1.0 if n_ct[u] in (8, 16) else 0.0))
        target[i] = new_var(high=l_max)
    for n in names:
        li[n] = new_var(high=l_max)
        lo[n] = new_var(high=l_max)
        y1[n] = new_var(cost=n_ct[n] * T1)
        y2[n] = new_var(cost=(n_ct[n] // 2) * T2
                        if n_ct[n] in (8, 16) else 0.0,
                        high=(1.0 if n_ct[n] in (8, 16) else 0.0))
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
        """Append one linear constraint given a {variable: coefficient} dict."""
        rows.append(coeff); lows.append(low); highs.append(high)

    # Chain prefix, target level and sharing of bootstrap prefixes.
    for i, (u, v, ct) in enumerate(edges):
        add({b1[i]: 1, b2[i]: 1}, high=1)
        add({target[i]: 1, lo[u]: -1, b1[i]: -bigM, b2[i]: -bigM}, high=0)
        add({target[i]: 1, b1[i]: bigM}, high=l_max + bigM)
        add({target[i]: 1, b2[i]: bigM}, high=l_max - 1 + bigM)
        for flag, yy in ((b1[i], y1[u]), (b2[i], y2[u])):
            add({yy: 1, flag: -1}, low=0)
        add({y1[u]: 1} | {b1[i]: -1 for i, (a, b, cc) in enumerate(edges) if a == u}, high=0)
        add({y2[u]: 1} | {b2[i]: -1 for i, (a, b, cc) in enumerate(edges) if a == u}, high=0)

    # Level propagation and exact fan-in minimum.
    for n in names:
        add({lo[n]: 1, li[n]: -1}, -d[n], -d[n])
        if not indeg[n]:
            add({li[n]: 1}, l_init, l_init)
        else:
            for i in indeg[n]:
                add({li[n]: 1, target[i]: -1}, high=0)
            if len(indeg[n]) > 1:
                add({sel[n, i]: 1 for i in indeg[n]}, 1, 1)
                for i in indeg[n]:
                    add({li[n]: 1, target[i]: -1, sel[n, i]: -bigM}, low=-bigM)
            else:
                add({li[n]: 1, target[indeg[n][0]]: -1}, 0, 0)

    res = milp(c=np.array(c, dtype=float), integrality=np.array(integrality),
               bounds=Bounds(np.array(lb), np.array(ub)),
               constraints=assemble(rows, lows, highs, len(c)),
               options={"mip_rel_gap": 0.0, "time_limit": time_limit})
    if not res.success:
        raise RuntimeError(f"SciPy ILP failed: {res.message}")

    lo_sol = {n: int(round(res.x[lo[n]])) for n in names}
    choices = []
    for i, (u, v, ct) in enumerate(edges):
        t = int(round(res.x[target[i]]))
        if round(res.x[b1[i]]):
            choices.append("B1" if t == l_max else f"B1M{t}")
        elif round(res.x[b2[i]]):
            choices.append("B2" if t == l_max - 1 else f"B2M{t}")
        else:
            choices.append("none" if t == lo_sol[u] else f"M{t}")

    result, placed_nodes, placed_edges = score(
        nodes, edges, choices, lo_sol, l_init, l_max)
    model_cost = float(res.fun) + const
    if abs(model_cost - result["latency"]) > 1e-4:
        raise AssertionError((model_cost, result["latency"]))
    b1ct, b2ct = bootstrap_counts(nodes, edges, choices)
    return {"latency": result["latency"], "model_cost": model_cost,
            "b1": b1ct, "b2": b2ct, "nodes": res.mip_node_count,
            "gap": res.mip_gap, "choices": choices}, placed_nodes, placed_edges


# --------------------------------------------------------------------------- #
#  Experiment
# --------------------------------------------------------------------------- #
def cases():
    """Return the six (L_init, L_max) cases used in Table 1 of the report."""
    return [(li, lm) for li in (1, 2) for lm in (14, 15, 16)]


def compare():
    """Print the four-way comparison table (Orbit / SciPy / greedy / baseline)."""
    from orbit_ilp import solve_orbit

    print(f"{'L_init':>6} {'L_max':>6} {'Orbit':>14} {'SciPy':>14} "
          f"{'Greedy':>14} {'Baseline':>14}")
    for li, lm in cases():
        nodes, edges = db.get_attn_empty()
        orb, _, _ = solve_orbit(nodes, edges, li, lm)
        sc, _, _ = solve_scipy(nodes, edges, li, lm)
        gr, _ = greedy_solver.solve_greedy(nodes, edges, li, lm,
                                           latency=db.LATENCY)
        bn, be = db.get_attn_baseline_inserted()
        base = db.evaluate_dag(bn, be, li, lm, db.LATENCY)["latency"]
        print(f"{li:>6} {lm:>6} {orb['latency']:>14.4f} "
              f"{sc['latency']:>14.4f} {gr[0]:>14.4f} {base:>14.4f}")


def main():
    """Parse CLI arguments and run either the comparison or the six ILP cases."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare", action="store_true",
                    help="compare Orbit / SciPy / greedy / baseline")
    args = ap.parse_args()
    if args.compare:
        compare()
        return
    for li, lm in cases():
        nodes, edges = db.get_attn_empty()
        res, _, _ = solve_scipy(nodes, edges, li, lm)
        print(f"L_init={li} L_max={lm}: latency={res['latency']:.4f} "
              f"B1ct={res['b1']} B2ct={res['b2']} nodes={res['nodes']} "
              f"gap={res['gap']:.2e}")


if __name__ == "__main__":
    main()
