"""Reproduce the full-range results of Section 3 (L_init = 0..14, L_max = 14).

For both blocks (MLP, attention) and every L_init, solve the compact SciPy ILP
(the Orbit-style ILP returns the same optimum), then report latency, #B1/#B2,
depth, budget and utilisation, and draw compact placed-block figures for two
input levels.

Run:
    python full_range.py            # tables to stdout, figures to PNG
    python full_range.py --no-figures

The figures use a serpentine layered layout (the long chain is folded into a
few columns) so they stay compact enough for the report's page limit.  Every
node, every inserted operator, the level of every edge and the four summary
numbers are shown, as required by Section 4.4 of the handout.
"""

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import dag_builder as db
from scipy_ilp import solve_scipy

BLOCKS = (("MLP", db.get_mlp_empty, db.get_mlp_baseline_inserted),
          ("Attention", db.get_attn_empty, db.get_attn_baseline_inserted))

# Palette kept in sync with dag_builder so inserted operators stand out.
_FILL = {
    "Bootstrap1": "#FDE2B6", "Bootstrap2": "#F6C98B",
    "Modswitch": "#DCE9F7", "Input": "#E8F2E2", "Output": "#E8F2E2",
}
_EDGE = {
    "Bootstrap1": "#B9741A", "Bootstrap2": "#95550A",
    "Modswitch": "#3E6FA8", "Input": "#4F7A45", "Output": "#4F7A45",
}


def metrics(nodes, edges, l_init, l_max=14):
    """Solve the ILP and return the Section 4.4 numbers for its placement.

    Returns a dict with the latency, the per-ciphertext bootstrap counts, the
    level-cost depth, the level budget, their ratio (utilisation), and the
    rebuilt inserted DAG for plotting.
    """
    res, placed_nodes, placed_edges = solve_scipy(nodes, edges, l_init, l_max)
    depth = db.depth_of(placed_nodes, placed_edges)
    b1, b2, budget = db.budget_of(placed_nodes, placed_edges, l_init, l_max)
    return {
        "latency": res["latency"], "b1": b1, "b2": b2,
        "depth": depth, "budget": budget,
        "util": depth / budget if budget else 0.0,
        "placed_nodes": placed_nodes, "placed_edges": placed_edges,
    }


def baseline_latency(get_baseline, l_init, l_max=14):
    """Estimated latency of the handout's level-agnostic reference placement."""
    bn, be = get_baseline()
    res = db.evaluate_dag(bn, be, l_init, l_max, db.LATENCY)
    return res["latency"] if res else None


def visualize_compact(nodes, edges, levels, title, subtitle, filename,
                      per_col=10, fig_w=13.0):
    """Draw a placed block as a compact, serpentine-layered DAG.

    The chain is folded into columns of ``per_col`` layers; alternate columns
    run in opposite vertical directions so consecutive layers stay adjacent.
    Every node (including inserted Bootstrap/Modswitch nodes), every edge
    level and the summary numbers are labelled.  Unlike
    :func:`dag_builder.visualize_empty_dag` this keeps the figure close to
    square so two of them fit on a report page.
    """
    topo, succ, pred = db._topological_order(nodes, edges)
    layer = {}
    for u in topo:
        layer[u] = max((layer[p] for p in pred[u]), default=-1) + 1
    n_layer = max(layer.values()) + 1
    n_col = math.ceil(n_layer / per_col)

    # Index of each node inside its layer (most layers hold one node).
    seen, idx = {}, {}
    for u in topo:
        L = layer[u]
        idx[u] = seen.get(L, 0)
        seen[L] = seen.get(L, 0) + 1
    max_per = max(seen.values())

    col_w, sub_x, row_h = 7.0, 3.2, 2.4
    pos = {}
    for u in topo:
        c, r = divmod(layer[u], per_col)
        if c % 2:
            r = per_col - 1 - r
        pos[u] = (c * col_w + idx[u] * sub_x, -r * row_h)

    span_x = (n_col - 1) * col_w + (max_per - 1) * sub_x + 6
    span_y = (per_col - 1) * row_h + 4
    fig_h = fig_w * span_y / span_x
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_title(title, fontsize=11, color="#1E2A36", pad=14)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8, color="#4A5866")
    ax.axis("off")

    for u, v, ct in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
            lw=0.9, color="#7A8794", zorder=1,
            shrinkA=22, shrinkB=22, connectionstyle="arc3,rad=0.06"))
        lbl = f"{levels.get((u, v), '?')}"
        ax.text((x0 + x1) / 2 + 0.18, (y0 + y1) / 2, f"L={lbl}",
                fontsize=5.0, color="#9A5B10", zorder=2,
                bbox=dict(boxstyle="round,pad=0.06", facecolor="white",
                          edgecolor="none", alpha=0.9))

    bw, bh = 3.0, 1.45
    for name, op, d, desc in nodes:
        x, y = pos[name]
        fill = _FILL.get(op, "#EDEFF1")
        ec = _EDGE.get(op, "#5A6672")
        ax.add_patch(FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            lw=0.9, facecolor=fill, edgecolor=ec, zorder=3))
        ax.text(x, y + 0.28, name, ha="center", va="center",
                fontsize=5.8, fontweight="bold", color="#1E2A36", zorder=4)
        ax.text(x, y - 0.30, f"{op} d={d}", ha="center", va="center",
                fontsize=5.0, color="#3E4B58", zorder=4)

    ax.set_xlim(-3, span_x - 3)
    ax.set_ylim(-span_y + 2, 2)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[ok] wrote {filename}")


def figure(name, get_empty, l_init, l_max, filename, per_col):
    """Solve one instance and draw its compact placed-block figure."""
    nodes, edges = get_empty()
    m = metrics(nodes, edges, l_init, l_max)
    levels = db.evaluate_dag(m["placed_nodes"], m["placed_edges"],
                             l_init, l_max, db.LATENCY)["edge_lvl"]
    title = f"{name} block: optimal placement at L_init = {l_init} (L_max = {l_max})"
    subtitle = (f"f = {m['latency']:.2f} ms   |   #B1 = {m['b1']}   "
                f"#B2 = {m['b2']}   |   depth = {m['depth']}   "
                f"budget = {m['budget']}   |   utilisation = {m['util']:.4f}")
    visualize_compact(m["placed_nodes"], m["placed_edges"], levels,
                      title, subtitle, filename, per_col=per_col)


def main():
    """Print the full-range tables and optionally write the figures."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    for name, get_empty, get_baseline in BLOCKS:
        nodes, edges = get_empty()
        print(f"\n=== {name} block (L_max = 14) ===")
        print(f"{'L_init':>6} {'ILP f(ms)':>14} {'#B1':>4} {'#B2':>4} "
              f"{'#B':>4} {'depth':>6} {'budget':>7} {'util':>7} "
              f"{'baseline':>14} {'gain':>8}")
        for li in range(15):
            m = metrics(nodes, edges, li)
            base = baseline_latency(get_baseline, li)
            gain = (base - m["latency"]) / base * 100 if base else float("nan")
            print(f"{li:>6} {m['latency']:>14.4f} {m['b1']:>4} {m['b2']:>4} "
                  f"{m['b1'] + m['b2']:>4} {m['depth']:>6} {m['budget']:>7} "
                  f"{m['util']:>7.4f} {base:>14.4f} {gain:>7.2f}%")

    if not args.no_figures:
        figure("Attention", db.get_attn_empty, 1, 14, "attn_ilp_placement_l1.png", 10)
        figure("Attention", db.get_attn_empty, 9, 14, "attn_ilp_placement_l9.png", 10)
        figure("MLP", db.get_mlp_empty, 6, 14, "mlp_ilp_placement_l6.png", 7)
        figure("MLP", db.get_mlp_empty, 14, 14, "mlp_ilp_placement_l14.png", 7)


if __name__ == "__main__":
    main()
