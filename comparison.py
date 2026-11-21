"""Greedy versus baseline comparison on the attention block.

Reproduces the two right-hand columns of Table 1 in the report: for
``L_init = 1, 2`` and ``L_max = 14, 15, 16`` it prints the estimated latency
of the handout's level-agnostic baseline and of the B1-only greedy solver in
``solver.py``.  The exact ILP results in the same table are produced by
``python scipy_ilp.py --compare``.

Run with ``python comparison.py``.
"""

from dag_builder import LATENCY, evaluate_dag, get_attn_baseline_inserted, get_attn_empty
from solver import solve_greedy


def main():
    """Print baseline, greedy and relative improvement for the six cases."""
    for l_init in (1, 2):
        print(f"L_init = {l_init}")

        for l_max in range(14, 17):
            nodes, edges = get_attn_baseline_inserted()
            baseline = evaluate_dag(nodes, edges, l_init, l_max, LATENCY)

            nodes, edges = get_attn_empty()
            greedy, _ = solve_greedy(nodes, edges, l_init, l_max, latency=LATENCY)

            print(f"  L_max = {l_max}")
            if baseline:
                print(f"    Baseline latency: {baseline['latency']}")
            else:
                print("    Baseline latency: infeasible")

            if greedy:
                greedy_latency = greedy[0]
                print(f"    Greedy latency: {greedy_latency}")
                if baseline:
                    improvement = (
                        (baseline["latency"] - greedy_latency)
                        / baseline["latency"] * 100
                    )
                    print(f"    Improvement: {improvement:.2f}%")
            else:
                print("    Greedy latency: infeasible")


if __name__ == "__main__":
    main()
