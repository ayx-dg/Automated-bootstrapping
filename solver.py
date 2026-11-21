"""Baseline and B1-only greedy solver for bootstrapping placement.

The greedy heuristic starts with a Bootstrap1 on every edge and repeatedly
removes one bootstrap if the placement stays feasible and its estimated
latency falls.  It is deliberately restricted to Bootstrap1: Bootstrap2 and
Modswitch are left to the exact ILP formulations in ``scipy_ilp.py`` and
``orbit_ilp.py``.  All placements are scored by the reference scorer in
``dag_builder.py``.

Run ``python solver.py`` to print the baseline and greedy latencies for the
MLP and attention blocks at ``L_init = 1, L_max = 14``.
"""

from dag_builder import (
    LATENCY,
    evaluate_dag,
    get_attn_empty,
    get_mlp_empty,
    get_mlp_baseline_inserted,
    get_attn_baseline_inserted,
)


def get_inserted(nodes, edges, booted, share_prefixes=True):
    """Rebuild the inserted DAG for a set of bootstrapped edges.

    A Bootstrap1 node is spliced in directly after each producer whose output
    is refreshed.  When ``share_prefixes`` is true all refreshed outgoing
    edges of a producer share one node, so the refresh is paid once rather
    than once per edge.

    Args:
        nodes: Original ``(name, op, level_cost, description)`` node tuples.
        edges: Original ``(u, v, num_ciphertexts)`` edge tuples.
        booted: Iterable of ``(u, v)`` edges chosen to carry a Bootstrap1.
        share_prefixes: Whether one producer shares a single inserted node.

    Returns:
        A ``(nodes, edges)`` pair for the inserted DAG.

    Raises:
        ValueError: If a shared producer's outgoing edges disagree on #cipher.
    """
    result_nodes, result_edges = list(nodes), []
    shared = {}
    for u, v, ct in edges:
        if (u, v) not in booted:
            result_edges.append((u, v, ct))
            continue
        key = u if share_prefixes else (u, v)
        if key not in shared:
            name = f"greedy_boot_{len(shared):03}"
            shared[key] = (name, ct)
            result_nodes.append((name, "Bootstrap1", 0, f"B1 after {u}"))
            result_edges.append((u, name, ct))
        name, original_ct = shared[key]
        if original_ct != ct:
            raise ValueError("Inconsistent ciphertext counts at fan-out")
        result_edges.append((name, v, ct))
    return result_nodes, result_edges


def report_baseline_mlp():
    """Print the handout baseline latency of the MLP block at L_init = 1."""
    nodes, edges = get_mlp_baseline_inserted()
    print("baseline of mlp is:")
    print(evaluate_dag(nodes, edges, 1, 14, LATENCY)["latency"])


def report_baseline_attn():
    """Print the handout baseline latency of the attention block at L_init = 1."""
    nodes, edges = get_attn_baseline_inserted()
    print("baseline of attention is:")
    print(evaluate_dag(nodes, edges, 1, 14, LATENCY)["latency"])


def solve_greedy(NODES, EDGES, l_init, l_max, latency, share_prefixes=True):
    """Steepest-descent B1-only greedy search.

    Start from a Bootstrap1 on every edge, then remove the single bootstrap
    that lowers latency the most among all feasible single deletions, until
    no feasible deletion improves the score.

    Args:
        NODES, EDGES: The empty block returned by ``dag_builder``.
        l_init: Initial level of the block.
        l_max: Maximum level.
        latency: Latency table passed to ``evaluate_dag``.
        share_prefixes: Whether inserted nodes are shared per producer.

    Returns:
        A pair ``((latency, sorted_booted_edges), [same_result])``.  The
        bracketed second element keeps the historical tie-list interface used
        by ``comparison.py`` and the ILP comparison.  Returns ``(None, [])``
        if the all-B1 starting placement is infeasible.
    """
    latency_of = dict(latency)

    # Initial placement: insert Bootstrap1 on every edge.
    booted = {(u, v) for u, v, _ in EDGES}

    def evaluate(chosen):
        """Rebuild and score one placement; return None if infeasible."""
        nodes2, edges2 = get_inserted(NODES, EDGES, chosen, share_prefixes)
        return evaluate_dag(nodes2, edges2, l_init, l_max, latency_of)

    # Check that the all-B1 placement is feasible.
    res = evaluate(booted)
    if res is None:
        print("Initial placement is infeasible!")
        nodes2, edges2 = get_inserted(NODES, EDGES, booted, share_prefixes)
        print("Original nodes:", len(NODES))
        print("Original edges:", len(EDGES))
        print("Inserted nodes:", len(nodes2))
        print("Inserted edges:", len(edges2))
        return None, []

    best_latency = res["latency"]

    # Repeatedly remove the single bootstrap giving the largest improvement.
    while True:
        best_candidate = None
        for edge in sorted(booted):
            candidate = booted - {edge}
            res = evaluate(candidate)
            if res is None:
                continue
            candidate_latency = res["latency"]
            if candidate_latency < best_latency - 1e-9:
                if (best_candidate is None
                        or candidate_latency < best_candidate[0]):
                    best_candidate = (candidate_latency, candidate)
        if best_candidate is None:
            break
        best_latency, booted = best_candidate

    best = (best_latency, sorted(booted))
    return best, [best]


if __name__ == "__main__":
    report_baseline_mlp()
    nodes, edges = get_mlp_empty()
    print("mlp result:")
    best, ties = solve_greedy(nodes, edges, 1, 14, latency=LATENCY)
    print(best, ties)

    report_baseline_attn()
    nodes, edges = get_attn_empty()
    print("attention result:")
    best, ties = solve_greedy(nodes, edges, 1, 14, latency=LATENCY)
    print(best, ties)
