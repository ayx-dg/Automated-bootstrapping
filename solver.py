from dag_builder import (
    LATENCY,
    evaluate_dag,
    get_attn_empty,
    get_mlp_empty,
    get_mlp_baseline_inserted,      # 新增
    get_attn_baseline_inserted,     # 新增
)


def get_inserted(nodes, edges, booted, share_prefixes=True):
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
    nodes, edges = get_mlp_baseline_inserted()
    print("baseline of mlp is:")
    print(evaluate_dag(nodes, edges, 1, 14, LATENCY)['latency']) 

def report_baseline_attn():
    nodes, edges = get_attn_baseline_inserted()
    print("baseline of attention is:")
    print(evaluate_dag(nodes, edges, 1, 14, LATENCY)['latency']) 


def solve_greedy(NODES, EDGES, l_init, l_max, latency, share_prefixes=True):
    latency_of = dict(latency)

    # 初始状态：每条边都插入 Bootstrap1
    booted = {(u, v) for u, v, _ in EDGES}

    def evaluate(booted):
        nodes2, edges2 = get_inserted(
            NODES, EDGES, booted, share_prefixes
        )
        return evaluate_dag(
            nodes2, edges2, l_init, l_max, latency_of
        )

    # 检查初始 placement 是否可行
    res = evaluate(booted)

    if res is None:
        print("Initial placement is infeasible!")
        nodes2, edges2 = get_inserted(
            NODES, EDGES, booted, share_prefixes
        )
        print("Original nodes:", len(NODES))
        print("Original edges:", len(EDGES))
        print("Inserted nodes:", len(nodes2))
        print("Inserted edges:", len(edges2))
        return None, []

    best_latency = res["latency"]

    # 不断尝试删除 Bootstrap1
    while True:
        best_candidate = None

        for edge in sorted(booted):
            candidate = booted - {edge}
            res = evaluate(candidate)

            if res is None:
                continue

            candidate_latency = res["latency"]

            if candidate_latency < best_latency - 1e-9:
                if (
                    best_candidate is None
                    or candidate_latency < best_candidate[0]
                ):
                    best_candidate = (candidate_latency, candidate)

        # 没有找到更好的 placement
        if best_candidate is None:
            break

        best_latency, booted = best_candidate

    best = (best_latency, sorted(booted))
    return best, [best]


if __name__ == "__main__":
    report_baseline_mlp()
    nodes, edges = get_mlp_empty()
    print("mlp result:")
    best, ties = solve_greedy(
        nodes, edges, 1, 14, latency=LATENCY
    )
    print(best, ties)

    report_baseline_attn()

    nodes, edges = get_attn_empty()
    print("attention result:")
    best, ties = solve_greedy(
            nodes, edges, 1, 14, latency=LATENCY
        )
    print(best, ties)
