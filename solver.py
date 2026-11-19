from dag_solver_example import *
from dag_builder import *
def solve_greedy(NODES, EDGES, l_init, l_max, latency):

    latency_of = dict(latency)
    latency_of["Bootstrap1"] = (
        lambda l, lm=l_max: latency_bootstrap(lm)
    )
    # 初始状态：每条边都插入 Bootstrap1
    booted = {(u, v) for u, v, _ in EDGES}

    def get_inserted(NODES, EDGES, booted):
        nodes = list(NODES)
        edges = []

        for u, v, ciphers in EDGES:
            if (u, v) in booted:
                boot = f"Boot1_{u}_{v}"
                nodes.append(
                    (boot, "Bootstrap1", 0, f"Boot1({u} -> {v})")
                )
                edges.extend([
                    (u, boot, ciphers),
                    (boot, v, ciphers)
                ])
            else:
                edges.append((u, v, ciphers))

        return nodes, edges

    def evaluate(booted):
        nodes2, edges2 = get_inserted(NODES, EDGES, booted)
        return evaluate_dag(
           nodes2, edges2, l_init, l_max, latency_of
        )

    missing = {
    op for _, op, *_ in NODES
    if op not in latency_of
}
    print("Missing latency operators:" + str(missing) if missing != set() else '')
    # 检查初始 placement 是否可行
    res = evaluate(booted)

    if res is None:
        print("Initial placement is infeasible!")
        nodes2, edges2 = get_inserted(NODES, EDGES, booted)
        print("Original nodes:", len(NODES))
        print("Original edges:", len(EDGES))
        print("Inserted nodes:", len(nodes2))
        print("Inserted edges:", len(edges2))
        return None, []
    else:
        print("Initial latency:", res["latency"])

    best_latency = res["latency"]

    # 不断尝试删除 Bootstrap1
    while True:
        best_candidate = None

        for edge in sorted(booted):
            candidate = booted - {edge}
            res = evaluate(candidate)

            if res is None:
                continue

            latency = res["latency"]

            if latency < best_latency - 1e-9:
                if (
                    best_candidate is None
                    or latency < best_candidate[0]
                ):
                    best_candidate = (latency, candidate)

        # 没有找到更好的 placement
        if best_candidate is None:
            break

        best_latency, booted = best_candidate

    best = (best_latency, sorted(booted))
    return best, [best]

#from dag_sodlver_example import latency_table
if __name__ == "__main__":

    from dag_solver_example import *
    from dag_builder import *

    #from dag_solver_example import latency_bootstrap
    nodes, edges = get_attn_empty()
    best, ties = solve_greedy(nodes, edges, 1, 10, latency=LATENCY)
    print(best, ties)
