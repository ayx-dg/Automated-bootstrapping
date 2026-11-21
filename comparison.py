from dag_solver_example import *
from dag_builder import *
from solver import *


for l_init in (1, 2):
    print(f"L_init = {l_init}")

    for l_max in range(14, 17):
        nodes, edges = get_attn_baseline_inserted()
        baseline = evaluate_dag(
            nodes, edges, l_init, l_max, LATENCY
        )

        nodes, edges = get_attn_empty()
        greedy, _ = solve_greedy(
            nodes, edges, l_init, l_max, latency=LATENCY
        )

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
                    baseline["latency"] - greedy_latency
                ) / baseline["latency"] * 100
                print(f"    Improvement: {improvement:.2f}%")
        else:
            print("    Greedy latency: infeasible")


