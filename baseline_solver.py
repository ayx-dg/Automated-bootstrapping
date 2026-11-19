from dag_solver_example import *
from dag_builder import *
from solver import *

l_init = 1
nodes, edges = get_attn_baseline_inserted()
print(LATENCY)
# Reference baseline
for l_max in range(100):
    nodes, edges = get_attn_baseline_inserted()
    baseline = evaluate_dag(
        nodes, edges, l_init, l_max, LATENCY
    )
    if baseline:
        print("Baseline latency:", baseline["latency"])
        print("Baseline minimal feasible l_max", l_max)
        break
# # Greedy placement

nodes, edges = get_attn_empty()
best, _ = solve_greedy(nodes, edges, l_init, l_max, latency=LATENCY)

print("Baseline latency:", baseline["latency"])
print("Greedy latency:", best[0])
print(
    "Improvement:",
    (baseline["latency"] - best[0]) / baseline["latency"] * 100,
    "%"
)
