# Automated Bootstrapping Placement

Decide where to insert FHE Bootstrap operations in an MLP / attention compute
graph so every operator has enough levels while estimated latency is minimized.
The DAGs, latency table and authoritative scorer live in `dag_builder.py`.

## Setup

Use the system `python` (the repo's `.venv` lacks scipy and matplotlib):

```sh
pip install -r requirements.txt
```

## Reproduce the report's Table 1

```sh
python scipy_ilp.py --compare
```

This evaluates four methods on the attention block for `L_init = 1, 2` and
`L_max = 14, 15, 16` (about 7 seconds) and prints:

```
L_init  L_max          Orbit          SciPy         Greedy       Baseline
     1     14     45698.2664     45698.2664     82342.5385     53558.1718
     1     15     45398.0927     45398.0927     84968.6327     56534.5924
     1     16     45398.0927     45398.0927     86555.7703     59511.0130
     2     14     45698.2664     45698.2664     79784.7229     53558.1718
     2     15     45398.0927     45398.0927     81677.5908     56534.5924
     2     16     45398.0927     45398.0927     82531.5021     59511.0130
```

Both ILP formulations are exact integer programs solved with
`scipy.optimize.milp` (HiGHS) at zero MIP gap; they return identical latencies
on all six instances and reduce latency by 14.68%–23.71% over the baseline.
The greedy solver is B1-only (no Bootstrap2 or Modswitch) and is 38.7%–53.7%
slower than the baseline. These are the numbers in Section 3 of
`milestone_report_v2.docx`.

## Other entrypoints

```sh
python orbit_ilp.py [--l-init L] [--l-max L]   # Orbit-style ILP alone
python scipy_ilp.py                            # SciPy ILP on the six cases
python comparison.py                           # greedy vs baseline on attention
python solver.py                               # greedy on MLP (L_init=1, L_max=14)
python dag_builder.py                          # regenerate DAG / level figures
```

## Files

- `dag_builder.py` — DAG definitions, `LATENCY` table, `evaluate_dag()` scorer
- `orbit_ilp.py` — Orbit-style dense ILP (explicit level variables, big-M)
- `scipy_ilp.py` — compact chain-prefix ILP and the `--compare` entrypoint
- `solver.py` — B1-only greedy solver
- `comparison.py` — greedy vs baseline experiment
- `dag_solver_example.py` — example DAG solver
