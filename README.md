# Automated Bootstrapping Placement

## What this code is about

This repository studies **where to insert FHE Bootstrap operations in an
encrypted MLP / attention block** so that every operator receives enough
levels while the estimated latency is minimized.  A ciphertext enters a block
at level `L_init <= L_max = 14`; multiplications and polynomial evaluations
consume levels, and a Bootstrap restores them at a cost roughly four orders of
magnitude above an addition.  The task is to choose, for every edge, one of
the six chains `{}`, `B1`, `B2`, `M(l)`, `B1M(l)`, `B2M(l)` so the block is
feasible for **every `L_init = 0..14`** at minimum latency.

The DAGs, the affine latency table and the *reference scorer*
(`evaluate_dag`, `depth_of`, `budget_of`) are the ones supplied in the
handout and live unchanged in `dag_builder.py`.  Our contribution is two exact
integer-linear-programming formulations solved with HiGHS through
`scipy.optimize.milp`, plus a B1-only greedy heuristic for comparison.  Every
placement produced by a solver is rebuilt as an inserted DAG and re-scored by
the handout scorer, so the reported numbers use the official cost model.

The results are in `milestone_report_v3.docx` (submitted as
`milestone_report_v3.pdf`); this README reproduces every number in it.

## Setup

Requires **Python 3.12 or newer** and [`uv`](https://docs.astral.sh/uv/)
(any recent version; the lock file was generated with uv 0.11.4).  The
environment is pinned by `pyproject.toml` and `uv.lock`:

```sh
git clone https://github.com/ayx-dg/Automated-bootstrapping.git
cd Automated-bootstrapping
uv sync                 # creates .venv with scipy 1.18.1, matplotlib 3.11.2, ...
```

All commands below are run inside that environment.  Prefix each with
`uv run` (as shown), or activate it once with `. .venv/bin/activate`.
`requirements.txt` lists the same two direct dependencies for anyone who
prefers a different environment manager; `uv.lock` is the pinned, reproducible
environment used for the submitted results.

## Reproducing the report

### Table 1 — four-way comparison

```sh
uv run python scipy_ilp.py --compare
```

Takes **~7 s**.  Expected output (latencies in ms; Orbit and SciPy ILPs are
identical because both are exact and terminate at zero MIP gap):

```
L_init  L_max          Orbit          SciPy         Greedy       Baseline
     1     14     45698.2664     45698.2664     82342.5385     53558.1718
     1     15     45398.0927     45398.0927     84968.6327     56534.5924
     1     16     45398.0927     45398.0927     86555.7703     59511.0130
     2     14     45698.2664     45698.2664     79784.7229     53558.1718
     2     15     45398.0927     45398.0927     81677.5908     56534.5924
     2     16     45398.0927     45398.0927     82531.5021     59511.0130
```

The ILPs are 14.68%–23.71% faster than the baseline; greedy is 38.7%–53.7%
slower.  To run only one ILP on one instance:

```sh
uv run python orbit_ilp.py --l-init 1 --l-max 14   # ~0.05 s
uv run python scipy_ilp.py                         # all six cases, ~2.4 s
```

### Tables 2–3 — full range `L_init = 0..14` for both blocks

```sh
uv run python full_range.py --no-figures
```

Takes **~4 s** and prints one 15-row table per block.  Columns are the
estimated latency `f`, the per-ciphertext bootstrap counts `#B1`/`#B2`
(`#B = #B1 + #B2`), the level-cost `depth`, the level `budget`, the
`utilisation = depth/budget`, the level-agnostic baseline latency and the ILP
gain.  Expected output:

```
=== MLP block (L_max = 14) ===
L_init      ILP f(ms)  #B1  #B2   #B  depth  budget    util       baseline     gain
     0     26069.9524    4    4    8     26     108  0.2407     28129.6897    7.32%
     1     26069.9524    4    4    8     26     109  0.2385     28129.8497    7.32%
     2     26069.9524    4    4    8     26     110  0.2364     28130.0097    7.32%
    ...   (rows 3..12 have the same latency 26069.9524)                       ...
    13     26069.9524    4    4    8     26     121  0.2149     28131.7697    7.33%
    14     21911.5924    0    4    4     26      66  0.3939     28131.9297   22.11%

=== Attention block (L_max = 14) ===
L_init      ILP f(ms)  #B1  #B2   #B  depth  budget    util       baseline     gain
     0     45698.2664   10    8   18     62     244  0.2541     53558.1718   14.68%
     1     45698.2664   10    8   18     62     245  0.2531     53558.1718   14.68%
     2     45698.2664   10    8   18     62     246  0.2520     53558.1718   14.68%
    ...   (rows 3..5 have the same latency 45698.2664)                        ...
     6     41537.6664    6    8   14     62     194  0.3196     53558.1718   22.44%
    ...   (rows 7..14 have the same latency 41537.6664)                       ...
    14     41537.6664    6    8   14     62     202  0.3069     53558.1718   22.44%
```

### Figures 1–2 — placed blocks

```sh
uv run python full_range.py
```

Takes **~8 s**.  In addition to the tables above it writes four compact
placed-block figures to the repository root:

```
[ok] wrote attn_ilp_placement_l1.png     # Figure 1 in the report
[ok] wrote attn_ilp_placement_l9.png
[ok] wrote mlp_ilp_placement_l6.png      # Figure 2 in the report
[ok] wrote mlp_ilp_placement_l14.png
```

Each figure shows every node, every inserted operator (orange = Bootstrap1/2,
blue = Modswitch), the delivered level of every edge, and the summary numbers
`f`, `#B1`, `#B2`, `depth`, `budget`, `utilisation`, as required by Section 4.4
of the handout.

### Baseline latencies quoted in Section 1

```sh
uv run python solver.py
```

Takes **~0.7 s** and prints the MLP and attention baseline latencies at
`L_init = 1, L_max = 14` (28,129.8497 ms and 53,558.1718 ms) together with the
corresponding greedy results.

### Greedy versus baseline (supporting experiment)

```sh
uv run python comparison.py
```

Takes **~2 s**; prints baseline, greedy and relative improvement for the six
`L_init = 1, 2` × `L_max = 14, 15, 16` cases.

### Empty-block DAG figures (handout utility)

```sh
uv run python dag_builder.py
```

Takes **~14 s** and regenerates `mlp_empty_dag.png`, `attn_empty_dag.png`,
`baseline_inserted_dag.png` and `baseline_level_dag.png`.

## Files

- `dag_builder.py` — handout DAGs, latency table and reference scorer (unchanged)
- `scipy_ilp.py` — compact chain-prefix ILP; `--compare` reproduces Table 1
- `orbit_ilp.py` — Orbit-style dense ILP with explicit level variables and big-M
- `full_range.py` — full-range tables (Tables 2–3) and placed-block figures
- `solver.py` — B1-only steepest-descent greedy heuristic
- `comparison.py` — greedy versus baseline experiment
- `dag_solver_example.py` — handout reference example and brute-force template
- `pyproject.toml`, `uv.lock` — pinned Python environment
- `requirements.txt` — plain-pip dependency list
- `milestone_report_v3.docx` / `.pdf` — the report whose numbers are reproduced above

## Notes

- Both ILPs rebuild and re-score every solution with `dag_builder.evaluate_dag()`
  and assert that the model objective matches the scorer latency.
- Bootstrap2 is only offered on 8- and 16-ciphertext edges, as the handout
  requires; refresh cost is charged once per producer via shared-prefix OR
  variables.
- The report and this README describe only the exact ILPs and the B1-only
  greedy heuristic; no other heuristic is needed to reproduce any number.
