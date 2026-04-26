# Monte Carlo flask-mixing prediction (`mctree_predict.py`)

**Date:** 2026-04-24
**Status:** Design approved, pending implementation plan
**Scope:** Replace the (currently beam-search) body of `mctree_predict.py` with a
Monte Carlo flask-mixing simulation. Touches `mctree_predict.py` and adds three
fields to `settings.py`. No changes to `beam_predict.py` or model code.

## Motivation

The file `mctree_predict.py` is currently an exact copy of `beam_predict.py`
(deterministic top-k tree expansion). We want a different inference mode: a
flask-style population simulation where molecules from different ancestries
crossover by being co-sampled into the same packet, then react together via the
existing FlowER mechanistic forward step (which conserves atoms within a
packet).

This gives:

- **Crossover** — products from different reaction lineages can mix and react.
- **Mass conservation per step** — each forward call is one mechanistic step,
  preserved by the BE-matrix flow model.
- **Population view** — the result is a flask `{smiles → weight}` rather than a
  tree of paths, which is a more natural object for downstream
  concentration / probability work.

## Algorithm

For each starting reactant in a chunk:

1. **Initialize flask.** Split the starting reactant on `.` (after running
   through `reactant_process` for explicit Hs and contiguous atom-map numbers).
   Each fragment enters the flask at weight 1.0; duplicate fragments collapse
   with summed weights.
2. **Iterate** for a fixed `mc_max_iter` iterations. No early stopping.
3. **Per iteration:**
   1. Build packets for every flask. The combined number of packets across all
      flasks equals `args.test_batch_size`; each flask gets
      `test_batch_size // n_flasks` packets, with the remainder distributed to
      the first few.
   2. Each packet is assembled by sampling `mc_packet_size` SMILES *with
      replacement*, weighted by flask values, then re-numbering atom maps
      contiguously across the joined mol so packets like `A.A` produce two
      physically distinct atom sets.
   3. Concatenate all packets into one `smiles_list`; remember a `packet_owner`
      list mapping each packet index to its source flask.
   4. Build a `ReactionDataset` + loader, call `expand(args, model, flow,
      loader)` once.
   5. For each packet, look up its product dict in `expand`'s return value. For
      each `(prod_smi, count)`, split `prod_smi` on `.` and deposit
      `count / args.sample_size` into the owner flask for each fragment.
   6. Append a deep copy of each flask to its snapshot list.
4. **Save** `(flask_dict, root_smi, (ori_reactant, products), snapshots)` per
   starting reactant to a pickle in `args.result_path`.

## Architecture

Single file touched: `mctree_predict.py`. One config touch: `settings.py`.

**Removed from `mctree_predict.py`:** `select`, `beam_search`,
`check_if_successful` (beam-specific, not used by MC).

**Reused as-is:** `standardize_smiles`, `reactant_process`, `clean`,
`remove_stereo`, `expand` (with one tiny tweak — see below).

**Added:**

| Function | Purpose |
| --- | --- |
| `init_flask(reactant_smi)` | Build the initial `{frag_smi → 1.0}` flask for a starting reactant. |
| `assemble_packet(flask, k, rng)` | Sample `k` SMILES from the flask weighted by values, with replacement, and return one joined SMILES with re-numbered atom maps. |
| `update_flask(flask, product_smi, weight)` | Single-line additive deposit. Isolated so a future custom-weight rule can swap it in. |
| `run_flask_mc(args, model, flow, flask_list)` | Replaces `beam_search`. Runs the iteration loop. |

**Tweak to `expand`:** drop the `[:args.nbest]` truncation so the full unique
product distribution is returned. `expand` is currently used only by
`mctree_predict.py` (the identical function in `beam_predict.py` is a separate
copy, untouched).

**`main()` changes:** rename `graph_list` to `flask_list`; replace
`graph.add_node(reactant, depth=1)` initialisation with `init_flask(reactant)`;
swap `beam_search(...)` call for `run_flask_mc(...)`; replace the
`check_if_successful` post-processing with a flask-based result tuple. RNG is
created once at the top of `main()` from `args.mc_seed` (XOR'd with the `seed`
arg already accepted by `main`) and threaded into `run_flask_mc`.

**New `Args` fields in `settings.py`:**

```python
mc_packet_size = int(os.environ.get("MC_PACKET_SIZE", 2))
mc_max_iter    = int(os.environ.get("MC_MAX_ITER", 10))
mc_seed        = int(os.environ.get("MC_SEED", 42))
```

Existing `beam_size`, `nbest`, `max_depth` remain in `settings.py` (still used
by `beam_predict.py`) but are simply unused by the MC script.

## Output schema

Per-starting-reactant tuple, pickled into
`{result_path}/result_chunk_{i}_s{seed}.pickle`:

```python
(
    flask: dict[str, float],          # {smiles → cumulative weight}
    root_smi: str,                    # the reactant after reactant_process()
    (ori_reactant: str, products: list[str]),
    snapshots: list[dict[str, float]],  # one flask copy per iteration
)
```

This is a deliberate schema change from the existing `(graph, root, ...)` tuple
in `beam_predict.py`'s pickle. The MC pickle is a separate file (different
script, different output dir setup at the user's discretion), so there is no
back-compat constraint.

## Data flow (one iteration)

Concrete trace for two starting reactants `R0`, `R1` with
`test_batch_size = 8`, `mc_packet_size = 2`:

```
state in:
  flasks = [
    flask_0 = {A: 1.0, B: 0.5, C: 0.25},          # owned by R0
    flask_1 = {X: 1.0, Y: 0.7},                   # owned by R1
  ]

step 1 — quota: 8 packets / 2 flasks = 4 each
  packets_0 = ["A.B", "A.A", "C.A", "B.A"]
  packets_1 = ["X.Y", "X.X", "Y.X", "X.Y"]

step 2 — flatten + remember owners
  smiles_list  = packets_0 + packets_1
  packet_owner = [0,0,0,0, 1,1,1,1]

step 3 — single batched forward call
  overall_dict = expand(args, model, flow, loader)
    # {"A.B": {"P1.P2": 12, "P3": 5, ...},
    #  "A.A": {"A.A":   18, "Q":  3, ...},
    #  ...}

step 4 — per-packet flask updates
  for i, packet_smi in enumerate(smiles_list):
      flask = flasks[packet_owner[i]]
      for prod_smi, count in overall_dict[packet_smi].items():
          weight = count / args.sample_size
          for frag in prod_smi.split('.'):
              update_flask(flask, frag, weight)

step 5 — snapshot
  for fi, flask in enumerate(flasks):
      snapshots[fi].append(copy.deepcopy(flask))
```

**Notes:**

- `expand`'s dedupe-by-input-string is harmless: a packet SMILES that appears
  twice in `smiles_list` produces one model evaluation, and *both* owner slots
  receive the same product distribution. Same input ⇒ same model distribution
  ⇒ correct deposits.
- Self-loops (e.g. `"A.A" → {"A.A": …}`) deposit weight back onto `A`, growing
  its concentration. Terminal molecules naturally accumulate mass and dominate
  sampling — convergence emerges without an early-stop check.

## Error handling & edge cases

- **Packet assembly failures** (RDKit parse / re-numbering): `try/except` per
  packet; on failure, drop the packet and its owner entry from the iteration.
- **Empty flask** (defensive only — additive deposits make this unreachable):
  skip the flask for the iteration; its snapshot reuses the prior step's
  snapshot.
- **`expand()` returns nothing for a packet**: silently no deposits, matching
  current behaviour.
- **Empty product / fragment string**: skip with a one-line warning.
- **`k > unique molecules in flask`**: with-replacement sampling handles
  natively (1-entry flask gives `A.A.A...`).
- **Atom-count sanity**: in `assemble_packet`, assert that the joined mol's
  atom count equals the sum of component atom counts (cheap; catches
  re-numbering bugs).
- **Determinism**: one `np.random.default_rng(args.mc_seed ^ seed)` at the top
  of `main()`, threaded into `run_flask_mc`. Packet assembly is fully
  deterministic for a given seed; model sampling has its own stochasticity.
- **Logging**: per-iteration `log_rank_0` line with iteration index and total
  unique molecules across all flasks.

## Testing

The repo has no `tests/` folder; verification matches existing convention
(manual smoke runs via `scripts/eng.sh`).

1. **Smoke run.** `mc_max_iter=2`, `mc_packet_size=2`, `chunk_size=2` on the
   existing test data. Verify completion and that snapshots have length 2.
2. **Pickle inspection.** Ad-hoc REPL script (not committed) loads the pickle
   and prints top-10 products by weight + the snapshot trajectory per starting
   reactant. Confirms (a) deposits accumulate, (b) original reactant fragments
   stay near weight 1.0, (c) high-probability model products appear at weight
   ≈ `count / sample_size`.
3. **Sampling distribution sanity** (one-off REPL, not committed). Build a
   synthetic 3-entry flask, call `assemble_packet` 1000× with a fixed seed,
   confirm empirical sample frequency matches the normalized weights within
   tolerance.
4. **Determinism check.** Two runs with the same `mc_seed` + `seed` should
   produce byte-identical packet sequences (model output may differ on CUDA
   non-determinism).

No `pytest` framework is added. If a test framework is desired later, it can
be folded in as a follow-up.

## Out of scope

There are two distinct weighting rules in this system. Both are fixed in this
spec; both are intended to become pluggable later, but no hook is pre-built now.

- **Input draw weighting** (in `assemble_packet`). Today: each packet member is
  drawn from the flask with probability proportional to its current flask
  weight (which serves as the running concentration). Future: caller-supplied
  rule that may bias draws by something other than running concentration —
  e.g. an explicit per-fragment initial-concentration vector for starting
  materials, or stoichiometric multipliers. `assemble_packet` is the single
  insertion point for that change.
- **Output deposit weighting** (in `update_flask`). Today: each unique product
  in a packet's prediction adds `count / sample_size` to the flask. Future
  inference systems may want a different rule — e.g. thermodynamic weighting
  of deposits. `update_flask` is the single insertion point for that change.
- **Refactoring shared code with `beam_predict.py`.** The two files duplicate
  several helpers; that cleanup is not part of this work.
- **Distributed (multi-rank) packet sharding.** `run_flask_mc` matches the
  existing `beam_search` rank behaviour (single rank does the work).

## Future follow-ups

- Pluggable input draw-weighting rule via a callable passed into
  `run_flask_mc` (e.g. concentration-aware draws for starting materials).
- Pluggable output deposit-weighting rule via a callable passed into
  `run_flask_mc` (e.g. thermodynamic weighting at inference time).
- Optional early-stop on target-product match (parallel mode to the fixed
  `mc_max_iter` decided here).
- Unify `expand` / `reactant_process` / `clean` between
  `beam_predict.py` and `mctree_predict.py` into a shared module.
