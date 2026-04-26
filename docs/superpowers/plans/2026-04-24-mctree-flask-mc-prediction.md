# Monte Carlo flask-mixing prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the (currently beam-search) body of `mctree_predict.py` with a Monte Carlo flask-mixing simulation. Each forward step is one mechanistic reaction on a small "packet" of molecules sampled from a per-reactant flask, with products redeposited into the flask. After `mc_max_iter` iterations the flask + per-iteration snapshots are pickled.

**Architecture:** Single-file change in `mctree_predict.py` (Approach A from the spec). Reuse the existing `expand()` for the batched forward call (with `nbest` truncation removed). Add four new helpers: `init_flask`, `assemble_packet`, `update_flask`, `run_flask_mc`. Three new `Args` fields in `settings.py` (`mc_packet_size`, `mc_max_iter`, `mc_seed`).

**Tech Stack:** PyTorch (model), RDKit (`MolFromSmiles` with the existing `ps` parser params for explicit-H SMILES, `CombineMols`, `MolToSmiles`), NumPy (`default_rng` for reproducible weighted sampling), pickle (output).

**Spec:** `docs/superpowers/specs/2026-04-24-mctree-flask-mc-prediction-design.md`

**Notes for the implementing engineer:**

- `docs/` is in `.gitignore` for this repo. The user has chosen NOT to commit the spec or this plan. Code commits to tracked files (`mctree_predict.py`, `settings.py`) are normal — include them as planned.
- This repo does not use `pytest`. The spec explicitly defers adding a test framework. Verification per task is via `python -c "..."` snippets and a final smoke run of the existing driver script. If a snippet's import line fails because the module pulls in CUDA at import-time, fall back to copying just the helper into a scratch script.
- The repo has explicit-H atom-map-numbered SMILES throughout. The parser to use when re-parsing flask members is `utils.data_utils.ps` (already imported at the top of `mctree_predict.py`). It's `SmilesParserParams` with `removeHs=False`.
- Atom map numbers are used by the model as **array indices** (`smi2vocabid` indexes `atom.GetIntProp('molAtomMapNumber') - 1`). Therefore every SMILES handed to the model MUST have atom map numbers `1..N` contiguously. `assemble_packet` is responsible for enforcing this.
- The branch is `flask_sampling`. Make small commits along the way; each task ends with a commit step.

---

## File Structure

| File | Change | Responsibility |
| --- | --- | --- |
| `settings.py` | Modify (~3 lines added) | Add `mc_packet_size`, `mc_max_iter`, `mc_seed` to the `Args` class. |
| `mctree_predict.py` | Modify (substantive) | Add 4 new helpers (`init_flask`, `assemble_packet`, `update_flask`, `run_flask_mc`); tweak `expand` to drop `nbest` truncation; replace `main()`'s beam-search bits with MC initialization and loop; remove `select`, `beam_search`, `check_if_successful`, and the now-unused `networkx` import. |

No new files are created. No tests folder is added.

---

## Task 1 — Add MC config fields to `settings.py`

**Files:**
- Modify: `settings.py`

- [ ] **Step 1: Add three Args fields**

Open `settings.py`. After the `# Testing uniform weighting #` block (currently the last lines of the `Args` class, ~line 91-92), add:

```python
    # monte-carlo flask sampling #
    mc_packet_size = int(os.environ.get("MC_PACKET_SIZE", 2))
    mc_max_iter    = int(os.environ.get("MC_MAX_ITER", 10))
    mc_seed        = int(os.environ.get("MC_SEED", 42))
```

- [ ] **Step 2: Verify the import still works**

Run:
```bash
python -c "from settings import Args; print(Args.mc_packet_size, Args.mc_max_iter, Args.mc_seed)"
```
Expected output:
```
2 10 42
```

- [ ] **Step 3: Verify env-var override works**

Run:
```bash
MC_PACKET_SIZE=3 MC_MAX_ITER=5 MC_SEED=7 python -c "from settings import Args; print(Args.mc_packet_size, Args.mc_max_iter, Args.mc_seed)"
```
Expected output:
```
3 5 7
```

- [ ] **Step 4: Commit**

```bash
git add settings.py
git commit -m "Add MC config fields (mc_packet_size, mc_max_iter, mc_seed)"
```

---

## Task 2 — Drop `nbest` truncation from `expand` in `mctree_predict.py`

**Files:**
- Modify: `mctree_predict.py:106`

The current `expand` truncates each input's product dict to `args.nbest`. The MC code path needs the full unique-product distribution. The identical `expand` in `beam_predict.py` is a separate copy and is unaffected.

- [ ] **Step 1: Remove the truncation line**

In `mctree_predict.py`, replace:

```python
            pred_smis_tuples = sorted(pred_smis_dict.items(), key=lambda x: x[1], reverse=True)
            
            pred_smis_dict = dict(pred_smis_tuples[:args.nbest])
            overall_dict[reac_smi] = pred_smis_dict
```

with:

```python
            pred_smis_tuples = sorted(pred_smis_dict.items(), key=lambda x: x[1], reverse=True)

            overall_dict[reac_smi] = dict(pred_smis_tuples)
```

(Keeps the sort order — convenient for downstream inspection — but no length cap.)

- [ ] **Step 2: Smoke-import the file**

Run:
```bash
python -c "import mctree_predict; print(mctree_predict.expand.__name__)"
```
Expected output:
```
expand
```

If this fails because of an unrelated import error from the existing file (e.g. `train.py` requires CUDA), record the error but proceed — the smoke run in Task 8 will catch true regressions.

- [ ] **Step 3: Commit**

```bash
git add mctree_predict.py
git commit -m "mctree_predict: stop truncating expand() output to nbest"
```

---

## Task 3 — Add `init_flask` helper

**Files:**
- Modify: `mctree_predict.py` (add new function near `reactant_process`)

- [ ] **Step 1: Add a verification scratch script**

Create a temporary file (gitignored, will be deleted at end of plan):

```bash
mkdir -p /tmp/mc_verify
cat > /tmp/mc_verify/test_init_flask.py <<'EOF'
import sys, os
sys.path.insert(0, os.path.expanduser("~/FlowER/FlowERrs"))
from mctree_predict import init_flask

# Ethanol + water as starting materials
flask = init_flask("CCO.O")
print("flask:", flask)
print("n_unique:", len(flask))
print("total_weight:", sum(flask.values()))

assert len(flask) == 2, f"expected 2 fragments, got {len(flask)}"
assert all(w == 1.0 for w in flask.values()), "expected each fragment at weight 1.0"
assert sum(flask.values()) == 2.0
print("OK")
EOF
```

- [ ] **Step 2: Run it (expect failure — `init_flask` not yet defined)**

```bash
python /tmp/mc_verify/test_init_flask.py
```
Expected: `ImportError: cannot import name 'init_flask' from 'mctree_predict'`.

- [ ] **Step 3: Add `init_flask` to `mctree_predict.py`**

Insert this function in `mctree_predict.py` immediately after the `clean(smi)` function (around line 129):

```python
def init_flask(reactant_smi):
    """Build the initial flask for a starting reactant.

    Splits the processed reactant on '.' and assigns weight 1.0 to each
    fragment. Duplicate fragments collapse with summed weights.
    """
    processed = reactant_process(reactant_smi)
    flask = {}
    for frag in processed.split('.'):
        if not frag:
            continue
        flask[frag] = flask.get(frag, 0.0) + 1.0
    return flask
```

- [ ] **Step 4: Re-run verification (expect pass)**

```bash
python /tmp/mc_verify/test_init_flask.py
```
Expected: prints the flask, then `OK`. The fragments will be the explicit-H, atom-map-numbered SMILES forms of ethanol and water (something like `[H][O:8][H]` for water; exact map numbers depend on `reactant_process`).

- [ ] **Step 5: Commit**

```bash
git add mctree_predict.py
git commit -m "mctree_predict: add init_flask helper"
```

---

## Task 4 — Add `update_flask` helper

**Files:**
- Modify: `mctree_predict.py`

- [ ] **Step 1: Add a verification scratch script**

```bash
cat > /tmp/mc_verify/test_update_flask.py <<'EOF'
import sys, os
sys.path.insert(0, os.path.expanduser("~/FlowER/FlowERrs"))
from mctree_predict import update_flask

flask = {"A": 1.0, "B": 0.5}
update_flask(flask, "A", 0.25)
update_flask(flask, "C", 0.75)

assert flask["A"] == 1.25, flask
assert flask["B"] == 0.5, flask
assert flask["C"] == 0.75, flask
print("OK", flask)
EOF
```

- [ ] **Step 2: Run it (expect failure)**

```bash
python /tmp/mc_verify/test_update_flask.py
```
Expected: `ImportError: cannot import name 'update_flask'`.

- [ ] **Step 3: Add `update_flask` to `mctree_predict.py`**

Insert immediately after `init_flask`:

```python
def update_flask(flask, product_smi, weight):
    """Additive deposit of `weight` onto `product_smi` in `flask`.

    Single insertion point for future per-product weighting rules
    (e.g. thermodynamic weighting at inference time).
    """
    flask[product_smi] = flask.get(product_smi, 0.0) + weight
```

- [ ] **Step 4: Re-run verification (expect pass)**

```bash
python /tmp/mc_verify/test_update_flask.py
```
Expected:
```
OK {'A': 1.25, 'B': 0.5, 'C': 0.75}
```

- [ ] **Step 5: Commit**

```bash
git add mctree_predict.py
git commit -m "mctree_predict: add update_flask helper"
```

---

## Task 5 — Add `assemble_packet` helper (with atom-map renumbering)

This is the most subtle helper: it samples `k` SMILES from the flask weighted by flask values (with replacement), parses each with the explicit-H parser, strips and reassigns atom map numbers `1..N` contiguously across the combined molecule, and returns one joined SMILES string.

**Files:**
- Modify: `mctree_predict.py`

- [ ] **Step 1: Add a verification scratch script**

```bash
cat > /tmp/mc_verify/test_assemble_packet.py <<'EOF'
import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser("~/FlowER/FlowERrs"))
from rdkit import Chem
from mctree_predict import init_flask, assemble_packet
from utils.data_utils import ps

# Build a flask with ethanol and water at unequal weights
flask = init_flask("CCO.O")
# Re-weight so we can check sampling distribution
keys = list(flask.keys())
flask = {keys[0]: 3.0, keys[1]: 1.0}
print("flask:", flask)

rng = np.random.default_rng(0)

# 1) Single-molecule packet (k=1) should still be valid SMILES with maps 1..N
pkt1 = assemble_packet(flask, k=1, rng=rng)
m = Chem.MolFromSmiles(pkt1, ps)
assert m is not None, f"failed to reparse k=1 packet: {pkt1}"
maps = sorted(a.GetIntProp('molAtomMapNumber') for a in m.GetAtoms())
assert maps == list(range(1, m.GetNumAtoms() + 1)), f"non-contiguous maps: {maps}"
print("k=1 OK:", pkt1)

# 2) Two-molecule packet should have atom maps 1..N (no collisions)
pkt2 = assemble_packet(flask, k=2, rng=rng)
m = Chem.MolFromSmiles(pkt2, ps)
assert m is not None, f"failed to reparse k=2 packet: {pkt2}"
maps = sorted(a.GetIntProp('molAtomMapNumber') for a in m.GetAtoms())
assert maps == list(range(1, m.GetNumAtoms() + 1)), f"non-contiguous maps: {maps}"
print("k=2 OK:", pkt2)

# 3) k=2 packet must contain >=2 disconnected components (water + something,
#    or two waters, etc.) — verify by counting fragments via RDKit
frags = Chem.GetMolFrags(m, asMols=False)
assert len(frags) == 2, f"expected 2 fragments in k=2 packet, got {len(frags)}: {pkt2}"
print("fragments:", frags)

# 4) Empirical sampling distribution (k=1, 1000 draws) should track weights
counts = {keys[0]: 0, keys[1]: 0}
rng2 = np.random.default_rng(123)
for _ in range(1000):
    p = assemble_packet(flask, k=1, rng=rng2)
    # Count which starting fragment landed in the packet by atom count
    m = Chem.MolFromSmiles(p, ps)
    n_atoms = m.GetNumAtoms()
    # ethanol has 9 explicit atoms (C2H6O), water has 3 (H2O); pick whichever
    if n_atoms == 3:
        counts[keys[1]] += 1   # water
    else:
        counts[keys[0]] += 1   # ethanol
print("empirical counts (1000 draws):", counts)
# Expected ratio is 3:1; tolerate 0.65–0.85 for ethanol fraction
ratio = counts[keys[0]] / 1000
assert 0.65 <= ratio <= 0.85, f"sampling ratio off: {ratio}"
print("distribution OK")

print("ALL OK")
EOF
```

- [ ] **Step 2: Run it (expect failure)**

```bash
python /tmp/mc_verify/test_assemble_packet.py
```
Expected: `ImportError: cannot import name 'assemble_packet'`.

- [ ] **Step 3: Add `assemble_packet` to `mctree_predict.py`**

Insert immediately after `update_flask`:

```python
def assemble_packet(flask, k, rng):
    """Build one packet SMILES by sampling k flask members weighted by their
    current values (with replacement) and reassigning atom map numbers
    1..N contiguously across the joined mol.

    Raises any RDKit error on parse/serialize failure; the caller is
    responsible for catching and skipping the packet.
    """
    smis = list(flask.keys())
    weights = np.array([flask[s] for s in smis], dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("flask has zero total weight")
    probs = weights / total
    chosen = rng.choice(len(smis), size=k, replace=True, p=probs)

    combined = None
    expected_atom_count = 0
    for idx in chosen:
        m = Chem.MolFromSmiles(smis[idx], ps)
        if m is None:
            raise ValueError(f"could not parse flask SMILES: {smis[idx]!r}")
        for atom in m.GetAtoms():
            atom.SetAtomMapNum(0)
        expected_atom_count += m.GetNumAtoms()
        combined = m if combined is None else Chem.CombineMols(combined, m)

    for i, atom in enumerate(combined.GetAtoms()):
        atom.SetAtomMapNum(i + 1)

    assert combined.GetNumAtoms() == expected_atom_count, \
        f"atom count mismatch: {combined.GetNumAtoms()} vs {expected_atom_count}"

    return Chem.MolToSmiles(combined, isomericSmiles=False, allHsExplicit=True)
```

- [ ] **Step 4: Re-run verification (expect pass)**

```bash
python /tmp/mc_verify/test_assemble_packet.py
```
Expected: prints each step's OK marker, then `ALL OK`. The empirical 1000-draw fraction for the heavier weight should be in `[0.65, 0.85]`.

- [ ] **Step 5: Commit**

```bash
git add mctree_predict.py
git commit -m "mctree_predict: add assemble_packet (weighted sampling + atom-map renumbering)"
```

---

## Task 6 — Add `run_flask_mc` loop

The iteration loop: for each iteration, build packets across all flasks (filling `test_batch_size`), call `expand` once, deposit products back. Returns per-flask snapshot lists.

**Files:**
- Modify: `mctree_predict.py` — add `import copy` near the top, add `run_flask_mc` near `beam_search`.

- [ ] **Step 1: Add `import copy` near the top of `mctree_predict.py`**

After `import pickle` (around line 14), add:

```python
import copy
```

- [ ] **Step 2: Add `run_flask_mc` to `mctree_predict.py`**

Insert this function immediately above (or in place of — see Task 7) the `beam_search` function:

```python
def run_flask_mc(args, model, flow, flask_list, rng):
    """Run the Monte Carlo flask-mixing loop for `args.mc_max_iter` iterations.

    `flask_list` is a list of (flask_dict, root_smi, (ori_reactant, products))
    tuples. Each iteration assembles up to args.test_batch_size packets across
    all flasks, calls expand() once, and deposits count/sample_size into the
    owner flask for each product fragment.

    Returns a parallel list `snapshots_list[fi]` of per-iteration deepcopies of
    flask `fi`.
    """
    mc_max_iter = args.mc_max_iter
    k = args.mc_packet_size
    test_batch_size = args.test_batch_size
    sample_size = args.sample_size

    n_flasks = len(flask_list)
    snapshots_list = [[] for _ in range(n_flasks)]
    if n_flasks == 0:
        return snapshots_list

    base_quota = test_batch_size // n_flasks
    remainder = test_batch_size % n_flasks

    for it in range(mc_max_iter):
        smiles_list = []
        packet_owner = []
        for fi, (flask, _root, _meta) in enumerate(flask_list):
            quota = base_quota + (1 if fi < remainder else 0)
            for _ in range(quota):
                try:
                    packet = assemble_packet(flask, k, rng)
                except Exception as e:
                    log_rank_0(f"MC iter {it+1}: flask {fi} packet skipped: {e}")
                    continue
                smiles_list.append(packet)
                packet_owner.append(fi)

        if not smiles_list:
            log_rank_0(f"MC iter {it+1}/{mc_max_iter}: no packets assembled")
            for fi, (flask, _r, _m) in enumerate(flask_list):
                snapshots_list[fi].append(copy.deepcopy(flask))
            continue

        test_dataset = ReactionDataset(args, smiles_list, reactant_only=True)
        try:
            test_loader = init_loader(args, test_dataset,
                                      batch_size=test_batch_size,
                                      shuffle=False, epoch=None, use_sort=False)
        except Exception as e:
            log_rank_0(f"MC iter {it+1}/{mc_max_iter}: loader init failed: {e}")
            for fi, (flask, _r, _m) in enumerate(flask_list):
                snapshots_list[fi].append(copy.deepcopy(flask))
            continue

        overall_dict = expand(args, model, flow, test_loader)

        for i, packet_smi in enumerate(smiles_list):
            flask = flask_list[packet_owner[i]][0]
            product_dict = overall_dict.get(packet_smi, {})
            for prod_smi, count in product_dict.items():
                if not prod_smi:
                    continue
                weight = count / sample_size
                for frag in prod_smi.split('.'):
                    if not frag:
                        continue
                    update_flask(flask, frag, weight)

        for fi, (flask, _r, _m) in enumerate(flask_list):
            snapshots_list[fi].append(copy.deepcopy(flask))

        n_unique = sum(len(flask_list[fi][0]) for fi in range(n_flasks))
        log_rank_0(
            f"MC iter {it+1}/{mc_max_iter}: "
            f"{len(smiles_list)} packets, {n_unique} unique molecules across {n_flasks} flasks"
        )

    return snapshots_list
```

- [ ] **Step 3: Smoke-import the file**

```bash
python -c "import mctree_predict; print(mctree_predict.run_flask_mc.__name__)"
```
Expected output: `run_flask_mc`.

If this fails because the existing file does not import cleanly outside of CUDA, record the error and proceed — the end-to-end smoke run in Task 8 is the real check.

- [ ] **Step 4: Commit**

```bash
git add mctree_predict.py
git commit -m "mctree_predict: add run_flask_mc loop"
```

---

## Task 7 — Replace `main()` body and remove dead beam helpers

Now the helpers are in place; rewrite `main()` to use `init_flask` + `run_flask_mc`, change the pickle output to the new schema, and remove `select`, `beam_search`, `check_if_successful`, and the now-unused `import networkx as nx`.

**Files:**
- Modify: `mctree_predict.py` (substantive edits to `main()` and removal of three functions + one import)

- [ ] **Step 1: Remove the now-unused `networkx` import**

In `mctree_predict.py`, delete:

```python
import networkx as nx
```

- [ ] **Step 2: Delete the three beam helpers**

Delete the entire bodies of these functions from `mctree_predict.py`:

- `select(args, frontiers_dict, graph_list)`
- `beam_search(args, model, flow, frontiers_dict, graph_list)`
- `check_if_successful(graph, products)`

Keep `standardize_smiles`, `expand`, `reactant_process`, `clean`, `remove_stereo`, `init_flask`, `update_flask`, `assemble_packet`, `run_flask_mc`.

- [ ] **Step 3: Rewrite `main()`**

Replace the existing `main()` function body. The new version reads:

```python
def main(args, seed=0):
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = args.device
    if args.local_rank != -1:
        dist.init_process_group(backend=args.backend, init_method='env://', timeout=datetime.timedelta(0, 7200))
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = True

    with open(args.test_path, 'r') as test_o:
        test_smiles_list = test_o.readlines()

    chunk_size = args.chunk_size
    chunked_list = [test_smiles_list[i:i + chunk_size] for i in range(0, len(test_smiles_list), chunk_size)]

    rng = np.random.default_rng(args.mc_seed ^ seed)

    for i, chunk in enumerate(chunked_list):
        log_rank_0(f"Group Chunk-{i} called:")
        checkpoint = os.path.join(args.model_path, args.model_name)
        state = torch.load(checkpoint, weights_only=False, map_location=device)
        pretrain_args = state["args"]
        pretrain_args.load_from = None
        pretrain_args.device = device

        pretrain_state_dict = state["state_dict"]
        pretrain_args.local_rank = args.local_rank

        attn_model, flow, _state = init_model(pretrain_args)
        if hasattr(attn_model, "module"):
            attn_model = attn_model.module

        pretrain_state_dict = {k.replace("module.", ""): v for k, v in pretrain_state_dict.items()}
        attn_model.load_state_dict(pretrain_state_dict)
        log_rank_0(f"Loaded pretrained state_dict from {checkpoint}")

        flask_list = []
        for line in chunk:
            if ">>" in line:
                ori_reactant = line.strip().split(">>")[0]
                products = line.strip().split(">>")[1].split("|")
                products = [remove_stereo(smi) for smi in products]
            else:
                ori_reactant = line.strip()
                products = []
            flask = init_flask(ori_reactant)
            root_smi = reactant_process(ori_reactant)
            flask_list.append((flask, root_smi, (ori_reactant, products)))

        snapshots_list = run_flask_mc(args, attn_model, flow, flask_list, rng)

        os.makedirs(args.result_path, exist_ok=True)
        all_results = []
        for fi, (flask, root, (reactant, products)) in enumerate(flask_list):
            snapshots = snapshots_list[fi]
            all_results.append((flask, root, (reactant, products), snapshots))
            log_rank_0(
                f"MC Result {fi}: flask has {len(flask)} unique molecules, "
                f"snapshots={len(snapshots)}"
            )

        saving_file = os.path.join(args.result_path, f'result_chunk_{i}_s{seed}.pickle')
        log_rank_0(f"Saving MC results to {saving_file}")
        with open(saving_file, "wb") as f_out:
            pickle.dump(all_results, f_out)
```

- [ ] **Step 4: Smoke-import the file**

```bash
python -c "import mctree_predict; print('ok', [f for f in dir(mctree_predict) if not f.startswith('_')][:20])"
```
Expected: prints `ok` followed by a list including `assemble_packet`, `expand`, `init_flask`, `main`, `run_flask_mc`, `update_flask`. `select`, `beam_search`, and `check_if_successful` should NOT appear.

- [ ] **Step 5: Commit**

```bash
git add mctree_predict.py
git commit -m "mctree_predict: replace beam main() with flask-MC loop; drop dead helpers"
```

---

## Task 8 — End-to-end smoke run

Run the existing driver against a small chunk to confirm the file produces a pickle with the expected new schema.

**Files:**
- (No code changes; verification only.)

- [ ] **Step 1: Inspect the existing driver to know how to invoke**

```bash
cat run_FlowER_large_newData.sh
```
Expected: a shell script that exports the `MODEL_NAME`, `TEST_FILE`, `MODEL_PATH`, `RESULT_PATH`, etc. env vars and then invokes `python mctree_predict.py` (or similar) under `torchrun`. Note the exact invocation line.

- [ ] **Step 2: Run a small smoke test**

Choose `MC_MAX_ITER=2` and `CHUNK_SIZE` set so the run touches at most 2 reactants. Either edit a copy of the driver or invoke directly. Example (adapt env vars to the driver's actual names):

```bash
MC_MAX_ITER=2 MC_PACKET_SIZE=2 MC_SEED=0 \
    sh scripts/eng.sh
```

Or, if running on the head node without SLURM, an inline driver:

```bash
MC_MAX_ITER=2 MC_PACKET_SIZE=2 MC_SEED=0 \
    DATA_NAME=USPTO EMB_DIM=512 SIGMA=0.1 RBF_HIGH=4.0 RBF_GAP=0.1 \
    MODEL_NAME=<actual_ckpt> MODEL_PATH=<actual_path> \
    TEST_FILE=<small_test_file> RESULT_PATH=/tmp/mc_smoke \
    python mctree_predict.py
```

(Substitute the actual values from `run_FlowER_large_newData.sh`. If the smoke run is too heavy for the head node, run it as a short SLURM job via `scripts/eng.sh`.)

- [ ] **Step 3: Verify the output pickle**

```bash
python <<'EOF'
import os, pickle, glob
result_dir = "/tmp/mc_smoke"   # or wherever RESULT_PATH points
files = glob.glob(os.path.join(result_dir, "result_chunk_*.pickle"))
assert files, f"no result pickles in {result_dir}"
print("found:", files)
with open(files[0], "rb") as f:
    results = pickle.load(f)
print("n entries:", len(results))
flask, root, (reactant, products), snapshots = results[0]
print("flask size:", len(flask))
print("root:", root[:60], "...")
print("products:", products)
print("snapshots:", len(snapshots), "iterations")
print("top-5 by weight:", sorted(flask.items(), key=lambda x: -x[1])[:5])

assert len(snapshots) == 2, f"expected 2 snapshots, got {len(snapshots)}"
assert len(flask) >= 1
print("OK")
EOF
```
Expected: `OK` printed at the end; `snapshots` length matches `MC_MAX_ITER=2`; the top-weighted entries include the original reactant fragments.

- [ ] **Step 4: Cleanup scratch verification scripts**

```bash
rm -rf /tmp/mc_verify
```

(Do NOT delete the smoke-run output pickles; they're useful for downstream inspection.)

- [ ] **Step 5: Final commit (no code changes; just to checkpoint)**

If `git status` shows nothing, skip. Otherwise:

```bash
git status
git diff
# only commit if there are intentional changes
```

---

## Self-review notes

**Spec coverage** — every section of the spec has at least one task:

- Initial flask construction → Task 3 (`init_flask`)
- Packet assembly with atom-map renumbering → Task 5 (`assemble_packet`)
- Per-product weight deposit → Task 4 (`update_flask`)
- Iteration loop with `test_batch_size` quota and `packet_owner` tracking → Task 6 (`run_flask_mc`)
- `expand` truncation removal → Task 2
- New `Args` fields (`mc_packet_size`, `mc_max_iter`, `mc_seed`) → Task 1
- `main()` schema change to flask + snapshots pickle → Task 7
- Removal of beam helpers (`select`, `beam_search`, `check_if_successful`) → Task 7
- End-to-end smoke run → Task 8

**Out-of-scope items** (custom weight rules, refactor with `beam_predict.py`, multi-rank sharding) — correctly absent from the plan.

**Type consistency** — `flask_list` shape `(flask_dict, root_smi, (ori_reactant, products))` matches between Task 6 (consumer) and Task 7 (producer); `snapshots_list` is `list[list[dict]]` in both places; `update_flask(flask, smi, weight)` signature matches Task 4 def and Task 6 call.

**Placeholder scan** — no TBDs / TODOs / "appropriate error handling" / undefined symbols.
