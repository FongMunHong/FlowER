#!/usr/bin/env python3
"""Conservation metrics for SMILES-output baselines (Graph2SMILES / Molecular Transformer).

Unlike FlowER (which conserves by construction and emits tallies during inference,
see flower_conservation.py), these baselines only output raw SMILES strings, so
conservation must be *recomputed* from each predicted SMILES against the ground-truth
product.

Inputs
------
--gt   : ground-truth test file, one reaction per line, FlowER format:
             <mapped_reaction>|<sequence_idx>          (reactant>>product)
--pred : baseline prediction file, one reaction per line, nbest predictions
         comma-separated, each as <SMILES>_<loglikelihood>, e.g.:
             CCO.CCOC(=O)...[OH-]_-0.069...,CCO...._-3.76...,...   (<= nbest per line)

The two files must have the same number of lines (aligned reaction-by-reaction).

Reports, over the first --nbest predictions per line, the percentage that are:
    validity   - parseable SMILES
    heavy_atom - same heavy-atom composition as the product
    proton(H)  - same all-atom (incl. H) composition as the product
    electron   - same total bond-electron (BE) matrix sum as the product

This script is self-contained (rdkit + numpy only); the BE-matrix helpers are copied
from utils/data_utils.py so it stays lightweight and torch-free.

Usage:
    python g2s_conservation.py --gt <test.txt> --pred <result file> [--nbest 30]
"""

import argparse
import sys
import time
from collections import Counter
from multiprocessing import Pool, cpu_count

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ps = Chem.SmilesParserParams()
ps.removeHs = False
ps.sanitize = True

bt_to_electron = {
    Chem.rdchem.BondType.SINGLE: 2,
    Chem.rdchem.BondType.DATIVE: 2,
    Chem.rdchem.BondType.DOUBLE: 4,
    Chem.rdchem.BondType.TRIPLE: 6,
    Chem.rdchem.BondType.AROMATIC: 3,
}

tbl = Chem.GetPeriodicTable()

NBEST = 30  # set by main(); module-level so Pool workers inherit it


def bond_features(bond):
    return bt_to_electron[bond.GetBondType()]


def count_lone_pairs(a):
    v = tbl.GetNOuterElecs(a.GetAtomicNum())
    c = a.GetFormalCharge()
    b = sum(bond.GetBondTypeAsDouble() for bond in a.GetBonds())
    h = a.GetTotalNumHs()
    return v - c - b - h


def get_BE_matrix(r):
    rmol = Chem.MolFromSmiles(r, ps)
    Chem.Kekulize(rmol)
    max_natoms = len(rmol.GetAtoms())
    f = np.zeros((max_natoms, max_natoms))
    for atom in rmol.GetAtoms():
        idx = atom.GetIntProp("molAtomMapNumber") - 1
        f[idx, idx] = count_lone_pairs(atom)
    for bond in rmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIntProp("molAtomMapNumber") - 1
        a2 = bond.GetEndAtom().GetIntProp("molAtomMapNumber") - 1
        # /2 so the bond-electron difference matrix sums to 0
        f[(a1, a2)] = f[(a2, a1)] = bond_features(bond) / 2
    return f


def atom_count_dict(mol, heavy=True):
    """Sorted (symbol, count) pairs for a molecule's atoms."""
    if heavy:
        counts = Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() != "H")
    else:
        counts = Counter(a.GetSymbol() for a in mol.GetAtoms())
    return sorted(dict(counts).items())


def map_atoms_by_index(smi):
    """SMILES with atom-map numbers set to atom index (+1), Hs made explicit."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smi, ps))
    for atom in mol.GetAtoms():
        atom.SetIntProp("molAtomMapNumber", atom.GetIdx() + 1)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def conservation_count(lines):
    """Return an (<=nbest, 4) 0/1 invalidity matrix for one (gt, pred) line pair."""
    gt_line, pred_line = lines
    rxn = gt_line.strip().split("|")[0]
    reactant, product = rxn.split(">>")
    prod = Chem.MolFromSmiles(product, ps)
    assert prod is not None

    [a.ClearProp("molAtomMapNumber") for a in prod.GetAtoms()]
    prod_smi = Chem.MolToSmiles(prod, isomericSmiles=False)
    prod = Chem.MolFromSmiles(prod_smi, ps)
    prod_smi = Chem.MolToSmiles(prod, isomericSmiles=False)

    heavy_atom_count = atom_count_dict(prod)
    proton_count = atom_count_dict(prod, heavy=False)

    predictions = pred_line.strip().split(",")[:NBEST]
    invalidity = np.zeros((len(predictions), 4))
    for i, entry in enumerate(predictions):
        pred = "_".join(entry.split("_")[:-1])  # strip trailing _loglikelihood
        try:
            pred_mol = Chem.MolFromSmiles(
                Chem.MolToSmiles(Chem.MolFromSmiles(pred, ps), isomericSmiles=True), ps
            )
        except Exception:
            pred_mol = None

        if pred_mol is None:
            invalidity[i, 0:] = 1  # invalid SMILES fails every metric
            continue

        pred_mol = Chem.AddHs(pred_mol)
        if heavy_atom_count != atom_count_dict(pred_mol):
            invalidity[i, 1] = 1
        if proton_count != atom_count_dict(pred_mol, heavy=False):
            invalidity[i, 2] = 1
        gt_be = get_BE_matrix(map_atoms_by_index(prod_smi))
        pred_be = get_BE_matrix(map_atoms_by_index(pred))
        if np.sum(gt_be) != np.sum(pred_be):
            invalidity[i, 3] = 1

    return invalidity


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gt", required=True, help="ground-truth test.txt (reaction|seq_idx)")
    ap.add_argument("--pred", required=True, help="baseline prediction file (smiles_loglik,...)")
    ap.add_argument("--nbest", type=int, default=30, help="predictions per line to score (default: 30)")
    ap.add_argument("--jobs", type=int, default=cpu_count(), help="worker processes (default: all cores)")
    args = ap.parse_args()

    global NBEST
    NBEST = args.nbest


    with open(args.gt) as gt_o, open(args.pred) as result_o:
        gt = gt_o.readlines()
        result = result_o.readlines()

    assert len(gt) == len(result), (
        f"line count mismatch: gt={len(gt)} pred={len(result)} (files must be aligned)"
    )

    print("Loaded data already...")

    n_total = len(result)
    metric = []
    t0 = time.time()
    with Pool(args.jobs) as p:
        for i, inv in enumerate(
            p.imap_unordered(conservation_count, zip(gt, result), chunksize=64), 1
        ):
            metric.append(inv)
            if i % 1000 == 0 or i == n_total:
                rate = i / (time.time() - t0)
                eta = (n_total - i) / rate if rate else 0
                print(
                    f"  processed {i}/{n_total} reactions "
                    f"({100 * i / n_total:.1f}%)  {rate:.0f}/s  ETA {eta:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
    invalidity = np.vstack(metric)

    n = len(invalidity)
    valid_pct = (n - np.sum(invalidity, axis=0)) * 100 / n
    labels = ["validity", "heavy_atom", "proton(H)", "electron"]

    print(f"predictions scored = {n}  (<= {args.nbest} per reaction)")
    for name, pct in zip(labels, valid_pct):
        print(f"  {name:<11} = {pct:.2f}%")


if __name__ == "__main__":
    main()
