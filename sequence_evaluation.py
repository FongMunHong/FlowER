
import networkx as nx
import numpy as np
from rdkit import Chem
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 


BAD = 1000

def assign_rank(predictions):
    sorted_items = sorted(predictions, key=lambda x: x[1], reverse=True)

    rank_list = []
    current_rank = 0
    prev_count = None

    for i, (smi, count, _) in enumerate(sorted_items):
        # If the current value is the same as the previous one, assign the same rank
        if prev_count == count:
            rank_list.append((current_rank, smi, count, _))
        else:
            # Otherwise, assign a new rank
            current_rank = i
            rank_list.append((current_rank, smi, count, _))

        prev_count = count

    return rank_list

def clean(smi):
    # try:
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    mol = Chem.RemoveHs(mol)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def process_topk_acc_n_seq_rank(line):
    sequence_idx, sequence_preds = line
    seq_graph_gt = nx.DiGraph()
        
    new_sequence_edges = {}
    gt_neigh = defaultdict(set)
    for pred_info in sequence_preds:
        rxn = pred_info["rxn"]
        reactant, product = rxn.strip().split('>>')
        seq_graph_gt.add_edge(reactant, product)
        new_sequence_edges[(reactant, product)] = np.inf
        gt_neigh[reactant].add(product)
        
    
    starting_reac = [node for node, in_degree in seq_graph_gt.in_degree() if in_degree == 0]
    terminal_prods = list(nx.nodes_with_selfloops(seq_graph_gt))

    if len(starting_reac) != 1: return BAD, 1, sequence_idx    # if starting reactant is not 1
    starting_reac = starting_reac[0]

    if len(terminal_prods) == 0:  return BAD, 2, sequence_idx  # if we have a loop

    # merge predictions
    for pred_info in sequence_preds:
        reactant, _ = pred_info["rxn"].strip().split('>>')
        predictions = pred_info["predictions"]
        predictions = assign_rank(predictions)
        for rank, pred, pred_count, _ in predictions:
            if pred in gt_neigh[reactant]:
                cur_rank = new_sequence_edges.get((reactant, pred))
                if cur_rank == np.inf: # and it's the first time
                    new_sequence_edges[(reactant, pred)] = rank

    seq_graph_pred = nx.DiGraph()
    for (reac, prod), rank in new_sequence_edges.items():
        seq_graph_pred.add_edge(reac, prod, weight=rank)

    max_depth = 0
    min_sequences_rank = np.inf
    for terminal in terminal_prods:
        for path in nx.all_simple_paths(seq_graph_pred, source=starting_reac, target=terminal):
            max_depth = max(len(path), max_depth)
            edges = nx.utils.pairwise(path)
            ranks = [seq_graph_pred.get_edge_data(u, v)['weight'] for u, v in edges]
            max_topk_within_one_seq = max(ranks)
            min_sequences_rank = min(max_topk_within_one_seq, min_sequences_rank)

    terminal_prods = [clean(prod) for prod in terminal_prods]
    return min_sequences_rank, 0, sequence_idx, (clean(starting_reac), terminal_prods), max_depth # min of all sequences


def remove_atom_map_rxn(line):
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    ps.sanitize = True
    try:
        rxn, sequence_idx = line.strip().split("|")
    except:
        rxn, rxn_class, condition, elem_step, sequence_idx = line.strip().split("|")

    reactant, product = rxn.split(">>")
    reac = Chem.MolFromSmiles(reactant, ps)
    prod = Chem.MolFromSmiles(product, ps)
    
    assert reac is not None
    assert prod is not None
    
    [a.ClearProp('molAtomMapNumber') for a in reac.GetAtoms()]
    [a.ClearProp('molAtomMapNumber') for a in prod.GetAtoms()]

    reac_smi = Chem.MolToSmiles(reac, isomericSmiles=False)
    prod_smi = Chem.MolToSmiles(prod, isomericSmiles=False)

    reac = Chem.MolFromSmiles(reac_smi, ps)
    prod = Chem.MolFromSmiles(prod_smi, ps)
    reac_smi = Chem.MolToSmiles(reac, isomericSmiles=False)
    prod_smi = Chem.MolToSmiles(prod, isomericSmiles=False)

    rxn = f"{reac_smi}>>{prod_smi}|{sequence_idx}"

    return rxn

def reparse(line):
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    ps.sanitize = True
    metrics, not_sym, predictions = line.strip().split("|")
    predictions =  eval(predictions)
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    # new_predictions = []
    pred_dict = defaultdict(int)
    for (pred, pred_count, val) in predictions:
        pred_mol = Chem.MolFromSmiles(pred, ps)
        if pred_mol is None: continue
        pred_smi = Chem.MolToSmiles(pred_mol, isomericSmiles=False)
        # new_predictions.append((pred_smi, pred_count, val))
        pred_dict[pred_smi] += pred_count

    pred_dict = dict(sorted(pred_dict.items(), key=lambda x: x[1], reverse=True))
    new_predictions = [(pred_smi, prob, True) for pred_smi, prob in pred_dict.items()]

    return f"{metrics}|{not_sym}|{new_predictions}"


with open("data/flower_new_dataset/test.txt") as gt_o, \
    open("results/flower_new_dataset/best_large_hyperparam/result-32-2940000_97.txt") as result_o:

    # Preprocessing
    result = result_o.readlines()
    gt = gt_o.readlines()

    assert len(gt) == len(result)

    print("Ground Truth lines")
    p = Pool(cpu_count())
    gt = p.imap(remove_atom_map_rxn, (rxn for rxn in gt))
    gt = list(gt)

    print("Prediction lines")
    result = p.imap(reparse, (res for res in result))
    result = list(result)

    nbest = 10
    topk_accs = np.zeros([len(gt), nbest], dtype=np.float32)

    invalid = [] 
    bag_of_vals = defaultdict(list)
    reac_prod_rank = {}
    for i, (line_res, line_gt) in enumerate(zip(result, gt)):
        metrics, not_sym, predictions = line_res.strip().split("|")
        metrics, predictions = eval(metrics), eval(predictions)

        invalid.append(metrics[3] / sum(metrics))
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

        rxn, sequence_idx = line_gt.strip().split("|")
        reactant, product = rxn.split(">>")
        if reactant in reac_prod_rank:
            extract_rank = reac_prod_rank[(reactant, product)]
            topk_accs[i, extract_rank:] = 1
        else:
            for rank, (pred, pred_count, _) in enumerate(predictions):
                if pred == product:
                    topk_accs[i, rank:] = 1
                    reac_prod_rank[(reactant, product)] = rank
                    break

        if sequence_idx in ['PM', 'RS', 'RC', 'PC']: continue
        bag_of_vals[sequence_idx].append(
            {
                "rxn": rxn,
                "metrics": metrics,
                "predictions": predictions
            }
        )
    avg_invalid = sum(invalid) / len(invalid)
    print(f"Valid percentage: {((1 - avg_invalid) * 100): .2f}%")

    
    print("Calculating Topk Step Accuracy")
    mean_seq_accuracies = np.mean(topk_accs, axis=0)
    for n in range(nbest):
        line = f"Top {n+1} step accuracy: {mean_seq_accuracies[n] * 100: .2f} %"
        print(line)

    sequence_accs = np.zeros([len(bag_of_vals), nbest], dtype=np.float32)

    print("Calculating Pathway Accuracy")

    no_starting_point = 0
    no_starting_point_set = set()
    count_no_terminal = 0
    count_no_terminal_set = set()

    seq_ranks = p.imap(process_topk_acc_n_seq_rank, ((seq_idx, seq_infos)  for seq_idx, seq_infos in bag_of_vals.items()))
    for i, (rank, error, seq_idx, (reactant, prod_list), max_depth) in enumerate(seq_ranks):
        if error == 1:
            no_starting_point += 1
            no_starting_point_set.add(seq_idx)
        if error == 2:
            count_no_terminal += 1
            count_no_terminal_set.add(seq_idx)
        if rank >= nbest: continue
        sequence_accs[i, rank:] = 1

    p.close()
    p.join()

    # print('no_starting_point', no_starting_point)
    # print('no_starting_point_set', no_starting_point_set)
    # print('count_no_terminal', count_no_terminal)
    # print('count_no_terminal_set', count_no_terminal_set)

    mean_seq_accuracies = np.mean(sequence_accs, axis=0)
    for n in range(nbest):
        line = f"Top {n+1} pathway accuracy: {mean_seq_accuracies[n] * 100: .2f} %"
        print(line)
