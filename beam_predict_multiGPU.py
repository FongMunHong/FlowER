import os
import torch
import numpy as np
from rdkit import Chem
from utils.data_utils import ReactionDataset, BEmatrix_to_mol, ps
import torch.distributed as dist
from train import init_model, init_loader
from utils.train_utils import log_rank_0, setup_logger, log_args
from eval_multiGPU import custom_round
from settings import Args
from collections import defaultdict
import networkx as nx
import pickle
import torch.multiprocessing as mp
import time
from eval_multiGPU import predict_batch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def standardize_smiles(mol):
    return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)

def select(args, frontiers_dict, graph_list):
    filtered_frontiers_dict = {}
    for g_idx, frontiers in frontiers_dict.items():
        graph, root, _ = graph_list[g_idx]
        rank_frontiers = {}
        for frontier in frontiers:
            min_sequences_rank = np.inf
            for path in nx.all_simple_paths(graph, root, frontier):
                max_depth = max(graph.nodes[root]['depth'], len(path))
                graph.nodes[root]['depth'] = max_depth
                edges = list(nx.utils.pairwise(path))
                ranks = [graph.get_edge_data(u, v)['rank'] for u, v in edges]
                probs = [graph.get_edge_data(u, v)['count'] / args.sample_size for u, v in edges]
                cum_prob = np.prod(probs)
                max_topk_within_one_seq = max(ranks)
                min_sequences_rank = min(max_topk_within_one_seq, min_sequences_rank)

            # rank_frontiers[frontier] = min_sequences_rank
            rank_frontiers[frontier] = -cum_prob
        rank_frontiers = sorted(rank_frontiers.items(), key=lambda x:x[1])[:args.beam_size]
        # leftover_frontiers = sorted(rank_frontiers.items(), key=lambda x:x[1])[args.beam_size:]
        # graph.remove_nodes_from([frontier for frontier, prob in leftover_frontiers])

        filtered_frontiers_dict[g_idx] = list(dict(rank_frontiers).keys())
    return filtered_frontiers_dict


def expand(args, model, flow, data_loader):
    sample_size = args.sample_size
    model.eval()

    overall_dict = {}
    with torch.no_grad():
     for batch_idx, data_batch in enumerate(data_loader):
        data_batch.to(args.device)
        src_data_indices = data_batch.src_data_indices
        y = data_batch.src_token_ids
        y_len = data_batch.src_lens
        x0 = data_batch.src_matrices
        matrix_masks = data_batch.matrix_masks
        src_smiles_list = data_batch.src_smiles_list

        batch_size, n, n = x0.shape

        # FIX# missing model.eval() and torch.no_grad() — gradients are being tracked during inference
        if (batch_size*n*n) <= 5*360*360:
            traj_list = predict_batch(args, batch_idx, data_batch, model, flow, 1)
        else:
            traj_list = predict_batch(args, batch_idx, data_batch, model, flow, 2)

        
        last_step = traj_list[-1]
        product_BE_matrices = custom_round(last_step)
        product_BE_matrices_batch = torch.split(product_BE_matrices, sample_size)

        for idx in range(batch_size):
            reac_smi, product_BE_matrices = \
                src_smiles_list[idx], product_BE_matrices_batch[idx]

            reac_mol = Chem.MolFromSmiles(reac_smi, ps)
            matrices, counts = torch.unique(product_BE_matrices, dim=0, return_counts=True)
            matrices, counts = matrices.cpu().numpy(), counts.cpu().numpy()
            
            pred_smis_dict = defaultdict(int)
            for i in range(matrices.shape[0]): # all unique matrices
                pred_prod_be_matrix, count = matrices[i], counts[i] # predicted product matrix and it's count
                num_nodes = y_len[idx]
                pred_prod_be_matrix = pred_prod_be_matrix[:num_nodes, :num_nodes]
                reac_be_matrix = x0[idx][:num_nodes, :num_nodes].detach().cpu().numpy()

                assert pred_prod_be_matrix.shape == reac_be_matrix.shape, "pred and reac not the same shape"
                
                try:
                    pred_mol = BEmatrix_to_mol(reac_mol, pred_prod_be_matrix)
                    pred_smi = standardize_smiles(pred_mol)
                    pred_mol = Chem.MolFromSmiles(pred_smi, ps)
                    pred_smi = standardize_smiles(pred_mol)
                    pred_smis_dict[pred_smi] += count
                except: pass

            pred_smis_tuples = sorted(pred_smis_dict.items(), key=lambda x: x[1], reverse=True)
            
            pred_smis_dict = dict(pred_smis_tuples[:args.nbest])
            overall_dict[reac_smi] = pred_smis_dict

    return overall_dict

def reactant_process(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol, explicitOnly=False)
        for idx, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(idx+1)
        return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)
    except:
        print(smi)
        raise

def clean(smi):
    # try:
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    mol = Chem.RemoveHs(mol)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def remove_stereo(smiles: str) -> str:
    """
    Return a non-isomeric (stereo-stripped) SMILES.
    - Removes chiral tags and E/Z bond stereo.
    - Returns canonical non-isomeric SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

def beam_search(args, model, flow, frontiers_dict, graph_list):
    smiles_list = [frontier for frontiers in frontiers_dict.values() for frontier in frontiers]
    # print('frontiers', smiles_list)
    # print()
    if len(smiles_list) == 0: return
    print(f"Current Depth: {[graph.nodes[root]['depth'] for graph, root, _ in graph_list]}")
    exclude_gidx = [g_idx for g_idx, (graph, root, _) in enumerate(graph_list) 
                    if graph.nodes[root]['depth'] >= args.max_depth]

    test_dataset = ReactionDataset(args, smiles_list, reactant_only=True)
    try:
        test_loader = init_loader(args, test_dataset,
                                batch_size=args.test_batch_size,
                                shuffle=False, epoch=None, use_sort=False)
    except Exception as e:
        print(e)
        return
    
    overall_dict = expand(args, model, flow, test_loader)
    new_frontiers_dict = defaultdict(list)

    existing_reactions = {g_idx: {} for g_idx in frontiers_dict.keys()}
    for g_idx, frontiers in frontiers_dict.items():
        if g_idx in exclude_gidx: continue
        existing_reaction = existing_reactions[g_idx]
        graph, _, _ = graph_list[g_idx]
        for frontier in frontiers:
            clean_frontier = clean(frontier) # ---
            try: product_info_dict = overall_dict[frontier] # given reactant, product info
            except: continue
            for rank, (product, count) in enumerate(product_info_dict.items()):
                try: clean_product = clean(product) # --
                except: continue
                if (clean_frontier, clean_product) in existing_reaction:
                    stored_frontier, stored_product = existing_reaction[(clean_frontier, clean_product)]
                    parent_current = list(graph.predecessors(frontier))
                    parent_stored = list(graph.predecessors(stored_frontier))

                    if parent_current == parent_stored:
                        graph[stored_frontier][stored_product]["count"] += count
                    
                else:
                    if not graph.has_node(product):
                        new_frontiers_dict[g_idx].append(product)
                    graph.add_edge(frontier, product, rank=rank, count=count)
                    existing_reaction[(clean_frontier, clean_product)] = (frontier, product)
                    

    filtered_frontiers_dict = select(args, new_frontiers_dict, graph_list)
    beam_search(args, model, flow, filtered_frontiers_dict, graph_list)


def group_lists(lists, group_size):
    result = []
    # Process lists in chunks of group_size
    for i in range(0, len(lists), group_size):
        # Take a slice of size group_size (or remaining elements if less)
        chunk = lists[i:i + group_size]
        # Convert the chunk to a tuple and add to result
        result.append(tuple(chunk))

    return result

import signal
import os

def init_process_killer():
    global processes
    processes = []
    
    def signal_handler(sig, frame):
        print('\nTerminating all processes...')
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()  # Wait for process to finish
        os._exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    return processes

def worker(rank, args, chunk, chunk_idx, lock, queue):
    """Worker function that runs on each GPU"""
    # Set random seeds for reproducibility
    
    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    args.device = device
    args.local_rank = -1  # Disable distributed training
    
    # Load model for this process
    checkpoint = os.path.join(args.model_path, args.model_name)
    state = torch.load(checkpoint, weights_only=False, map_location=device)
    pretrain_args = state["args"]
    pretrain_args.load_from = None
    pretrain_args.device = device
    pretrain_args.local_rank = -1  # Disable distributed training
    
    # Initialize model without DDP
    pretrain_state_dict = state["state_dict"]
    attn_model, flow, _ = init_model(pretrain_args)
    
    # Remove DDP wrapper if present
    if hasattr(attn_model, "module"):
        attn_model = attn_model.module
    
    pretrain_state_dict = {k.replace("module.", ""): v for k, v in pretrain_state_dict.items()}
    attn_model.load_state_dict(pretrain_state_dict)
    
    # print(f"GPU {rank} starting processing {len(chunk)} items")
    
    # Process chunk
    graph_list = []
    frontiers_dict = defaultdict(list)
    for idx, line in enumerate(chunk):
        if ">>" in line:
            ori_reactant = line.strip().split(">>")[0]
            products = line.strip().split(">>")[1].split("|")
            products = [remove_stereo(smi) for smi in products]
        else:
            ori_reactant = line.strip()
            products = []
        reactant = reactant_process(ori_reactant)
        graph = nx.DiGraph()
        graph.add_node(reactant, depth=1)
        graph_list.append((graph, reactant, (ori_reactant, products)))
        frontiers_dict[idx].append(reactant)
    
    beam_search(args, attn_model, flow, frontiers_dict, graph_list)

    lock.acquire()
    try:
        queue.put((rank, chunk_idx, graph_list))
    finally:
        lock.release()
    
    # print(f"GPU {rank} finished processing")

def check_if_successful(graph, products):
    nodes_with_loops = list(nx.nodes_with_selfloops(graph))
    achieved_products = set()
    for node in graph.nodes():
        node_in_products = set(clean(node).split('.')) & set(products)
        if node_in_products and node in nodes_with_loops:
            achieved_products.update(node_in_products)
    return achieved_products

def main_multi_gpu(args):
    start = time.time()
    global processes
    processes = init_process_killer()

    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    log_rank_0(f"Found {world_size} GPUs")
    
    # Read all test smiles
    with open(args.test_path, 'r') as test_o:
        test_smiles_list = test_o.readlines()
    
    # Calculate chunk size and create chunks
    # chunk_size = math.ceil(len(test_smiles_list) / world_size)
    chunk_size = args.chunk_size // world_size
    chunks = [test_smiles_list[i:i + chunk_size] for i in range(0, len(test_smiles_list), chunk_size)]

    group_chunks = group_lists(chunks, world_size)
    log_rank_0(f"Number of group chunks: {len(group_chunks)}")

    os.makedirs(args.result_path, exist_ok=True)
    # Start processes
    lock = mp.Lock()
    q = mp.Queue()
    chunk_idx = 0
    for group_chunk_id, group_chunk in enumerate(group_chunks):        
        log_rank_0(f"Group Chunk-{group_chunk_id} called:")
        all_results = []
        processes = []
        for gpu_idx, chunk in enumerate(group_chunk):
            p = mp.Process(target=worker, args=(gpu_idx, args, chunk, chunk_idx, lock, q))
            p.start()
            processes.append(p)
            time.sleep(1)  # Add small delay between process starts
            chunk_idx += 1

        outputs = []
        for _ in processes:
            output = q.get(timeout=1000)
            if output is None: continue
            outputs.append(output)
        outputs  = sorted(outputs, key=lambda x:x[0])
        for output in outputs:
            _, output_chunk_idx, graph_list = output
            for beam_idx, (graph, root, (reactant, products)) in enumerate(graph_list):
                check = check_if_successful(graph, products)
                log_rank_0(f"Beam Search Results {beam_idx}: {len(check)}/{len(products)} - {check}")
                all_results.append((graph, root, (reactant, products), check))

        # Wait for all processes to complete
        for p in processes:
            p.join()

        saving_file = os.path.join(args.result_path, f'result_chunk_{group_chunk_id}.pickle')
        with open(saving_file, "wb") as f_out:
            pickle.dump(all_results, f_out)

        log_rank_0(f"---- Time used: {(time.time() - start):.2f}s ----")


    log_rank_0("Done!")

if __name__ == "__main__":
    # Ensure clean startup
    mp.set_start_method('spawn')
    
    args = Args
    logger = setup_logger(args, "beam")
    log_args(args, 'evaluation')
    main_multi_gpu(args)