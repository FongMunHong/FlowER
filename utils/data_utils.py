import torch
from torch.utils.data import Dataset
from rdkit import Chem
import numpy as np
import sys
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
from multiprocessing import Pool, cpu_count

np.set_printoptions(threshold=sys.maxsize, linewidth=500)
torch.set_printoptions(profile="full")

ELEM_LIST = ['PAD', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', \
            'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', \
            'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Sr', 'Y', 'Zr', 'Mo', 'Tc', 'Ru', \
            'Rh', 'Pd', 'Ag', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Eu', \
             'Yb', 'Ta', 'W', 'Os', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'V', 'Sm']

MATRIX_PAD = -30
bt_to_electron = {Chem.rdchem.BondType.SINGLE: 2, 
                 Chem.rdchem.BondType.DOUBLE: 4,
                 Chem.rdchem.BondType.TRIPLE: 6,
                 Chem.rdchem.BondType.AROMATIC: 3}

tbl = Chem.GetPeriodicTable()

def count_electrons(map_number:int, smiles: str) -> int:
    """
    Count the electrons on an atom.  This will create a BE matrix of the

    Input: 
        map_number (int) : atom of which you want the electron count
        smiles (str) : atom-mapped smiles.

    """
    pass




def bond_features(bond):
    bt = bond.GetBondType()
    
    return bt_to_electron[bt]

def count_lone_pairs(a):
    v=tbl.GetNOuterElecs(a.GetAtomicNum())
    c=a.GetFormalCharge()
    b=sum([bond.GetBondTypeAsDouble() for bond in a.GetBonds()])
    h=a.GetTotalNumHs()
    return v-c-b-h

ps = Chem.SmilesParserParams()
ps.removeHs = False
ps.sanitize = True
def get_BE_matrix(r):
    rmol = Chem.MolFromSmiles(r, ps)
    Chem.Kekulize(rmol)
    max_natoms = len(rmol.GetAtoms())
    f = np.zeros((max_natoms,max_natoms))
    
    for atom in rmol.GetAtoms():
        lone_pair = count_lone_pairs(atom)
        f[atom.GetIntProp('molAtomMapNumber') - 1, atom.GetIntProp('molAtomMapNumber') - 1] = lone_pair

    for bond in rmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
        a2 = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
        f[(a1,a2)] = f[(a2,a1)] = bond_features(bond)/2 # so that bond electron diff matrix sums up to 0

    return f

electron_to_bo = {val:key for key, val in bt_to_electron.items()}

def get_formal_charge(a, electron):
    v=tbl.GetNOuterElecs(a.GetAtomicNum())
    b=sum([bond.GetBondTypeAsDouble() for bond in a.GetBonds()])
    h=a.GetTotalNumHs()
    f =v -  electron  - b - h
    return f

def mol_prop_compute(matrix):
    """
    vectorized way of computing atom dict and bond dict from matrix
    """
    n = matrix.shape[0]

    # 1) Compute symmetric bond sums once:
    Mplus = matrix + matrix.T

    # 2) Extract all off-diagonal i<j where there's at least one bond
    iu, ju = np.triu_indices(n, k=1)
    vals = Mplus[iu, ju]
    mask = vals != 0

    # 3) Build bond_dict
    bond_dict = {
        (i + 1, j + 1): int(val)
        for i, j, val in zip(iu[mask], ju[mask], vals[mask])
    }
    # 5) Build atom_dict from the diagonal
    diag = matrix.diagonal()
    atom_dict = {
            (i + 1, i + 1): int(diag_val)
        for i, diag_val in enumerate(diag)
    }
    return atom_dict, bond_dict


def BEmatrix_to_mol(rmol, matrix, idxfunc=lambda x:x.GetIdx()):
    atom_dict, bond_dict = mol_prop_compute(matrix)
                
    new_mol = Chem.RWMol(rmol)
    new_mol.UpdatePropertyCache(strict=False)
    
    amap = {}
    for atom in new_mol.GetAtoms():
        amap[atom.GetIntProp('molAtomMapNumber') - 1] = atom.GetIdx()

    for bond in rmol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        new_mol.RemoveBond(a1, a2)
        
    for (a1, a2), electron in bond_dict.items():
        new_mol.AddBond(amap[a1-1], amap[a2-1], electron_to_bo[electron])
        
    for (a1, a1), electron in atom_dict.items():
        a =  new_mol.GetAtomWithIdx(amap[a1-1])
        fc = get_formal_charge(a, electron)
        a.SetFormalCharge(int(fc))
    return new_mol

atom2idx_dict = {elem:i for i, elem in enumerate(ELEM_LIST)}
def smi2vocabid(smi):
    mol = Chem.MolFromSmiles(smi, ps)

    smi_vocab_id_list = np.zeros(len(mol.GetAtoms()))
    for atom in mol.GetAtoms():
        idx = atom2idx_dict[atom.GetSymbol()]
        smi_vocab_id_list[atom.GetIntProp('molAtomMapNumber') - 1] = idx
    return smi_vocab_id_list, len(smi_vocab_id_list)

def process_smiles(smiles):
    src_smi, tgt_smi = smiles.strip().split('|')[0].split('>>')

    error = ""
    try:
        _ = get_BE_matrix(src_smi)
        _ = get_BE_matrix(tgt_smi)
        src_vocab_id_list, src_len = smi2vocabid(src_smi)
        tgt_vocab_id_list, tgt_len = smi2vocabid(tgt_smi)
        assert (src_vocab_id_list == tgt_vocab_id_list).all()
    except Exception as e:
        error = e
        src_smi, tgt_smi = '', ''
        src_vocab_id_list, src_len = [], 0
        tgt_vocab_id_list, tgt_len = [], 0

    # Return a tuple of results for this smiles pair
    return {
        'src_smi': src_smi,
        'tgt_smi': tgt_smi,
        'src_vocab_id_list': src_vocab_id_list,
        'tgt_vocab_id_list': tgt_vocab_id_list,
        'src_len': src_len,
        'tgt_len': tgt_len
        # "error": error
    }

class ReactionBatch:
    def __init__(self,
                 src_data_indices: torch.Tensor,
                 src_token_ids: torch.Tensor,
                 src_lens: torch.Tensor,
                 src_matrices: torch.Tensor,
                 tgt_matrices: torch.Tensor,
                 matrix_masks: torch.Tensor,
                 src_smiles_list: list,
                 tgt_smiles_list: list,
                 ):
        self.src_data_indices = src_data_indices
        self.src_token_ids = src_token_ids
        self.src_lens = src_lens
        self.src_matrices = src_matrices
        self.tgt_matrices = tgt_matrices
        self.matrix_masks = matrix_masks
        self.src_smiles_list = src_smiles_list
        self.tgt_smiles_list = tgt_smiles_list

    def to(self, device):
        self.src_data_indices = self.src_data_indices.to(device)
        self.src_token_ids = self.src_token_ids.to(device)
        self.src_lens = self.src_lens.to(device)
        self.src_matrices = self.src_matrices.to(device)
        self.tgt_matrices = self.tgt_matrices.to(device)
        self.matrix_masks = self.matrix_masks.to(device)

    def pin_memory(self):
        self.src_data_indices = self.src_data_indices.pin_memory()
        self.src_token_ids = self.src_token_ids.pin_memory()
        self.src_lens = self.src_lens.pin_memory()
        self.src_matrices = self.src_matrices.pin_memory()
        self.tgt_matrices = self.tgt_matrices.pin_memory()
        self.matrix_masks = self.matrix_masks.pin_memory()

        return self

class ReactionDataset(Dataset):
    def __init__(self, args, smiles_list, parallel=True, reactant_only=False):
        self.args = args
        self.device = args.device
        self.reactant_only = reactant_only
        self.smiles_list = smiles_list
        self.src_smis = []
        self.tgt_smis = []

        self.src_token_ids = []
        self.tgt_token_ids = []

        self.src_lens = []
        self.tgt_lens = []

        if reactant_only:
            self.parse_reactant_only()
        else:
            if parallel:
                self.parse_data_parallel()
            else:
                self.parse_data()
        
        self.src_lens = np.asarray(self.src_lens)

        self.data_size = len(self.src_smis)
        self.data_indices = np.arange(self.data_size)

    def parse_reactant_only(self):
        for src_smi in self.smiles_list:
            src_smi = src_smi.strip()
            try:
                _ = get_BE_matrix(src_smi)
                src_vocab_id_list, src_len = smi2vocabid(src_smi)
            except Exception as e:
                print(e)
                continue

            self.src_smis.append(src_smi)
            self.src_token_ids.append(src_vocab_id_list)
            self.src_lens.append(src_len)
            self.tgt_lens.append(src_len)

        assert len(self.src_smis) > 0, "Empty Data"

    def parse_data(self):
        for smiles in self.smiles_list:
            src_smi, tgt_smi = smiles.strip().split('|')[0].split('>>')

            try:
                src_matrix = get_BE_matrix(src_smi)
                tgt_matrix = get_BE_matrix(tgt_smi)    
                src_vocab_id_list, src_len = smi2vocabid(src_smi)
                tgt_vocab_id_list, tgt_len = smi2vocabid(tgt_smi)
                assert (src_vocab_id_list == tgt_vocab_id_list).all()
                assert src_len == tgt_len, "src len and tgt len should be the same"
            except Exception as e:
                print(e)
                continue

            self.src_smis.append(src_smi)
            self.tgt_smis.append(tgt_smi)
            self.src_token_ids.append(src_vocab_id_list)
            self.tgt_token_ids.append(tgt_vocab_id_list)
            self.src_lens.append(src_len)
            self.tgt_lens.append(tgt_len)

        assert len(self.src_smis) == len(self.tgt_smis) == len(self.src_lens) == len(self.tgt_lens)  \
              == len(self.tgt_lens) == len(self.src_token_ids) == len(self.tgt_token_ids)

    def parse_data_parallel(self):

        p = Pool(cpu_count())
        results = p.imap(process_smiles, self.smiles_list)

        # Consume results before closing pool to avoid deadlock:
        # workers block on a full pipe if results aren't drained first.
        count = 0
        total = 0
        for result in results:
            total += 1
            if result['src_vocab_id_list'] is [] or result['src_len'] == 0: 
                # print(f"{result['src_smi']}>>{result['tgt_smi']}")
                # print(result['error'])
                count += 1
                continue
            self.src_smis.append(result['src_smi'])
            self.tgt_smis.append(result['tgt_smi'])
            self.src_token_ids.append(result['src_vocab_id_list'])
            self.tgt_token_ids.append(result['tgt_vocab_id_list'])
            self.src_lens.append(result['src_len'])
            self.tgt_lens.append(result['tgt_len'])

        p.close()
        p.join()
        print(f"{count*100/total}% data is unparseable")

    def sort(self):
        self.data_indices = np.argsort(self.src_lens)

    # Added a seed here so shuffle works on distributed workers
    def shuffle_in_bucket(self, bucket_size: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        for i in range(0, self.data_size, bucket_size):
            rng.shuffle(self.data_indices[i:i + bucket_size])

    def batch(self, batch_type: str, batch_size: int, verbose=False):

        self.batch_sizes = []
        if batch_type.startswith("tokens"):
            sample_size = 0
            max_batch_src_len = 0
            max_batch_tgt_len = 0

            for data_idx in self.data_indices:
                src_len = self.src_lens[data_idx]
                tgt_len = self.tgt_lens[data_idx]

                max_batch_src_len = max(src_len, max_batch_src_len)
                max_batch_tgt_len = max(tgt_len, max_batch_tgt_len)

                if batch_type == "tokens" and \
                        max_batch_src_len * (sample_size + 1) <= batch_size:
                    sample_size += 1
                elif batch_type == "tokens_sum" and \
                        (max_batch_src_len + max_batch_tgt_len) * (sample_size + 1) <= batch_size:
                    sample_size += 1
                else:
                    self.batch_sizes.append(sample_size)

                    sample_size = 1
                    max_batch_src_len = src_len
                    max_batch_tgt_len = tgt_len

            # lastly
            self.batch_sizes.append(sample_size)
            self.batch_sizes = np.array(self.batch_sizes)
            assert np.sum(self.batch_sizes) == self.data_size, \
                f"Size mismatch! Data size: {self.data_size}, sum batch sizes: {np.sum(self.batch_sizes)}"

            self.batch_ends = np.cumsum(self.batch_sizes)
            self.batch_starts = np.concatenate([[0], self.batch_ends[:-1]])

        else:
            raise ValueError(f"batch_type {batch_type} not supported!")


    def __len__(self):
        return len(self.batch_sizes)

    def __getitem__(self, idx : int):
        batch_index = idx
        
        batch_start = self.batch_starts[batch_index]
        batch_end = self.batch_ends[batch_index]

        data_indices = self.data_indices[batch_start:batch_end]

        # print(self.src_lens[data_indices], data_indices)
        max_len = max(self.src_lens[data_indices])

        src_token_id_batch = []
        src_len_batch = []
        src_matrix_batch = []
        tgt_matrix_batch = []
        src_smiles_batch = []
        tgt_smiles_batch = []
        for data_index in data_indices:
            # src_token_id, _ = smi2vocabid(self.src_smis[data_index])
            src_token_id = self.src_token_ids[data_index]
            src_len = self.src_lens[data_index]
            src_token_id = np.pad(src_token_id, (0, max_len - src_len),
                                   mode='constant', constant_values=0) # constant value 0 based on 'PAD' in ELEM_LIST
            src_token_id =  torch.as_tensor(src_token_id, dtype=torch.long)

            src_token_id_batch.append(src_token_id)

            src_matrix = get_BE_matrix(self.src_smis[data_index])
            src_matrix = np.pad(src_matrix, ((0, max_len - src_len), (0, max_len - src_len)), 
                       mode='constant', constant_values=MATRIX_PAD)
            src_len_batch.append(src_len)
            src_matrix_batch.append(src_matrix)
            src_smiles_batch.append(self.src_smis[data_index])

            if not self.reactant_only:
                tgt_matrix = get_BE_matrix(self.tgt_smis[data_index])
                tgt_matrix = np.pad(tgt_matrix, ((0, max_len - src_len), (0, max_len - src_len)), 
                        mode='constant', constant_values=MATRIX_PAD)
                tgt_matrix_batch.append(tgt_matrix)
                tgt_smiles_batch.append(self.tgt_smis[data_index])
            
        src_data_indices = torch.as_tensor(data_indices, dtype=torch.long)
        src_len_batch = torch.as_tensor(src_len_batch, dtype=torch.long)
        src_token_id_batch = torch.stack(src_token_id_batch)
        src_matrix_batch = torch.as_tensor(np.stack(src_matrix_batch), dtype=torch.float)
        if not self.reactant_only: 
            tgt_matrix_batch = torch.as_tensor(np.stack(tgt_matrix_batch), dtype=torch.float)
        else: tgt_matrix_batch = src_matrix_batch
        
        node_mask = (src_matrix_batch[:, :, 0] != MATRIX_PAD)
        matrix_masks = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2)).long()

        reaction_batch = ReactionBatch(
            src_data_indices=src_data_indices,
            src_token_ids=src_token_id_batch,
            src_lens=src_len_batch,
            src_matrices=src_matrix_batch,
            tgt_matrices=tgt_matrix_batch,
            matrix_masks=matrix_masks,
            src_smiles_list=src_smiles_batch,
            tgt_smiles_list=tgt_smiles_batch
        )
        
        return reaction_batch
