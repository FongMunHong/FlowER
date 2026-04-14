import math
import torch
import torch.nn as nn
from utils.attn_utils import PositionwiseFeedForward
from utils.attn_utils import sequence_mask
from utils.data_utils import ELEM_LIST
from model.flow_matching import zero_center_func

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def zero_center_output(x_batch, ori_node_mask_batch):
    x_batch = x_batch.masked_fill(ori_node_mask_batch, 1e-19)
    node_mask_batch = (~ori_node_mask_batch).long()
    map_zero_center = torch.vmap(zero_center_func)
    return map_zero_center(x_batch, node_mask_batch).masked_fill(~(node_mask_batch.bool()), 1e-19)

class RBFExpansion(nn.Module):
    def __init__(self, args):
        """
        Adapted from Schnet.
        https://github.com/atomistic-machine-learning/SchNet/blob/master/src/schnet/nn/layers/rbf.py
        """
        super().__init__()
        self.args = args
        self.device = args.device
        self.low = args.rbf_low
        self.high = args.rbf_high
        self.gap = args.rbf_gap

        self.xrange = self.high - self.low
        
        self.centers = torch.linspace(self.low, self.high, 
            int(torch.ceil(torch.tensor(self.xrange / self.gap)))).to(self.device)
        self.dim = len(self.centers)

    def forward(self, matrix, matrix_mask):
        matrix = matrix.masked_fill(matrix_mask, 1e9)
        matrix = matrix.unsqueeze(-1)  # Add a new dimension at the end
        # Compute the RBF
        matrix = matrix - self.centers
        rbf = torch.exp(-(matrix ** 2) / self.gap)
        return rbf

class MultiHeadedRelAttention(nn.Module):
    def __init__(self, args, head_count, model_dim, dropout, u, v):
        super().__init__()
        self.args = args

        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, model_dim)
        self.linear_values = nn.Linear(model_dim, model_dim)
        self.linear_query = nn.Linear(model_dim, model_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.u = u if u is not None else \
            nn.Parameter(torch.randn(self.d_model), requires_grad=True)
        self.v = v if v is not None else \
            nn.Parameter(torch.randn(self.d_model), requires_grad=True)

    def forward(self, inputs, mask, rel_emb):
        """
        Compute the context vector and the attention vectors.

        Args:
           inputs (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
           rel_emb: graph distance matrix (BUCKETED), ``(batch, key_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """

        batch_size = inputs.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query. Seems that we don't need layer_cache here
        query = self.linear_query(inputs)
        key = self.linear_keys(inputs)
        value = self.linear_values(inputs)

        key = shape(key)                # (b, t_k, h) -> (b, head, t_k, h/head)
        value = shape(value)
        query = shape(query)            # (b, t_q, h) -> (b, head, t_q, h/head)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)

        if rel_emb is None:
            scores = torch.matmul(
                query, key.transpose(2, 3))                 # (b, head, t_q, t_k)
            # scores = scores + rel_emb.unsqueeze(1)
        else:
            # a + c
            u = self.u.reshape(1, head_count, 1, dim_per_head)
            a_c = torch.matmul(query + u, key.transpose(2, 3))

            # rel_emb = self.relative_pe(rel_emb)           # (b, t_q, t_k) -> (b, t_q, t_k, h)
            rel_emb = rel_emb.reshape(                      # (b, t_q, t_k, h) -> (b, t_q, t_k, head, h/head)
                batch_size, query_len, key_len, head_count, dim_per_head)

            # b + d
            query = query.unsqueeze(-2)                     # (b, head, t_q, h/head) -> (b, head, t_q, 1, h/head)
            rel_emb_t = rel_emb.permute(0, 3, 1, 4, 2)      # (b, t_q, t_k, head, h/head) -> (b, head, t_q, h/head, t_k)

            v = self.v.reshape(1, head_count, 1, 1, dim_per_head)
            b_d = torch.matmul(query + v, rel_emb_t
                               ).squeeze(-2)                # (b, head, t_q, 1, t_k) -> (b, head, t_q, t_k)

            scores = a_c + b_d

        scores = scores.float()

        mask = mask.unsqueeze(1)                            # (B, 1, 1, T_values)
        scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)   # -> (b, head, t_q, h/head)
        context = unshape(context_original)                 # -> (b, t_q, h)

        output = self.final_linear(context)
        attns = attn.view(batch_size, head_count, query_len, key_len)

        return output, attns


class SALayerXL(nn.Module):
    """
    A single layer of the self-attention encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout: dropout probability(0-1.0).
    """

    def __init__(self, args, d_model, heads, d_ff, dropout, attention_dropout, u, v):
        super().__init__()

        self.self_attn = MultiHeadedRelAttention(
            args,
            heads, d_model, dropout=attention_dropout,
            u=u,
            v=v
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask, rel_emb):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``
            rel_emb (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        normed_inputs = self.layer_norm(inputs)
        context, _ = self.self_attn(normed_inputs, mask=mask, rel_emb=rel_emb)
        out = self.dropout(context) + inputs
        return self.feed_forward(self.layer_norm_2(out)) + out

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(size, eps=1e-6)

    def forward(self, x: torch.Tensor):
        return x + self.layer_norm(self.act(self.ff(x)))

class AttnEncoderXL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.num_layers = args.enc_num_layers
        self.post_processing_layers = args.post_processing_layers
        self.d_model = args.emb_dim
        self.heads = args.enc_heads
        self.d_ff = args.enc_filter_size
        self.attention_dropout = args.attn_dropout

        self.atom_embedding = nn.Embedding(len(ELEM_LIST) , self.d_model, padding_idx=0)
        self.rbf = RBFExpansion(args)

        self.time_dim = self.d_model - self.rbf.dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.dropout = nn.Dropout(p=args.dropout)
        if args.rel_pos in ["enc_only", "emb_only"]:
            self.u = nn.Parameter(torch.randn(self.d_model), requires_grad=True)
            self.v = nn.Parameter(torch.randn(self.d_model), requires_grad=True)
        else:
            self.u = None
            self.v = None

        if args.shared_attention_layer == 1:
            self.attention_layer = SALayerXL(
                args, self.d_model, self.heads, self.d_ff, args.dropout, self.attention_dropout,
                self.u, self.v)
        else:
            self.attention_layers = nn.ModuleList(
                [SALayerXL(
                    args, self.d_model, self.heads, self.d_ff, args.dropout, self.attention_dropout,
                    self.u, self.v)
                 for i in range(self.num_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)
        
        self.query_w = torch.nn.Sequential(
            *[Block(self.d_model) for _ in range(self.post_processing_layers)]
        )
        self.key_w = torch.nn.Sequential(
            *[Block(self.d_model) for _ in range(self.post_processing_layers)]
        )

        self.query_diag_w = torch.nn.Sequential(
            *[Block(self.d_model) for _ in range(self.post_processing_layers)]
        )
        self.key_diag_w = torch.nn.Sequential(
            *[Block(self.d_model) for _ in range(self.post_processing_layers)]
        )
        self.value_diag_w = torch.nn.Sequential(
            *[Block(self.d_model) for _ in range(self.post_processing_layers)]
        )
        self.final_diag_w = torch.nn.Linear(self.d_model, 1)

        self.rel_emb_w = torch.nn.Sequential(
            *[*[Block(self.d_model) for _ in range(self.post_processing_layers)], 
            torch.nn.Linear(self.d_model, 1)]
        )
        self.softmax = nn.Softmax(dim=-1)

        rbf_layers = []
        # rbf_layers.append(torch.nn.Linear(self.rbf.dim+1, self.rbf.dim))
        for _ in range(self.post_processing_layers):
            rbf_layers.append(Block(self.rbf.dim))
        self.rbf_linear = torch.nn.Sequential(*rbf_layers)
        # self.rbf_final_linear = torch.nn.Linear(self.rbf.dim, 1)

    def id2emb(self, src_token_id):
        return self.atom_embedding(src_token_id)
        
    def forward(self, src_token_ids, lengths, bond_matrix, timestep):
        """adapt from onmt TransformerEncoder
            src_token_id: (b, t, h)
            lengths: (b,)

            NEW on Jan'23: return: (b, t, h)
        """
        src = self.id2emb(src_token_ids)
        if timestep.dim() == 0:
            timestep = timestep.repeat(lengths.shape[0])

        b, n, _ = bond_matrix.shape
        timestep = self.time_embed(timestep_embedding(timestep, self.time_dim))
        timestep = timestep.unsqueeze(1).unsqueeze(1) # unsqueeze to match bond n x n
        timestep = timestep.repeat(1, n, n, 1) # unsqueeze to match bond n x n

        mask = ~sequence_mask(lengths).unsqueeze(1)
        
        matrix_masks = ~(~mask * ~mask.transpose(1, 2)).bool()
        rbf_bond_matrix = self.rbf(bond_matrix, matrix_masks)
        rbf_bond_matrix = self.rbf_linear(rbf_bond_matrix)           # b, n, n, 1 -> b, n, n, rbf-dim
        
        rel_emb = torch.cat((rbf_bond_matrix, timestep), dim=-1) # b, n, n, d

        # src = self.atom_embedding(src_token_id)
        # h_place = (src_token_id == 1).float().unsqueeze(-1).repeat(1, 1, src.shape[-1])

        b, n, d = src.shape 
 
        # a_i - raw atom embeddings
        a_i = src * math.sqrt(self.d_model)
        a_i = self.dropout(a_i)

        if self.args.shared_attention_layer == 1:
            layer = self.attention_layer
            for i in range(self.num_layers):
                a_i = layer(a_i, mask, rel_emb)
        else:
            for layer in self.attention_layers:
                a_i = layer(a_i, mask, rel_emb)
        a_i = self.layer_norm(a_i)                        # b,n,d
        
        # a_i - atom embeddings after multiheaded attention on atom embeddings + rbf expansion

        # diagonal prediction
        query_diag = self.query_diag_w(a_i)             # b,n,d @ d,d -> b,n,d
        key_diag = self.key_diag_w(a_i)                 # b,n,d @ d,d -> b,n,d
        value_diag = self.value_diag_w(a_i)             # b,n,d @ d,d -> b,n,d

        diag_scores = torch.matmul(query_diag, key_diag.transpose(1, 2)) # b,n,d @ b,d,n -> b,n,n
        diag_scores = diag_scores.masked_fill(matrix_masks, 1e-9)
        diag_scores = self.softmax(diag_scores) / math.sqrt(self.d_model)
        context = torch.matmul(diag_scores, value_diag)    # b,n,n @ b,n,d -> b,n,d
        diag = self.final_diag_w(context).view(b, n)       # b,n,d @ d,1 -> b,n,1 -> b,n
        
        # non diagonal prediction
        query = self.query_w(a_i)                          # b,n,d @ d,d -> b,n,d
        key = self.key_w(a_i) 

        scores = torch.matmul(query, key.transpose(1, 2))  # b,n,d @ b,d,n -> b,n,n
        a_ij = scores / math.sqrt(self.d_model)

        rbfw_ij = self.rel_emb_w(rel_emb).view(b, n, n)   # b,n,n,d @ d,1 -> b,n,n,1 -> b,n,n
        out = a_ij + rbfw_ij

        for i in range(b):
            indices = torch.arange(n)
            out[i, indices, indices] = 0
            out[i].diagonal().add_(diag[i])

        out = zero_center_output(out, matrix_masks)
        out = (out + out.transpose(1, 2))

        return out
