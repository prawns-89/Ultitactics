"""The GNN: a dense GATv2 with D_2 frame averaging.

This is the Ultimate analogue of TacticAI's core model. We reproduce its two
essential ingredients and deliberately simplify the rest:

  * GATv2 attention over a fully-connected graph (their Eqs. 3-4). Attention
    lets each node weigh its neighbours by learned relevance -- the mechanism
    that lets the model discover, e.g., "the defender sitting in my lane matters
    more than the one behind me".
  * D_2 invariance via FRAME AVERAGING (their Eq. 6): run the same network on
    all four reflected views and average. Guarantees the prediction is identical
    under any reflection, which buys data efficiency under scarcity -- the whole
    reason TacticAI uses geometric deep learning.

Simplifications, each a conscious choice for a readable base implementation:

  * DENSE, not sparse. The graph is 15 nodes and fully connected, so attention
    is a 15x15 matrix. We use plain torch tensors and einsum instead of
    PyTorch Geometric. At this size dense is faster, and far easier to read and
    to reason about than scatter/gather sparse ops. Swapping in PyG later is
    mechanical; the feature layout in `graph.py` already matches theirs.
  * FRAME AVERAGING, not group convolution (their Eq. 8). Frame averaging is
    strictly simpler and gives exact invariance; group convolution is stronger
    but only worth it once there is real data to justify it. Documented as an
    extension.

The model exposes the three TacticAI heads, adapted:
  * receiver:   which candidate catches the next pass   (their receiver head)
  * completion: per-candidate completion probability     (their shot head, re-aimed)
  * shot/goal-level global score is omitted for now; add by pooling H like they
    do in Eq. 10.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import field, graph
from .symmetry import D2

# Candidate receiver node indices, as a tensor for gathering head outputs.
_CAND_IDX = torch.tensor(field.IDX_RECEIVER_CANDIDATES, dtype=torch.long)


class DenseGATv2Layer(nn.Module):
    """One GATv2 message-passing layer over a dense, fully-connected graph.

    Follows TacticAI Eq. 4 (the GATv2 attention coefficient), with edge features
    folded into the attention logit. Multi-head, concatenated across heads.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        # Separate source/target projections -- GATv2's key departure from GAT.
        self.lin_src = nn.Linear(in_dim, heads * out_dim)
        self.lin_dst = nn.Linear(in_dim, heads * out_dim)
        self.lin_edge = nn.Linear(edge_dim, heads * out_dim)
        # Attention vector `a` per head (Eq. 4).
        self.att = nn.Parameter(torch.empty(1, heads, out_dim))
        self.lin_val = nn.Linear(in_dim, heads * out_dim)
        # ROOT / SELF connection. This is the `h_u^(t-1)` argument of phi in
        # TacticAI Eq. 3 -- the node's own previous state, kept separate from the
        # neighbour aggregate. It is NOT optional on a fully-connected graph:
        # every node has the identical neighbourhood {all nodes}, so the only
        # thing distinguishing node i's output is its attention weights. If
        # attention drifts toward uniform, out_i = mean_j(val_j) becomes the same
        # vector for every node, every later layer then sees node-independent
        # input, and the whole encoder collapses to a constant. That collapse is
        # a self-reinforcing fixed point and it is exactly what happens without
        # this term. The root connection guarantees node identity always
        # survives a layer.
        self.lin_root = nn.Linear(in_dim, heads * out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in (
            self.lin_src,
            self.lin_dst,
            self.lin_edge,
            self.lin_val,
            self.lin_root,
        ):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        """x: [B, N, in_dim];  edge: [B, N, N, edge_dim] -> [B, N, heads*out]."""
        B, N, _ = x.shape
        H, D = self.heads, self.out_dim

        src = self.lin_src(x).view(B, N, H, D)  # per source node i
        dst = self.lin_dst(x).view(B, N, H, D)  # per target node j
        val = self.lin_val(x).view(B, N, H, D)
        e = self.lin_edge(edge).view(B, N, N, H, D)

        # GATv2: score(i,j) = a . LeakyReLU(W_src h_i + W_dst h_j + W_e e_ij).
        # i attends over j, so broadcast src on axis j and dst on axis i.
        pre = src[:, :, None] + dst[:, None, :] + e  # [B, N, N, H, D]
        scores = (self.att * F.leaky_relu(pre, 0.2)).sum(-1)  # [B, N, N, H]

        alpha = torch.softmax(scores, dim=2)  # normalise over neighbours j
        # Weighted sum of neighbour values.
        out = torch.einsum("bijh,bjhd->bihd", alpha, val)  # [B, N, H, D]
        # phi(h_u^(t-1), aggregate): add the root term so node identity survives.
        out = out + self.lin_root(x).view(B, N, H, D)
        return out.reshape(B, N, H * D)


class VerticalStackGNN(nn.Module):
    """Full encoder + heads, with D_2 frame averaging wrapped around the encoder."""

    def __init__(self, hidden: int = 128, heads: int = 8, layers: int = 4):
        super().__init__()
        self.encoder = nn.ModuleList()
        in_dim = graph.NODE_DIM
        for _ in range(layers):
            self.encoder.append(
                DenseGATv2Layer(in_dim, hidden // heads, graph.EDGE_DIM, heads)
            )
            in_dim = hidden
        # Heads operate on per-node embeddings gathered at candidate nodes.
        self.completion_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.receiver_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def encode(self, x: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        """Run the raw (single-view) encoder. [B,N,ND],[B,N,N,ED] -> [B,N,hidden]."""
        h = x
        for i, layer in enumerate(self.encoder):
            m = layer(h, edge)
            h = F.elu(m)
        return h

    def encode_invariant(
        self, views_x: torch.Tensor, views_edge: torch.Tensor
    ) -> torch.Tensor:
        """Frame-averaged node embeddings over the 4 D_2 views (TacticAI Eq. 6).

        views_x:    [B, 4, N, ND]      the 4 reflected views per sample
        views_edge: [B, 4, N, N, ED]
        returns:    [B, N, hidden]     invariant node embeddings

        Because a reflection permutes nothing (it only flips coordinate signs),
        node i in every view is the same physical player, so averaging view
        embeddings node-by-node is exactly right and yields exact invariance.
        """
        B, V, N, ND = views_x.shape
        h = self.encode(
            views_x.reshape(B * V, N, ND),
            views_edge.reshape(B * V, N, N, graph.EDGE_DIM),
        )
        return h.view(B, V, N, -1).mean(dim=1)

    def forward(
        self, views_x: torch.Tensor, views_edge: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return completion logits and receiver logits over candidate nodes.

        Both are [B, n_candidates]. `completion` is per-candidate independent
        (sigmoid); `receiver` is a distribution over candidates (softmax).
        """
        h = self.encode_invariant(views_x, views_edge)  # [B, N, hidden]
        cand = h[:, _CAND_IDX, :]  # [B, C, hidden]
        completion = self.completion_head(cand).squeeze(-1)  # [B, C]
        receiver = self.receiver_head(cand).squeeze(-1)  # [B, C]
        return {"completion": completion, "receiver": receiver}


# --- Bridge from numpy Graphs to batched view tensors ----------------------


def stack_views(graphs: list[graph.Graph]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the [B,4,...] view tensors a batch of Graphs feeds to the model.

    Kept here (not in symmetry.py) because it produces torch tensors; symmetry.py
    stays numpy-only so it can be used without torch.
    """
    from .symmetry import all_views

    B = len(graphs)
    vx = np.zeros((B, len(D2), field.N_NODES, graph.NODE_DIM), dtype=np.float32)
    ve = np.zeros(
        (B, len(D2), field.N_NODES, field.N_NODES, graph.EDGE_DIM), dtype=np.float32
    )
    for b, gr in enumerate(graphs):
        for v, view in enumerate(all_views(gr)):
            vx[b, v] = view.x
            ve[b, v] = view.edge
    return torch.from_numpy(vx), torch.from_numpy(ve)
