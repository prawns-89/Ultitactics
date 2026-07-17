"""Turn a `Frame` into the tensors a GNN consumes.

This is the Ultimate analogue of TacticAI Table 2 (the feature summary). We
follow TacticAI's two load-bearing choices deliberately:

  * Fully-connected graph. Every pair of nodes is connected; the edge feature
    carries the *relationship*, not the topology. TacticAI sets E = V x V and
    uses a single one-hot teammate/opponent flag (Methods, p.13). We keep the
    dense graph and enrich the edge features (below). Typed *edge topology*
    (separate stack-adjacency / marking / lane edge sets) is a documented
    upgrade path, not the base -- it complicates the symmetry handling and is
    not needed to reproduce the paper's result.

  * One shared node feature schema for every node. TacticAI's 22 players all
    share a schema; roles are distinguished by features, not by separate tensor
    shapes. We extend this to 15 nodes (14 players + disc) with a role one-hot
    and zero-fill for role-specific slots. This resolves the report's internal
    contradiction (it claimed a shared schema but then tabled per-role
    features).

Departures from TacticAI, all justified by the sport rather than by "porting":

  * A disc node. A corner-kick ball is barely modelled (one possession bit);
    an in-flight disc is a real object with its own trajectory, so it earns a
    node. It carries a z-height slot the players leave at zero.
  * Stack rank as a node feature. Football players have no canonical order;
    vertical-stack cutters do. Rank is a *value* in a feature slot, so it does
    NOT break the permutation-equivariance of message passing -- the report's
    "a plain GNN port isn't right" framing was too strong on this point.
  * A signed force feature and a wind vector, both of which the D_2 group action
    in `symmetry.py` must transform in lockstep with coordinates.

Node feature layout (NODE_DIM columns), same for every node
-----------------------------------------------------------
    0,1     position x, y            (normalised to [-1, 1] per axis)
    2,3     velocity x, y            (normalised by nominal sprint speed)
    4       z-height                 (disc only; 0 for players)
    5       is-offence               (1 / 0)
    6       stack rank               (1..5 for cutters, 0 otherwise; /5)
    7       stall count              (1..10, /10; carried on every node)
    8       force_open_y             (+1 / -1; the tactical frame, see field.py)
    9..14   role one-hot             (6 roles, see field.Role)

Edge feature layout (EDGE_DIM columns), for ordered pair (i -> j)
-----------------------------------------------------------------
    0       same-team    (1 if i, j both offence or both defence; else 0)
    1       is-marking   (1 if i, j are an assigned offence/defender pair)
    2,3     displacement (pos_j - pos_i), normalised
    4       distance     (normalised)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import field
from .formation import Frame

# Feature-block boundaries, named so downstream code (and symmetry.py) never
# hard-codes column indices.
NODE_POS = slice(0, 2)
NODE_VEL = slice(2, 4)
NODE_Z = 4
NODE_IS_OFF = 5
NODE_RANK = 6
NODE_STALL = 7
NODE_FORCE = 8
NODE_ROLE = slice(9, 9 + field.N_ROLES)
NODE_DIM = 9 + field.N_ROLES  # 15

EDGE_SAME_TEAM = 0
EDGE_IS_MARK = 1
EDGE_DISP = slice(2, 4)
EDGE_DIST = 4
EDGE_DIM = 5


@dataclass
class Graph:
    """Dense graph tensors for one frame. Batch-free; batching is a stack."""

    x: np.ndarray  # [N_NODES, NODE_DIM]      node features
    edge: np.ndarray  # [N_NODES, N_NODES, EDGE_DIM] dense edge features
    meta: dict  # provenance: attack_dir, force_open_y, stall, wind

    def __post_init__(self) -> None:
        assert self.x.shape == (field.N_NODES, NODE_DIM), self.x.shape
        assert self.edge.shape == (field.N_NODES, field.N_NODES, EDGE_DIM)


# Precomputed same-team mask over the 14 players (disc excluded). Constant.
def _same_team_matrix() -> np.ndarray:
    m = np.zeros((field.N_NODES, field.N_NODES), dtype=np.float64)
    off = field.IS_OFFENCE
    for i in range(field.N_NODES):
        for j in range(field.N_NODES):
            if i == field.IDX_DISC or j == field.IDX_DISC:
                continue  # disc is on no team
            m[i, j] = 1.0 if off[i] == off[j] else 0.0
    return m


def _marking_matrix() -> np.ndarray:
    m = np.zeros((field.N_NODES, field.N_NODES), dtype=np.float64)
    for off_idx in range(field.N_OFFENCE):
        d = field.defender_of(off_idx)
        m[off_idx, d] = m[d, off_idx] = 1.0
    return m


_SAME_TEAM = _same_team_matrix()
_MARKING = _marking_matrix()


def build_graph(frame: Frame) -> Graph:
    """Featurise one frame into dense node/edge tensors."""
    x = np.zeros((field.N_NODES, NODE_DIM), dtype=np.float64)

    pos_n = field.normalise_pos(frame.pos)
    x[:, NODE_POS] = pos_n
    x[:, NODE_VEL] = frame.vel / field.NOMINAL_SPEED
    # z-height only meaningful for the disc; 0 everywhere here (grounded disc).
    x[field.IDX_DISC, NODE_Z] = 0.0
    x[:, NODE_IS_OFF] = field.IS_OFFENCE.astype(np.float64)
    x[:, NODE_RANK] = field.STACK_RANK / field.N_CUTTERS
    x[:, NODE_STALL] = frame.stall / field.MAX_STALL
    x[:, NODE_FORCE] = frame.force_open_y
    x[np.arange(field.N_NODES), NODE_ROLE.start + field.NODE_ROLES] = 1.0

    # -- Edges ------------------------------------------------------------
    edge = np.zeros((field.N_NODES, field.N_NODES, EDGE_DIM), dtype=np.float64)
    edge[:, :, EDGE_SAME_TEAM] = _SAME_TEAM
    edge[:, :, EDGE_IS_MARK] = _MARKING
    # Displacement in normalised coords, so it lives on the same scale as
    # positions and transforms identically under reflections.
    disp = pos_n[None, :, :] - pos_n[:, None, :]  # [i, j] = pos_j - pos_i
    edge[:, :, EDGE_DISP] = disp
    edge[:, :, EDGE_DIST] = np.linalg.norm(disp, axis=2)

    meta = {
        "attack_dir": frame.attack_dir,
        "force_open_y": frame.force_open_y,
        "stall": frame.stall,
        "wind": frame.wind.copy(),
    }
    return Graph(x=x, edge=edge, meta=meta)
