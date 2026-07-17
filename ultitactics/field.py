"""Field geometry, coordinate conventions, and the node layout.

Read this file first. Every other module depends on the conventions fixed here,
and most of the subtle bugs in a project like this come from getting them wrong
in one place and right in another.

Coordinate frame
----------------
Origin at the centre of the field.

    x  runs goal-to-goal   (the "long" axis), x in [-50, +50]
    y  runs sideline-to-sideline,             y in [-18.5, +18.5]

This centring is what makes the D_2 reflections in `symmetry.py` clean: a
reflection is just a sign flip on a coordinate, with no offset term.

Two signed quantities carry the tactical frame:

    attack_dir    +1 => the offence is attacking the +x endzone
    force_open_y  +1 => the OPEN side is +y, i.e. the mark is standing on the
                        -y side of the thrower and taking that side away

`force_open_y` deserves a comment, because "force forehand / force backhand" is
the way players say it but is *not* what we encode. Handedness is not modelled,
so forehand/backhand is not recoverable from the geometry; what actually drives
the tactics is which half of the field is open, and that is a pure y-sign fact.
If you later add thrower handedness as a node feature, forehand/backhand becomes
derivable as (force_open_y x handedness).

Node layout
-----------
Fixed 15-node graph, always in this order. Index arithmetic below relies on it:

    0        thrower (handler with the disc)
    1        reset / dump handler
    2..6     stack cutters, stack rank 1..5 (rank 1 = front = closest to disc)
    7        mark (the defender on the thrower)
    8        defender on the reset
    9..13    defenders on cutters rank 1..5
    14       disc

so the defender of offensive player `i` is always node `i + N_OFFENCE`. The
report left the cutter count as "4-5", which does not close: 7v7 with 2 handlers
forces exactly 5 cutters, and a fixed-size graph needs a fixed number anyway.

Dimensions are WFDF (metres). USAU fields are 110x40 yards; if you switch, only
the four constants below change.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

# --- Field dimensions (WFDF, metres) ---------------------------------------

TOTAL_LENGTH = 100.0
ENDZONE_DEPTH = 18.0
PLAYING_PROPER_LENGTH = 64.0  # = TOTAL_LENGTH - 2 * ENDZONE_DEPTH
WIDTH = 37.0

HALF_LENGTH = TOTAL_LENGTH / 2.0  # 50.0
HALF_WIDTH = WIDTH / 2.0  # 18.5

# The attacking goal line sits at x = +GOAL_LINE_X when attack_dir = +1.
GOAL_LINE_X = HALF_LENGTH - ENDZONE_DEPTH  # 32.0

# --- Node layout ------------------------------------------------------------

N_OFFENCE = 7
N_DEFENCE = 7
N_CUTTERS = 5
N_PLAYERS = N_OFFENCE + N_DEFENCE  # 14
N_NODES = N_PLAYERS + 1  # 15, the +1 is the disc

IDX_THROWER = 0
IDX_RESET = 1
IDX_CUTTERS = tuple(range(2, 2 + N_CUTTERS))  # (2, 3, 4, 5, 6)
IDX_MARK = N_OFFENCE + IDX_THROWER  # 7
IDX_DISC = N_PLAYERS  # 14

#: Offensive nodes that can receive a pass (everyone but the thrower).
IDX_RECEIVER_CANDIDATES = (IDX_RESET,) + IDX_CUTTERS  # (1, 2, 3, 4, 5, 6)


def defender_of(offence_idx: int) -> int:
    """Node index of the defender assigned to offensive player `offence_idx`."""
    if not 0 <= offence_idx < N_OFFENCE:
        raise ValueError(f"{offence_idx} is not an offensive node (0..{N_OFFENCE - 1})")
    return offence_idx + N_OFFENCE


class Role(IntEnum):
    """Role of a node. Drives the one-hot block of the node feature vector."""

    THROWER = 0
    RESET = 1
    CUTTER = 2
    MARK = 3
    DEFENDER = 4
    DISC = 5


N_ROLES = len(Role)

#: Role of every node, in node order. Constant across every graph we build.
NODE_ROLES = np.array(
    [Role.THROWER, Role.RESET]
    + [Role.CUTTER] * N_CUTTERS
    + [Role.MARK]
    + [Role.DEFENDER] * (N_DEFENCE - 1)
    + [Role.DISC],
    dtype=np.int64,
)

#: True for the 7 offensive nodes. The disc is not offence.
IS_OFFENCE = np.zeros(N_NODES, dtype=bool)
IS_OFFENCE[:N_OFFENCE] = True

#: Stack rank per node: 1..5 for cutters, 0 ("no rank") otherwise.
STACK_RANK = np.zeros(N_NODES, dtype=np.int64)
STACK_RANK[list(IDX_CUTTERS)] = np.arange(1, N_CUTTERS + 1)

# --- Normalisation ----------------------------------------------------------
#
# TacticAI zero-centres and rescales the pitch onto a 10m x 10m box (Methods,
# p.13). We do the same thing in spirit but map each axis onto [-1, +1]
# independently. Note this is anisotropic: it squashes x ~3x harder than y, so
# normalised distances are not true distances. TacticAI's 110x63 -> 10x10 has
# exactly the same distortion, and it does not appear to hurt -- but every
# geometric quantity in `formation.py` is computed in METRES, before
# normalisation, precisely so the ground-truth rule is not affected by it.


def normalise_pos(pos: np.ndarray) -> np.ndarray:
    """Map field metres -> [-1, 1] per axis. `pos` is [..., 2]."""
    return pos / np.array([HALF_LENGTH, HALF_WIDTH])


def denormalise_pos(pos: np.ndarray) -> np.ndarray:
    """Inverse of `normalise_pos`."""
    return pos * np.array([HALF_LENGTH, HALF_WIDTH])


#: Velocities are normalised by a nominal sprint speed rather than by field size,
#: so that a value of ~1 means "running about as fast as a human does".
NOMINAL_SPEED = 7.5  # m/s

#: Stall count runs 1..10 under WFDF/USAU rules.
MAX_STALL = 10
