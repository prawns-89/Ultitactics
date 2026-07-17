"""A single frozen vertical-stack frame, and the ground-truth completion rule.

This is the "physics" of the synthetic world. The simulator (`simulate.py`)
produces `Frame` objects; the graph builder (`graph.py`) turns them into
tensors; and the label every model is trained against comes from
`completion_logits` below.

Why a hand-written rule at all?
-------------------------------
On real footage the label is the outcome that actually happened. We have no
footage, so we invent a rule, generate frames, label them with it, and then
check whether the GNN can *rediscover* the rule from coordinates alone. If it
can, the whole pipeline (graph construction, symmetry handling, training) is
wired up correctly, and only the rule needs to be swapped for real data later.

So the rule below is deliberately:
  * geometric and explainable (you can read off why a throw is good),
  * a smooth function of position (so gradients through the data make sense),
  * NOT handed to the model as a feature. The model sees raw (x, y, v); it must
    infer "is the lane blocked" itself. `graph.py` intentionally omits any
    lane-visibility feature for this reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import field


@dataclass
class Frame:
    """One frozen instant of a vertical-stack point.

    All positions are in METRES in the centred field frame (see `field.py`),
    shape [N_NODES, 2]. Velocities are in m/s. This is the unnormalised,
    human-readable representation; normalisation happens in `graph.py`.
    """

    pos: np.ndarray  # [15, 2] metres
    vel: np.ndarray  # [15, 2] m/s
    attack_dir: int  # +1 or -1, which endzone the offence attacks
    force_open_y: int  # +1 or -1, which sideline side is "open"
    stall: int  # 1..10
    wind: np.ndarray  # [2] m/s, field frame

    def __post_init__(self) -> None:
        assert self.pos.shape == (field.N_NODES, 2), self.pos.shape
        assert self.vel.shape == (field.N_NODES, 2), self.vel.shape
        assert self.attack_dir in (-1, 1)
        assert self.force_open_y in (-1, 1)
        assert 1 <= self.stall <= field.MAX_STALL
        assert self.wind.shape == (2,)

    # -- convenient views --------------------------------------------------

    @property
    def thrower_pos(self) -> np.ndarray:
        return self.pos[field.IDX_THROWER]

    def offence_pos(self) -> np.ndarray:
        return self.pos[: field.N_OFFENCE]

    def defence_pos(self) -> np.ndarray:
        return self.pos[field.N_OFFENCE : field.N_PLAYERS]


# --- The ground-truth completion rule --------------------------------------
#
# For each receiver candidate we score how good the throw thrower->candidate is,
# as a real-valued logit. Higher = more likely to be completed. Three factors,
# each with a clear tactical reading:
#
#   1. Separation   how open the receiver is from their own defender.
#   2. Lane         whether some defender is standing in the throwing lane.
#   3. Side/gain    downfield progress, rewarded on the open side, taxed on the
#                   break side (the whole point of a "force").
#
# These are combined linearly in logit space and returned per candidate. The
# constants are tuning knobs, gathered here so the rule is easy to inspect and
# perturb.

# Relative weights of the three factors.
W_SEPARATION = 1.4
W_LANE = 2.2
W_SIDE_GAIN = 0.9

# A defender within this perpendicular distance (m) of the throwing lane counts
# as "in the lane"; the block strength falls off smoothly on this scale.
LANE_HALF_WIDTH = 2.0

# Separation (m) at which a receiver is considered comfortably open.
SEPARATION_SCALE = 3.0

# Downfield gain (m) that counts as a "full" gain for the side/gain term.
GAIN_SCALE = 15.0

# Baseline offset so that a typical open throw sits at a sensible probability.
BIAS = -0.3


def _lane_block(thrower: np.ndarray, target: np.ndarray, defenders: np.ndarray) -> float:
    """How blocked is the straight lane thrower->target, in [0, 1]-ish.

    For each defender we find its perpendicular distance to the lane segment and
    whether it sits *between* thrower and target (not behind either end). A
    defender straddling the lane contributes ~1; one far off the lane ~0. We
    take the max over defenders: the single worst blocker defines the risk, the
    way a real thrower sees one poach and pulls the disc down.
    """
    lane = target - thrower
    lane_len = np.linalg.norm(lane)
    if lane_len < 1e-6:
        return 0.0
    lane_unit = lane / lane_len

    rel = defenders - thrower  # [D, 2]
    along = rel @ lane_unit  # projection onto the lane
    perp = np.linalg.norm(rel - np.outer(along, lane_unit), axis=1)

    # Only defenders between the two endpoints can block the flight.
    between = (along > 0.5) & (along < lane_len - 0.5)
    # Smooth falloff with perpendicular distance.
    block = np.exp(-0.5 * (perp / LANE_HALF_WIDTH) ** 2)
    block = np.where(between, block, 0.0)
    return float(block.max()) if block.size else 0.0


def completion_logits(frame: Frame) -> np.ndarray:
    """Ground-truth completion logit for each receiver candidate.

    Returns an array indexed to `field.IDX_RECEIVER_CANDIDATES` (length 6):
    the reset, then cutters rank 1..5. Convert to probabilities with a sigmoid,
    or to a "who catches the next pass" distribution with a softmax.
    """
    thrower = frame.thrower_pos
    defenders = frame.defence_pos()
    logits = []

    for cand in field.IDX_RECEIVER_CANDIDATES:
        target = frame.pos[cand]
        defender = frame.pos[field.defender_of(cand)]

        # 1. Separation from own defender.
        separation = np.linalg.norm(target - defender)
        sep_term = W_SEPARATION * np.tanh(separation / SEPARATION_SCALE)

        # 2. Lane blockage by *any* defender (worst one dominates).
        block = _lane_block(thrower, target, defenders)
        lane_term = -W_LANE * block

        # 3. Downfield gain, valued by which side of the force it attacks.
        gain = (target[0] - thrower[0]) * frame.attack_dir
        on_open_side = (target[1] - thrower[1]) * frame.force_open_y > 0
        side_mult = 1.0 if on_open_side else 0.45  # break side is harder
        side_term = W_SIDE_GAIN * side_mult * np.tanh(gain / GAIN_SCALE)

        logits.append(sep_term + lane_term + side_term + BIAS)

    return np.asarray(logits, dtype=np.float64)


def completion_probs(frame: Frame) -> np.ndarray:
    """Per-candidate probability the throw is completed (independent sigmoids)."""
    return 1.0 / (1.0 + np.exp(-completion_logits(frame)))


def best_receiver(frame: Frame) -> int:
    """Node index of the highest-value receiver candidate (the 'right read')."""
    logits = completion_logits(frame)
    return field.IDX_RECEIVER_CANDIDATES[int(np.argmax(logits))]
