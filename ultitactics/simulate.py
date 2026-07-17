"""Synthetic vertical-stack frame generator.

Produces `Frame` objects that look roughly like a real vertical stack: two
handlers behind the disc, five cutters in a single file down the middle of the
field offset to the break side, and a defender on each with man-defence
positioning plus noise. There is no simulation of *motion over time* here --
each frame is an independent frozen instant, which is exactly what TacticAI's
model consumes (the single frame when the corner is taken).

The randomness is what gives the model something to learn: cutters sit at
varied depths, defenders play with varying tightness and on varying sides, so
"which throw is on" genuinely varies frame to frame, and the ground-truth rule
in `formation.py` responds to it.

Everything is in METRES in the centred field frame. Pass a seeded
`np.random.Generator` for reproducibility.
"""

from __future__ import annotations

import numpy as np

from . import field
from .formation import Frame


def _stack_axis_offset(force_open_y: int) -> float:
    """Lateral offset (m) of the stack from the field centre.

    The stack sits toward the BREAK side, deliberately leaving the open side
    clear for cutters to attack into. Break side is the -force_open_y side.
    """
    return -force_open_y * 4.0


def sample_frame(rng: np.random.Generator) -> Frame:
    """Draw one random vertical-stack frame."""
    attack_dir = int(rng.choice((-1, 1)))
    force_open_y = int(rng.choice((-1, 1)))
    stall = int(rng.integers(1, field.MAX_STALL + 1))

    pos = np.zeros((field.N_NODES, 2))
    vel = np.zeros((field.N_NODES, 2))

    # -- Thrower: somewhere in the attacking half, off-centre laterally. -----
    thrower_x = rng.uniform(-10.0, 15.0) * attack_dir
    thrower_y = rng.uniform(-1.0, 1.0) + _stack_axis_offset(force_open_y) * 0.3
    pos[field.IDX_THROWER] = (thrower_x, thrower_y)

    # -- Reset handler: behind the disc (upfield-negative), slightly across. --
    pos[field.IDX_RESET] = (
        thrower_x - attack_dir * rng.uniform(4.0, 8.0),
        thrower_y + force_open_y * rng.uniform(-2.0, 4.0),
    )

    # -- Cutters: single file ahead of the disc, rank 1 nearest. -------------
    #
    # Not every cutter is a live option in a frozen frame. We pick 1-2 cutters
    # to be ACTIVELY cutting -- they have broken out of the stack line toward
    # open space and carry velocity -- while the rest wait in the stack. Which
    # cutters are active is chosen at random here (exogenous world state), NOT
    # derived from the completion outcome, so the receiver label does not leak
    # into the inputs. This is the crux that makes "the right read" vary: the
    # best receiver is usually one of the active cutters, and which one depends
    # on how well their defender reacted.
    stack_y = thrower_y + _stack_axis_offset(force_open_y)
    depth = rng.uniform(6.0, 10.0)  # gap from disc to front of stack
    spacing = rng.uniform(4.0, 6.0)  # gap between adjacent cutters

    ranks = np.arange(1, field.N_CUTTERS + 1)
    n_active = int(rng.choice((1, 2), p=(0.45, 0.55)))
    # Front cutters cut more often; weight the sampling toward low ranks.
    weights = 1.0 / ranks
    active = set(
        rng.choice(ranks, size=n_active, replace=False, p=weights / weights.sum()).tolist()
    )

    active_cutters = np.zeros(field.N_NODES, dtype=bool)
    for rank, node in enumerate(field.IDX_CUTTERS, start=1):
        along = depth + (rank - 1) * spacing
        base = np.array(
            [thrower_x + attack_dir * along, stack_y + rng.normal(0, 1.0)]
        )
        if rank in active:
            active_cutters[node] = True
            # Break out of the stack toward the OPEN side, either continuing
            # under (toward the disc) or going deep (away). Displace position
            # and set a matching velocity.
            go_deep = rng.random() < 0.5
            fwd = attack_dir * (1.0 if go_deep else -1.0)
            lateral = force_open_y  # attack the open side
            move = np.array([fwd, lateral], dtype=float)
            move /= np.linalg.norm(move)
            travel = rng.uniform(2.5, 6.0)
            pos[node] = base + move * travel
            vel[node] = move * rng.uniform(0.5, 1.0) * field.NOMINAL_SPEED
        else:
            pos[node] = base + rng.normal(0, 0.6, size=2)

    # Stash which cutters are active so defender reaction can use it. Not a
    # model input.
    _active = active_cutters

    # -- Defenders: man defence, positioned between assignment and the goal, --
    #    on the force side. Crucially, tightness depends on the threat level of
    #    the assignment: the front of the stack is the primary threat, so it is
    #    guarded tightest, and deeper/waiting cutters are given more cushion.
    #    This is what makes the "right read" genuinely vary -- if every defender
    #    played equally loose, the answer would always be "front cutter" and the
    #    task would be trivial. Real vertical-stack defence takes the front cut
    #    away and dares the offence to go elsewhere.
    for off_idx in range(field.N_OFFENCE):
        def_idx = field.defender_of(off_idx)
        assignment = pos[off_idx]
        if off_idx == field.IDX_THROWER:
            # The mark stands on the open side of the thrower, taking it away.
            pos[def_idx] = assignment + np.array(
                [attack_dir * 0.4, force_open_y * rng.uniform(0.6, 1.2)]
            )
            continue

        # Base cushion by role: reset gets a fair cushion (dump usually open),
        # cutters waiting in the stack are guarded tightly (they are the primary
        # threat), active cutters have separated by cutting.
        rank = field.STACK_RANK[off_idx]
        if off_idx == field.IDX_RESET:
            base_cushion = rng.uniform(1.5, 3.5)
        elif _active[off_idx]:
            # An active cutter has created some separation; how much depends on
            # whether the defender stayed with the cut. Trailing the cut is what
            # opens the throw.
            base_cushion = rng.uniform(1.0, 4.0)
            # A beaten defender trails BEHIND the cutter's motion.
            trail = -vel[off_idx]
            trail = trail / (np.linalg.norm(trail) + 1e-9)
            jitter = rng.normal(0, 0.5, size=2)
            pos[def_idx] = assignment + trail * base_cushion + jitter
            continue
        else:
            # Waiting in the stack: tight man coverage, front tighter than back.
            base_cushion = rng.uniform(0.4, 1.2) + 0.4 * (rank - 1)

        goal_shade = attack_dir * rng.uniform(0.3, 1.2)
        force_shade = force_open_y * rng.uniform(-0.3, 1.2)
        offset = np.array([goal_shade, force_shade])
        offset = offset / (np.linalg.norm(offset) + 1e-9) * base_cushion
        jitter = rng.normal(0, 0.4, size=2)
        pos[def_idx] = assignment + offset + jitter

    # -- Disc: co-located with the thrower, in hand. -------------------------
    pos[field.IDX_DISC] = pos[field.IDX_THROWER]

    # -- Wind: a light, mostly-along-the-field breeze. -----------------------
    wind = np.array([rng.normal(0, 3.0), rng.normal(0, 1.5)])

    return Frame(
        pos=pos,
        vel=vel,
        attack_dir=attack_dir,
        force_open_y=force_open_y,
        stall=stall,
        wind=wind,
    )


def sample_frames(n: int, rng: np.random.Generator) -> list[Frame]:
    """Draw `n` independent frames."""
    return [sample_frame(rng) for _ in range(n)]
