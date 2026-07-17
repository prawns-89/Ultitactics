"""The D_2 reflection group, acting correctly on the whole graph.

Background (TacticAI Methods, "Geometric deep learning", p.15)
--------------------------------------------------------------
TacticAI exploits G = D_2 = {id, <->, updown, both}, the four reflections of the
pitch. The invariance it relies on is (their Eq. 5):

        y( g.X , g.E , g.g )  =  y( X, E, g )      for every g in G

The part people miss -- and the part the report's Section 3.4 got wrong -- is
that g acts on the GLOBAL features g too, not only on coordinates. TacticAI
itself does this: under a horizontal reflection it negates the x-component of
every velocity, not just the x-position. Once you accept that the group acts on
*all* fields together, the "sideline reflection is invalid for Ultimate" claim
dissolves. A left-right flip is a perfectly good symmetry -- you just have to
flip everything that has a handedness to it: the force side, the wind, and the
attacking direction, all in lockstep.

So this module keeps the FULL D_2 group (four elements), matching TacticAI, and
defines the action carefully on every signed quantity.

The four group elements
------------------------
    identity   flip nothing
    flip_x     mirror the long axis: x -> -x. This swaps the two endzones, so
               the attacking direction flips.
    flip_y     mirror the sideline axis: y -> -y. This swaps open and break
               side, so the force flips (and the y-wind, and y-velocities).
    flip_xy    both.

For each, applying it to a frame must leave the ground-truth completion label
unchanged. `tests/test_symmetry.py` checks exactly that -- it is the single most
important invariant in the codebase, because if it fails, the group-averaged
model is enforcing a symmetry the data does not actually have.

Two ways to use the group (both standard, both in TacticAI):
  * FRAME AVERAGING (Eq. 6): average a plain model's output over the 4 views.
    Simple, architecture-agnostic. This is what `model.py` uses.
  * GROUP CONVOLUTION (Eq. 8): let the 4 views interact inside every layer.
    Stronger but fiddlier; left as a documented extension.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import field, graph
from .formation import Frame


@dataclass(frozen=True)
class Reflection:
    """One element of D_2, as a pair of per-axis signs."""

    name: str
    sx: int  # +1 or -1, applied to the x axis
    sy: int  # +1 or -1, applied to the y axis

    @property
    def sign(self) -> np.ndarray:
        return np.array([self.sx, self.sy], dtype=np.float64)


#: The four elements of D_2, in a fixed order used everywhere (indices matter
#: for frame averaging in model.py).
D2 = (
    Reflection("identity", +1, +1),
    Reflection("flip_x", -1, +1),
    Reflection("flip_y", +1, -1),
    Reflection("flip_xy", -1, -1),
)


def apply_to_frame(g: Reflection, frame: Frame) -> Frame:
    """Act with `g` on a raw Frame, transforming EVERY handed quantity.

    This is the ground truth of what the symmetry means; the tensor-level
    action below must agree with it.
    """
    return replace(
        frame,
        pos=frame.pos * g.sign,
        vel=frame.vel * g.sign,
        wind=frame.wind * g.sign,
        # x-flip swaps the endzones -> attacking direction flips.
        attack_dir=frame.attack_dir * g.sx,
        # y-flip swaps open/break -> force flips.
        force_open_y=frame.force_open_y * g.sy,
    )


def apply_to_graph(g: Reflection, gr: graph.Graph) -> graph.Graph:
    """Act with `g` on a featurised Graph, at the tensor level.

    Must produce exactly `build_graph(apply_to_frame(g, frame))`; the test suite
    pins this equivalence so the two code paths cannot silently drift apart.
    """
    x = gr.x.copy()
    x[:, graph.NODE_POS] *= g.sign
    x[:, graph.NODE_VEL] *= g.sign
    x[:, graph.NODE_FORCE] *= g.sy  # force is a y-handed scalar

    edge = gr.edge.copy()
    edge[:, :, graph.EDGE_DISP] *= g.sign  # displacement transforms like pos
    # same-team, is-marking, distance are all reflection-invariant.

    meta = dict(gr.meta)
    meta["attack_dir"] = gr.meta["attack_dir"] * g.sx
    meta["force_open_y"] = gr.meta["force_open_y"] * g.sy
    meta["wind"] = gr.meta["wind"] * g.sign
    return graph.Graph(x=x, edge=edge, meta=meta)


def all_views(gr: graph.Graph) -> list[graph.Graph]:
    """The 4-element orbit of `gr` under D_2, in canonical `D2` order."""
    return [apply_to_graph(g, gr) for g in D2]
