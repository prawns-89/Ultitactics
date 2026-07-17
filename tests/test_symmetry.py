"""The load-bearing invariant: D_2 must leave the ground-truth label unchanged.

If any test here fails, the frame-averaged model is enforcing a symmetry the
data does not have, and the whole geometric-deep-learning argument collapses.
These are the tests to run first after any change to field/graph/symmetry.
"""

import numpy as np
import pytest

from ultitactics import field, simulate
from ultitactics.formation import best_receiver, completion_logits
from ultitactics.graph import build_graph
from ultitactics.symmetry import D2, apply_to_frame, apply_to_graph


@pytest.fixture
def frames():
    rng = np.random.default_rng(42)
    return simulate.sample_frames(50, rng)


def test_label_is_d2_invariant(frames):
    """Reflecting the whole play must not change the completion logits.

    This only holds because apply_to_frame flips force, attack_dir and wind in
    lockstep with coordinates -- the correction to the report's Section 3.4.
    """
    for frame in frames:
        base = completion_logits(frame)
        for g in D2:
            reflected = completion_logits(apply_to_frame(g, frame))
            np.testing.assert_allclose(base, reflected, atol=1e-9)


def test_best_receiver_index_is_invariant(frames):
    """The node index of the best receiver is unchanged by reflection.

    Reflections flip coordinates but permute no nodes, so the *identity* of the
    right read is preserved -- which is why frame averaging per node is valid.
    """
    for frame in frames:
        base = best_receiver(frame)
        for g in D2:
            assert best_receiver(apply_to_frame(g, frame)) == base


def test_graph_action_matches_frame_action(frames):
    """apply_to_graph(g, build_graph(f)) == build_graph(apply_to_frame(g, f)).

    Pins the tensor-level group action to the ground-truth frame-level action so
    the two cannot silently drift apart.
    """
    for frame in frames:
        gr = build_graph(frame)
        for g in D2:
            via_graph = apply_to_graph(g, gr)
            via_frame = build_graph(apply_to_frame(g, frame))
            np.testing.assert_allclose(via_graph.x, via_frame.x, atol=1e-9)
            np.testing.assert_allclose(via_graph.edge, via_frame.edge, atol=1e-9)


def test_group_closure(frames):
    """Composing two reflections lands on another group element (D_2 is closed)."""
    gr = build_graph(frames[0])
    for g in D2:
        for h in D2:
            sx, sy = g.sx * h.sx, g.sy * h.sy
            composed = apply_to_graph(g, apply_to_graph(h, gr))
            # find the single D2 element with these signs
            target = next(k for k in D2 if k.sx == sx and k.sy == sy)
            direct = apply_to_graph(target, gr)
            np.testing.assert_allclose(composed.x, direct.x, atol=1e-9)


def test_reflections_are_self_inverse(frames):
    """In D_2 every element is its own inverse (TacticAI note, p.16)."""
    gr = build_graph(frames[0])
    for g in D2:
        twice = apply_to_graph(g, apply_to_graph(g, gr))
        np.testing.assert_allclose(twice.x, gr.x, atol=1e-9)
        np.testing.assert_allclose(twice.edge, gr.edge, atol=1e-9)
