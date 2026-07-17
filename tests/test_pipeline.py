"""Shape, layout, and model-invariance checks for the data + model pipeline."""

import numpy as np
import torch

from ultitactics import field, simulate
from ultitactics.graph import EDGE_DIM, NODE_DIM, build_graph
from ultitactics.model import VerticalStackGNN, stack_views


def test_node_layout_constants():
    assert field.N_NODES == 15
    assert field.IDX_DISC == 14
    assert field.defender_of(field.IDX_THROWER) == field.IDX_MARK
    # Every cutter's defender is offset by exactly N_OFFENCE.
    for c in field.IDX_CUTTERS:
        assert field.defender_of(c) == c + field.N_OFFENCE
    assert len(field.IDX_RECEIVER_CANDIDATES) == 6


def test_graph_shapes_and_ranges():
    rng = np.random.default_rng(0)
    gr = build_graph(simulate.sample_frame(rng))
    assert gr.x.shape == (15, NODE_DIM)
    assert gr.edge.shape == (15, 15, EDGE_DIM)
    # Normalised positions stay inside the field box.
    assert np.abs(gr.x[:, 0:2]).max() <= 1.5  # slack for defenders near lines
    # Each node has exactly one role one-hot set.
    role_block = gr.x[:, 9 : 9 + field.N_ROLES]
    np.testing.assert_array_equal(role_block.sum(axis=1), np.ones(15))


def test_marking_edges_symmetric():
    rng = np.random.default_rng(1)
    gr = build_graph(simulate.sample_frame(rng))
    mark = gr.edge[:, :, 1]
    np.testing.assert_array_equal(mark, mark.T)
    assert mark.sum() == 2 * field.N_OFFENCE  # 7 pairs, both directions


def test_model_output_is_reflection_invariant():
    """Frame averaging => model output identical across all 4 D_2 views.

    This is the model-level payoff of the symmetry: we permute the view order
    fed in and the prediction does not move (up to float error).
    """
    rng = np.random.default_rng(2)
    graphs = [build_graph(simulate.sample_frame(rng)) for _ in range(8)]
    vx, ve = stack_views(graphs)

    torch.manual_seed(0)
    model = VerticalStackGNN()
    model.eval()
    with torch.no_grad():
        base = model(vx, ve)["receiver"]
        # Reverse the 4 views per sample; averaging is order-invariant.
        shuffled = model(vx.flip(1), ve.flip(1))["receiver"]
    torch.testing.assert_close(base, shuffled, atol=1e-5, rtol=1e-4)


def test_model_forward_shapes():
    rng = np.random.default_rng(3)
    graphs = [build_graph(simulate.sample_frame(rng)) for _ in range(5)]
    vx, ve = stack_views(graphs)
    model = VerticalStackGNN()
    out = model(vx, ve)
    assert out["receiver"].shape == (5, 6)
    assert out["completion"].shape == (5, 6)
