"""End-to-end training and evaluation on synthetic vertical-stack data.

Run it directly:

    python -m ultitactics.train

What it demonstrates
--------------------
1. Generate N synthetic frames (`simulate.py`).
2. Label each with the ground-truth completion rule (`formation.py`) -- both a
   per-candidate completion probability and the single best receiver.
3. Featurise (`graph.py`), build the 4 D_2 views (`symmetry.py`).
4. Train the frame-averaged GATv2 (`model.py`) on two heads jointly:
     * completion  -- binary cross-entropy per candidate,
     * receiver    -- cross-entropy over candidates (TacticAI's receiver task).
5. Report held-out receiver top-1 / top-3 accuracy (TacticAI's headline metric
   is top-3) and completion calibration.

The point is not the accuracy number; it is that a model with the correct
symmetry, fed raw coordinates, recovers a rule it was never shown. If the
numbers are good, the plumbing is right and only the data source needs
replacing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import field, simulate
from .formation import completion_probs
from .graph import build_graph
from .model import VerticalStackGNN, stack_views


@dataclass
class Dataset:
    views_x: torch.Tensor  # [B, 4, N, ND]
    views_edge: torch.Tensor  # [B, 4, N, N, ED]
    completion: torch.Tensor  # [B, C]  ground-truth completion probability
    receiver: torch.Tensor  # [B]     index into candidates of the best receiver


def make_dataset(n: int, rng: np.random.Generator) -> Dataset:
    frames = simulate.sample_frames(n, rng)
    graphs = [build_graph(f) for f in frames]
    vx, ve = stack_views(graphs)

    probs = np.stack([completion_probs(f) for f in frames])  # [B, C]
    best = probs.argmax(axis=1)  # [B] index within candidates
    return Dataset(
        views_x=vx,
        views_edge=ve,
        completion=torch.from_numpy(probs).float(),
        receiver=torch.from_numpy(best).long(),
    )


def majority_baseline(ds: Dataset) -> float:
    """Top-1 you get by always predicting the most common receiver.

    The number to beat. If the model does not clear this, it has learned nothing
    beyond the label prior; clearing it means it is genuinely reading geometry.
    """
    counts = torch.bincount(ds.receiver, minlength=len(field.IDX_RECEIVER_CANDIDATES))
    return (counts.max() / counts.sum()).item()


def evaluate(model: VerticalStackGNN, ds: Dataset) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        out = model(ds.views_x, ds.views_edge)
        recv_logits = out["receiver"]  # [B, C]
        top1 = (recv_logits.argmax(1) == ds.receiver).float().mean().item()
        top3 = (
            recv_logits.topk(3, dim=1).indices == ds.receiver[:, None]
        ).any(1).float().mean().item()
        # Completion calibration: mean abs error vs the true probability.
        comp = torch.sigmoid(out["completion"])
        comp_mae = (comp - ds.completion).abs().mean().item()
    return {"top1": top1, "top3": top3, "completion_mae": comp_mae}


def train(
    n_train: int = 3000,
    n_test: int = 800,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 2e-3,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    train_ds = make_dataset(n_train, rng)
    test_ds = make_dataset(n_test, rng)
    baseline = majority_baseline(test_ds)
    print(f"majority-class baseline (top1 to beat): {baseline:.3f}\n")

    torch.manual_seed(seed)
    model = VerticalStackGNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Minibatch training. Full-batch gives only `epochs` weight updates, which is
    # far too few (TacticAI trains for 50,000 steps); minibatching turns each
    # epoch into n_train/batch_size steps so the model actually converges.
    n = train_ds.receiver.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            out = model(train_ds.views_x[idx], train_ds.views_edge[idx])
            # Receiver head: cross-entropy over candidates (TacticAI receiver task).
            loss_recv = F.cross_entropy(out["receiver"], train_ds.receiver[idx])
            # Completion head: BCE against the soft ground-truth probability.
            loss_comp = F.binary_cross_entropy_with_logits(
                out["completion"], train_ds.completion[idx]
            )
            (loss_recv + loss_comp).backward()
            opt.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            m = evaluate(model, test_ds)
            print(
                f"epoch {epoch:3d}  "
                f"top1 {m['top1']:.3f}  top3 {m['top3']:.3f}  "
                f"comp_mae {m['completion_mae']:.3f}"
            )

    return evaluate(model, test_ds)


if __name__ == "__main__":
    metrics = train()
    print("\nFinal held-out metrics:")
    for k, v in metrics.items():
        print(f"  {k:16} {v:.3f}")
    # Two reference points: random top-1 is 1/6 ~ 0.167, and the majority-class
    # baseline is printed at the top of the run. Beating the majority baseline is
    # the real bar -- it means the frame-averaged GNN is reading the geometry,
    # not just the label prior.
    print(f"\n(random top1 baseline = {1/len(field.IDX_RECEIVER_CANDIDATES):.3f})")
