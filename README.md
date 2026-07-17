# Ultitactics

A TacticAI-style tactical model for the **Ultimate frisbee vertical stack**.

This is a small, readable reproduction of the approach in **TacticAI: an AI
assistant for football tactics** (Wang et al., *Nature Communications* 15, 1906,
2024 — [paper](https://www.nature.com/articles/s41467-024-45965-x),
[arXiv:2310.10553](https://arxiv.org/abs/2310.10553)), retargeted from football
corner kicks to the Ultimate vertical stack. The paper PDF is in the repo as
`Tactic_AI.pdf`.

**Status: base theory, on synthetic data.** There is deliberately no computer
vision here yet. See [Scope](#scope) below.

---

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install torch numpy matplotlib pytest
.venv/bin/python -m pytest tests/ -q        # verify the invariants
.venv/bin/python -m ultitactics.train       # generate data, train, report metrics
```

## What this does

TacticAI turns a corner kick into a **graph** (one node per player), runs a
**graph neural network** over it, and predicts tactical outcomes — who receives
the ball, whether a shot follows. Its distinctive trick is **geometric deep
learning**: it hard-codes the fact that a mirrored pitch is the same situation,
so the model doesn't burn scarce data learning that.

We do the same for the vertical stack: 14 players + the disc as a 15-node graph,
a GATv2 over it, D₂ reflection symmetry baked in, and two heads — **which
candidate catches the next pass** and **how likely each throw is to complete**.

Because we have no footage, we generate synthetic vertical-stack frames, label
them with a hand-written "which throw is on" rule, and check the GNN can
**rediscover that rule from raw coordinates**. The model is never given the
lane-blocking or separation quantities the rule is built from — it has to infer
them. That's the test: if it works, the machinery is right, and only the data
source needs replacing.

## Read the code in this order

| File | What it is |
|---|---|
| [`ultitactics/field.py`](ultitactics/field.py) | **Read first.** Coordinate frame, node layout, constants. Everything depends on these conventions. |
| [`ultitactics/formation.py`](ultitactics/formation.py) | The `Frame` dataclass + the ground-truth completion rule. |
| [`ultitactics/simulate.py`](ultitactics/simulate.py) | Synthetic vertical-stack frame generator. |
| [`ultitactics/graph.py`](ultitactics/graph.py) | `Frame` → node/edge tensors. The analogue of TacticAI's Table 2. |
| [`ultitactics/symmetry.py`](ultitactics/symmetry.py) | The D₂ reflection group, acting on the **whole** graph. |
| [`ultitactics/model.py`](ultitactics/model.py) | Dense GATv2 + D₂ frame averaging (TacticAI Eqs. 3–6). |
| [`ultitactics/train.py`](ultitactics/train.py) | End-to-end training + evaluation. |

**[`docs/THEORY.md`](docs/THEORY.md) is the main document** — it maps every design
choice back to the paper, section by section, and explains the reasoning. Each
module also carries a long header docstring.

## Three things worth knowing

These came out of a review of the original feasibility report against the actual
paper, and they're the parts most likely to trip you up.

**1. The symmetry group is the full D₂ — the report was wrong to narrow it.**
The report argued that a left-right flip is invalid for Ultimate because it
swaps open and break side. But TacticAI's invariance condition (Eq. 5) has the
group acting on the **global features too**, not just coordinates — it already
negates velocity x-components under a horizontal flip. So the sideline reflection
is fine; you just flip the force label, the wind vector, and the attacking
direction in lockstep. That's a feature-transform detail, not an architectural
difference, and keeping all four group elements is free data efficiency. See
`symmetry.py` and `tests/test_symmetry.py`.

**2. TacticAI has no marking edges, and no ball node.** The report built on both.
TacticAI's graph is fully connected (`E = V×V`) with exactly one edge feature: a
teammate/opponent flag. It explicitly does *not* encode distances, and models the
ball with a single possession bit. Our typed edge features and disc node are
justified **additions for Ultimate** — not things ported across. Also: TacticAI's
headline 0.782 receiver accuracy is **top-3**, not top-1.

**3. Don't freeze the frame on the cut.** The report proposed starting a rep when
"a cutter's movement breaks from the stack" — that defines the frame by the
outcome and leaks the label into the input. TacticAI's freeze time is exogenous
(the moment the kick is taken). Here, which cutters are actively cutting is
sampled as part of the world state, never derived from the completion outcome.

## Scope

**Built:** the graph formalisation, the symmetry group, the GNN, the two
predictive heads, a synthetic world with known ground truth, and tests pinning
the invariants.

**Deliberately not built yet:**

- **Computer vision.** No detection / tracking / homography. Synthetic data
  first, so the model can be verified against known ground truth before anyone
  trusts a CV stack. To go real: replace `simulate.py` and the label in
  `formation.py` with tracked frames and observed outcomes — everything
  downstream is unchanged.
- **Guided generation** (TacticAI's VAE, Eq. 11) — "if cutter 2 had cleared half
  a second earlier, completion to cutter 1 rises by X%". The most novel
  Ultimate-specific idea, since vertical-stack failures are usually *timing*
  failures rather than positional ones. The natural next build.
- **Group convolution** (Eq. 8), **typed edge topology**, and a **formation
  classifier** (vertical vs horizontal vs zone) to gate the model in a
  multi-offence system. All upgrade paths; none needed for the base.

## Reference

Wang, Z., Veličković, P., Hennes, D. et al. *TacticAI: an AI assistant for
football tactics.* Nature Communications 15, 1906 (2024).
<https://www.nature.com/articles/s41467-024-45965-x>
