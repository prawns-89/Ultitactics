# Theory: TacticAI, mapped to the Ultimate vertical stack

This document explains *what* the code does and *why*, and maps every design
choice back to the TacticAI paper (Wang et al., "TacticAI: an AI assistant for
football tactics", *Nature Communications* 15, 1906, 2024 —
<https://www.nature.com/articles/s41467-024-45965-x>, arXiv:2310.10553). The PDF
is in the repo as `Tactic_AI.pdf`; section and equation references below point
into it.

Read this alongside the code. Each module has a long header docstring; this doc
is the connective tissue between them and the paper.

---

## 1. The one-paragraph idea

TacticAI represents a football corner kick as a **graph** — one node per player —
and runs a **graph neural network (GNN)** over it to answer tactical questions
(who receives the ball, will there be a shot, how to reposition). Its one
distinctive trick is **geometric deep learning**: it hard-codes the fact that a
mirror-image of the pitch is tactically the same situation, so the model does not
have to waste scarce data learning that symmetry. We do the same thing for the
**vertical stack** in Ultimate: one node per player (plus one for the disc), a
GNN over the graph, and the same reflection symmetry — adapted, because Ultimate's
symmetry is subtler than football's.

We have **no video**, so instead of learning from real outcomes we (a) generate
synthetic vertical-stack frames, (b) label them with a hand-written "which throw
is on" rule, and (c) check the GNN can *rediscover* that rule from raw
coordinates. If it can, the machinery is correct and only the data source needs
swapping later. This is the standard way to de-risk a modelling pipeline before
touching a CV stack.

---

## 2. The mapping, section by section

| TacticAI (football corner) | Here (Ultimate vertical stack) | Where in code |
|---|---|---|
| 22 player nodes | 14 player nodes + 1 disc node = 15 | `field.py` |
| No canonical player order | Cutters carry an explicit **stack rank** | `field.STACK_RANK`, `graph.py` |
| Fully-connected graph, `E = V×V` | Same: fully connected, relationships in edge *features* | `graph.py` |
| Edge feature: teammate/opponent (Table 2) | Same, plus is-marking + displacement + distance | `graph.py` |
| Node features: pos, vel, height, weight, possession | pos, vel, role, rank, stall, force; disc carries z | `graph.py` |
| Global features: receiver id, shot indicator | force side, wind, attack direction, stall | `graph.py`, `symmetry.py` |
| Symmetry group `D₂` (4 pitch reflections) | Same `D₂`, but the group acts on force/wind too | `symmetry.py` |
| GATv2 attention (Eqs. 3–4) | Same, dense implementation | `model.py` |
| `D₂`-invariance via frame averaging (Eq. 6) | Same | `model.py` |
| Receiver prediction (node classification) | Which candidate catches the next pass | `model.py`, `train.py` |
| Shot prediction (graph classification) | Per-candidate completion probability | `model.py`, `train.py` |
| Guided generation (VAE, node regression) | *Not yet built* — see §7 | — |

---

## 3. The graph (TacticAI §"Graph representation", Table 2)

**Nodes (15, fixed order).** See `field.py`. Two handlers (thrower + reset),
five stack cutters (rank 1 = front, nearest the disc), their seven defenders,
and the disc. The defender of offensive player `i` is always node `i + 7`, which
keeps all the index arithmetic trivial.

Three deliberate departures from TacticAI, each justified by the sport — **not**
by "porting", which the original report over-claimed:

- **A disc node.** A corner-kick ball is modelled by TacticAI with a single
  possession bit and nothing else (Methods, p.13). An *in-flight* disc is a real
  object with its own trajectory, so it earns a node with a z-height slot.
- **Stack rank as a node feature.** Football players have no canonical order, so
  TacticAI's permutation-invariant message passing is exactly right for it.
  Vertical-stack cutters *do* have a canonical order. Rank is a scalar in a
  feature slot — it is a *value*, not a tensor ordering — so it does **not**
  break permutation equivariance. (The original report's "a plain GNN port isn't
  right" framing was too strong here: adding rank as a feature is a plain GNN.)
- **A signed force feature and a wind vector.** These have a handedness, which is
  why the symmetry group has to be told how to transform them (§5).

**Edges (fully connected).** TacticAI sets `E = V×V` and puts a single one-hot
teammate/opponent flag on each edge. We keep the dense graph — at 15 nodes it
costs nothing — and carry four edge features: same-team, is-marking (the
assigned offence↔defender pair), and the normalised displacement + distance
between the two nodes. Typed *edge topology* (separate stack-adjacency / marking
/ lane edge sets, as the original report proposed) is a documented **upgrade
path**, not the base: it complicates the symmetry handling and is not needed to
reproduce the paper's result.

**One shared node schema.** Every node — thrower, cutter, defender, disc — uses
the *same* feature vector layout, with a role one-hot distinguishing them and
zero-fill for slots that do not apply (a defender has no stack rank). This
follows TacticAI, where all 22 players share a schema. (The original report
contradicted itself here — it claimed a shared schema but then tabulated
per-role features. `graph.py` resolves it with the union schema.)

---

## 4. Why a graph, and why attention (TacticAI §"Graph neural networks", Eqs. 2–4)

A GNN updates each node by aggregating **messages** from its neighbours. Stacking
a few layers lets information flow across the whole formation: a cutter's
embedding comes to reflect where its defender is, where the mark is, where the
open space is. This is why a graph beats feeding a flat list of coordinates to an
MLP — the relationships are the point (TacticAI, p.2: "these player relationships
may be of higher importance than the absolute distances").

**Attention (GATv2).** Not all neighbours matter equally. A defender standing in
your throwing lane matters more than one behind the disc. GATv2 (Eq. 4) learns a
per-pair weight so each node can attend to the neighbours that matter. We use the
same mechanism, implemented densely (a 15×15 attention matrix) instead of with
sparse scatter/gather — at this size dense is faster and far easier to read. See
`model.DenseGATv2Layer`.

---

## 5. Symmetry — the part worth getting right (TacticAI §"Geometric deep learning", Eqs. 5–8)

This is the heart of both papers, and the place the original vertical-stack
report went wrong, so it gets its own section.

**The claim.** A mirror image of the play is the same play. If you reflect every
player's position across the middle of the field, "who is open" does not change.
TacticAI's group is `D₂` = {identity, flip-x, flip-y, flip-both} — the four
reflections of the pitch (Methods, p.15). Baking this in means the model never
has to *learn* it, which matters enormously when data is scarce — and Ultimate
data is scarcer than football's.

**The subtlety the report missed.** TacticAI's invariance condition (their Eq. 5)
is

```
y( g·X , g·E , g·g )  =  y( X, E, g )     for every g in the group
```

The group element `g` acts on the **global feature vector `g` too** — not only on
coordinates. TacticAI itself relies on this: under a horizontal flip it negates
the x-component of every velocity, not just the x-position.

Once you accept that the group acts on *everything with a handedness*, the
report's conclusion — "a sideline reflection is invalid for Ultimate because it
flips open/break side" — dissolves. The sideline reflection is fine; you just
have to flip the **force label**, the **wind vector**, and the **attacking
direction** in lockstep with the coordinates. That is a feature-transform detail,
not an architectural difference. So we keep the **full `D₂` group**, exactly like
TacticAI, and `symmetry.apply_to_frame` transforms every handed quantity
together. `tests/test_symmetry.py` verifies that doing so leaves the ground-truth
completion label unchanged — the single most important invariant in the codebase.

**Two ways to use the group** (both in TacticAI):

- **Frame averaging (Eq. 6)** — run the plain model on all four reflected views
  and average. Simple, architecture-agnostic, gives *exact* invariance. This is
  what `model.py` implements.
- **Group convolution (Eq. 8)** — let the four views interact *inside* every
  layer. Stronger, fiddlier. Left as a documented extension; frame averaging is
  the right first choice.

Because a reflection flips coordinate signs but **permutes no nodes**, node *i* is
the same physical player in every view, so averaging the four view-embeddings
node-by-node is exactly correct (`model.encode_invariant`).

---

## 6. The learning tasks (TacticAI §"Benchmarking", Eqs. 9–11)

- **Receiver prediction** (TacticAI's headline task, node classification): given
  the frozen frame, which candidate catches the next pass. Reported as top-1 and
  **top-3** accuracy — note TacticAI's famous 0.782 number is *top-3*, and top-1
  is much lower; the original report omitted this, which matters for what you
  promise. See `train.evaluate`.
- **Completion probability** (TacticAI's shot head, re-aimed): for each candidate,
  the probability the throw is completed. A per-candidate sigmoid.

Both heads read the **frame-averaged** per-node embeddings at the candidate
nodes. The generative task (TacticAI's VAE, Eq. 11) is not built yet (§7).

### The label, and why it is synthetic

We have no footage, so the label comes from a hand-written rule in
`formation.completion_logits`. It scores each throw from three readable factors:

1. **Separation** — how open the receiver is from *their own* defender.
2. **Lane** — whether *any* defender is standing in the throwing lane (the worst
   blocker dominates, the way a thrower pulls the disc down at one poach).
3. **Side/gain** — downfield progress, rewarded on the open side and taxed on the
   break side (the whole point of a "force").

Crucially, the model is **not given** these — no lane-visibility feature, no
separation feature. It sees only raw positions, velocities, and roles, and must
*infer* blocking and separation itself. That is the actual test: if the GNN
recovers a rule built from lane geometry without being handed the lane, the
representation is doing real work.

### Avoiding two traps

- **Label leakage.** TacticAI freezes the frame at an *exogenous* moment — when
  the corner is taken, from the event stream, independent of who receives. The
  original report proposed freezing when "a cutter breaks from the stack", which
  *defines the frame by the outcome* and leaks the label. We avoid this: which
  cutters are actively cutting is chosen at random in `simulate.py` as part of
  the world state, never derived from the completion outcome.
- **Degenerate labels.** If every cutter just sits in the stack, the front cutter
  is always the best (shortest lane) and the task is trivially "throw front". Real
  vertical-stack defence takes the front cut away and dares you elsewhere. So the
  simulator marks the front of the stack tightest and pulls 1–2 cutters into
  *active cuts* with separation that depends on whether their defender reacted —
  which makes the correct read genuinely vary across the reset and the cutters.

---

## 7. What is deliberately not here yet

- **Computer vision.** No detection/tracking/homography. Synthetic data first, on
  purpose — known ground truth lets us verify the model before trusting a CV
  stack. When footage exists, replace `simulate.py` + the label in `formation.py`
  with real tracked frames and real outcomes; everything downstream is unchanged.
- **Generative guided placement** (TacticAI's VAE, Eq. 11): "if cutter rank 2 had
  cleared half a second earlier, completion to rank 1 rises by X%". This is the
  most novel Ultimate-specific idea (vertical-stack failures are usually *timing*
  failures, not spatial ones) and is the natural next build.
- **Group convolution** (Eq. 8), **typed edge topology**, and a **formation
  classifier** (vertical vs horizontal vs zone) that would gate this model in a
  multi-offence system — all upgrade paths, none needed for the base.

---

## 8. How to run

```bash
python -m ultitactics.train      # generate data, train, print held-out metrics
python -m pytest tests/ -q       # verify the invariants (esp. D₂ symmetry)
```

The number that matters is receiver top-1 **versus the majority-class baseline**
printed at the start of the run. Beating it means the model is reading geometry,
not just the label prior.
