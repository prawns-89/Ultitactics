"""Ultitactics: a TacticAI-style tactical model for the Ultimate vertical stack.

A small, readable reproduction of the TacticAI approach (Wang et al., Nature
Communications 2024, https://www.nature.com/articles/s41467-024-45965-x)
targeting the Ultimate frisbee vertical stack instead of football corner kicks.

Read the modules in this order -- each builds on the previous:

    field.py       coordinate frame, node layout, constants  (READ FIRST)
    formation.py   the Frame dataclass + ground-truth completion rule
    simulate.py    synthetic vertical-stack frame generator
    graph.py       Frame -> node/edge feature tensors (TacticAI Table 2 analogue)
    symmetry.py    the D_2 reflection group, acting on the whole graph
    model.py       dense GATv2 with D_2 frame averaging (TacticAI Eqs. 3-6)
    train.py       end-to-end training + evaluation on synthetic data

There is no computer-vision pipeline here on purpose: synthetic data gives us
known ground truth, so we can verify the model rediscovers a rule we planted.
See docs/THEORY.md for the mapping to the paper, section by section.
"""

from . import field, formation, graph, simulate, symmetry  # noqa: F401

__all__ = ["field", "formation", "simulate", "graph", "symmetry"]
